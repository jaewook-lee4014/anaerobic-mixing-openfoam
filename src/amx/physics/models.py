"""Refactored physics models with improved structure."""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Tuple

import numpy as np

from amx.core.base import BaseModel
from amx.core.interfaces import FluidModel, MixingModel, RheologyModel, TurbulenceModel


class FluidType(Enum):
    """Fluid type enumeration."""
    WATER = "water"
    SLUDGE = "sludge"
    CUSTOM = "custom"


class TurbulenceModelType(Enum):
    """Turbulence model types."""
    K_EPSILON = "k-epsilon"
    K_OMEGA = "k-omega"
    K_OMEGA_SST = "k-omega-sst"
    REALIZABLE_K_EPSILON = "realizable-k-epsilon"


class RheologyModelType(Enum):
    """Rheology model types."""
    NEWTONIAN = "newtonian"
    POWER_LAW = "power_law"
    BINGHAM = "bingham"
    HERSCHEL_BULKLEY = "herschel_bulkley"
    CASSON = "casson"


@dataclass
class FluidProperties(BaseModel):
    """Fluid properties container."""
    temperature: float  # [K]
    density: float  # [kg/m³]
    dynamic_viscosity: float  # [Pa·s]
    specific_heat: float = 4180.0  # [J/kg·K]
    thermal_conductivity: float = 0.6  # [W/m·K]
    
    @property
    def kinematic_viscosity(self) -> float:
        """Calculate kinematic viscosity [m²/s]."""
        return self.dynamic_viscosity / self.density
    
    @property
    def prandtl_number(self) -> float:
        """Calculate Prandtl number."""
        return self.specific_heat * self.dynamic_viscosity / self.thermal_conductivity


class NewtonianFluid(FluidModel):
    """Newtonian fluid model."""
    
    def __init__(self, properties: FluidProperties):
        """
        Initialize Newtonian fluid.
        
        Args:
            properties: Fluid properties
        """
        self.props = properties
    
    def density(self, temperature: float, pressure: float = 101325) -> float:
        """Calculate density with temperature correction."""
        # Simplified temperature correction
        T_ref = self.props.temperature
        rho_ref = self.props.density
        beta = 0.0002  # Thermal expansion coefficient [1/K]
        
        return rho_ref * (1 - beta * (temperature - T_ref))
    
    def viscosity(self, temperature: float, shear_rate: Optional[float] = None) -> float:
        """Calculate viscosity (constant for Newtonian)."""
        # Arrhenius temperature dependence
        T_ref = self.props.temperature
        mu_ref = self.props.dynamic_viscosity
        E_a = 20000  # Activation energy [J/mol]
        R = 8.314  # Gas constant [J/mol·K]
        
        return mu_ref * np.exp(E_a / R * (1/temperature - 1/T_ref))
    
    def properties(self, temperature: float) -> Dict[str, float]:
        """Get all properties at given temperature."""
        return {
            "density": self.density(temperature),
            "dynamic_viscosity": self.viscosity(temperature),
            "kinematic_viscosity": self.viscosity(temperature) / self.density(temperature),
            "temperature": temperature,
        }


class NonNewtonianFluid(FluidModel, RheologyModel):
    """Non-Newtonian fluid model for sludge."""
    
    def __init__(self, total_solids: float, temperature: float = 308.15,
                 model_type: RheologyModelType = RheologyModelType.HERSCHEL_BULKLEY):
        """
        Initialize non-Newtonian fluid.
        
        Args:
            total_solids: Total solids content [%]
            temperature: Temperature [K]
            model_type: Rheological model type
        """
        self.TS = total_solids
        self.T = temperature
        self.model_type = model_type
        self._calculate_parameters()
    
    def _calculate_parameters(self):
        """Calculate model parameters based on TS content."""
        # Herschel-Bulkley parameters (from literature correlations)
        self.tau_0 = np.exp(-3.2 + 0.75 * self.TS**0.8)  # Yield stress [Pa]
        self.K = 0.05 * self.TS**1.5  # Consistency index
        self.n = max(0.3, min(0.8, 0.6 - 0.05 * self.TS))  # Flow index
        
        # Density correlation
        rho_water = 999.8 - 0.0642 * (self.T - 273.15)
        self.rho = rho_water * (1 + 0.0075 * self.TS)
    
    def density(self, temperature: float, pressure: float = 101325) -> float:
        """Calculate sludge density."""
        # Update with temperature
        T_C = temperature - 273.15
        rho_water = 999.8 - 0.0642 * T_C
        return rho_water * (1 + 0.0075 * self.TS)
    
    def viscosity(self, temperature: float, shear_rate: Optional[float] = None) -> float:
        """Calculate apparent viscosity."""
        if shear_rate is None or shear_rate < 1e-6:
            return 1000.0  # Maximum viscosity at zero shear
        
        return self.apparent_viscosity(shear_rate)
    
    def apparent_viscosity(self, shear_rate: float) -> float:
        """Calculate apparent viscosity for given shear rate."""
        if self.model_type == RheologyModelType.NEWTONIAN:
            return 0.001 * (1 + 0.5 * self.TS)
        
        elif self.model_type == RheologyModelType.HERSCHEL_BULKLEY:
            if shear_rate < 1e-6:
                return 1000.0
            mu_app = self.tau_0 / shear_rate + self.K * shear_rate**(self.n - 1)
            return min(mu_app, 1000.0)
        
        elif self.model_type == RheologyModelType.POWER_LAW:
            return self.K * shear_rate**(self.n - 1)
        
        elif self.model_type == RheologyModelType.BINGHAM:
            if shear_rate < 1e-6:
                return 1000.0
            mu_p = 0.001 * (1 + self.TS)
            return self.tau_0 / shear_rate + mu_p
        
        else:
            raise ValueError(f"Unknown rheology model: {self.model_type}")
    
    def shear_stress(self, shear_rate: float) -> float:
        """Calculate shear stress."""
        if self.model_type == RheologyModelType.HERSCHEL_BULKLEY:
            return self.tau_0 + self.K * shear_rate**self.n
        elif self.model_type == RheologyModelType.POWER_LAW:
            return self.K * shear_rate**self.n
        elif self.model_type == RheologyModelType.BINGHAM:
            mu_p = 0.001 * (1 + self.TS)
            return self.tau_0 + mu_p * shear_rate
        else:
            return self.apparent_viscosity(shear_rate) * shear_rate
    
    def model_parameters(self) -> Dict[str, float]:
        """Get rheological model parameters."""
        params = {
            "model_type": self.model_type.value,
            "total_solids": self.TS,
            "temperature": self.T,
            "density": self.rho,
        }
        
        if self.model_type in [RheologyModelType.HERSCHEL_BULKLEY, RheologyModelType.BINGHAM]:
            params["yield_stress"] = self.tau_0
        
        if self.model_type in [RheologyModelType.HERSCHEL_BULKLEY, RheologyModelType.POWER_LAW]:
            params["consistency_index"] = self.K
            params["flow_index"] = self.n
        
        return params
    
    def properties(self, temperature: float) -> Dict[str, float]:
        """Get all properties."""
        return {
            "density": self.density(temperature),
            "model_parameters": self.model_parameters(),
        }


class StandardKEpsilon(TurbulenceModel):
    """Standard k-epsilon turbulence model."""
    
    # Model constants
    C_mu = 0.09
    C_1e = 1.44
    C_2e = 1.92
    sigma_k = 1.0
    sigma_e = 1.3
    
    def __init__(self, kinematic_viscosity: float):
        """
        Initialize turbulence model.
        
        Args:
            kinematic_viscosity: Kinematic viscosity [m²/s]
        """
        self.nu = kinematic_viscosity
    
    def eddy_viscosity(self, k: float, epsilon: float) -> float:
        """Calculate turbulent eddy viscosity."""
        if epsilon <= 0 or k <= 0:
            return 0.0
        return self.C_mu * k**2 / epsilon
    
    def production_rate(self, grad_u: np.ndarray, k: float = None, 
                       epsilon: float = None) -> float:
        """Calculate turbulence production rate."""
        # Strain rate tensor
        S = 0.5 * (grad_u + grad_u.T)
        S_mag = np.sqrt(2 * np.sum(S * S))
        
        # Get eddy viscosity
        if k is not None and epsilon is not None:
            nu_t = self.eddy_viscosity(k, epsilon)
        else:
            raise ValueError("k and epsilon required for production calculation")
        
        return nu_t * S_mag**2
    
    def dissipation_rate(self, k: float, length_scale: float) -> float:
        """Calculate dissipation rate."""
        if length_scale <= 0 or k <= 0:
            return 1e-10
        return self.C_mu**(3/4) * k**(3/2) / length_scale
    
    def initial_conditions(self, U_ref: float, L_ref: float, 
                         intensity: float = 0.05) -> Dict[str, float]:
        """Calculate initial turbulence conditions."""
        k = 1.5 * (U_ref * intensity)**2
        l = 0.07 * L_ref
        epsilon = self.C_mu**(3/4) * k**(3/2) / l
        nu_t = self.eddy_viscosity(k, epsilon)
        
        return {
            'k': k,
            'epsilon': epsilon,
            'nu_t': nu_t,
            'length_scale': l,
            'time_scale': k / epsilon if epsilon > 0 else float('inf'),
        }


class ComprehensiveMixingModel(MixingModel):
    """Comprehensive mixing model implementation."""
    
    def __init__(self, fluid: FluidModel):
        """
        Initialize mixing model.
        
        Args:
            fluid: Fluid model
        """
        self.fluid = fluid
    
    def mixing_time(self, volume: float, flow_rate: float, 
                   method: str = "circulation") -> float:
        """Calculate mixing time using specified method."""
        if method == "circulation":
            # 4 circulation times
            return 4.0 * volume / flow_rate
        elif method == "grenville":
            # Grenville & Tilton correlation
            return 5.4 * volume / flow_rate
        elif method == "fossett":
            # Fossett & Prosser correlation
            return 4.0 * volume / flow_rate
        else:
            # Default
            return 4.0 * volume / flow_rate
    
    def velocity_gradient(self, power: float, volume: float, viscosity: float) -> float:
        """Calculate Camp velocity gradient (G-value)."""
        if power <= 0 or volume <= 0 or viscosity <= 0:
            return 0.0
        return np.sqrt(power / (viscosity * volume))
    
    def dead_zone_fraction(self, velocity_field: np.ndarray, 
                          threshold: float = 0.05) -> float:
        """Calculate fraction of dead zones."""
        if isinstance(velocity_field, np.ndarray):
            if velocity_field.ndim == 2:  # Vector field
                vel_mag = np.linalg.norm(velocity_field, axis=1)
            else:
                vel_mag = velocity_field
            
            dead_zones = vel_mag < threshold
            return np.sum(dead_zones) / len(vel_mag)
        return 0.0
    
    def mixing_efficiency(self, actual: float, theoretical: float) -> float:
        """Calculate mixing efficiency."""
        if actual <= 0:
            return 0.0
        return min(theoretical / actual, 1.0)
    
    def camp_number(self, G: float, time: float) -> float:
        """Calculate Camp number (Gt)."""
        return G * time
    
    def peclet_number(self, velocity: float, length: float, 
                     diffusivity: Optional[float] = None) -> float:
        """Calculate Peclet number."""
        if diffusivity is None:
            diffusivity = 0.1 * velocity * length  # Turbulent approximation
        
        if diffusivity <= 0:
            return float('inf')
        
        return velocity * length / diffusivity
    
    def reynolds_number(self, velocity: float, length: float, 
                       viscosity: float, density: float) -> float:
        """Calculate Reynolds number."""
        if viscosity <= 0:
            return float('inf')
        return density * velocity * length / viscosity