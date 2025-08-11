"""Rheological models for anaerobic digester sludge.

Based on experimental data and correlations from literature for
accurate representation of non-Newtonian behavior in digester sludge.
"""

import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class RheologyModel(Enum):
    """Available rheology models for sludge."""
    NEWTONIAN = "newtonian"
    POWER_LAW = "power_law"
    BINGHAM = "bingham"
    HERSCHEL_BULKLEY = "herschel_bulkley"
    CASSON = "casson"


@dataclass
class SludgeProperties:
    """
    Physical and rheological properties of anaerobic digester sludge.
    
    References:
    - Eshtiaghi et al. (2013). Rheological characterisation of municipal sludge.
    - Seyssiecq et al. (2003). State-of-the-art: rheological characterisation of wastewater treatment sludge.
    - Baudez (2008). Physical aging and thixotropy in sludge rheology.
    """
    
    # Basic properties
    temperature: float  # Temperature [K]
    total_solids: float  # Total solids content [%]
    volatile_solids: float  # Volatile solids content [% of TS]
    
    # Derived properties
    density: float = None  # Will be calculated if not provided
    
    def __post_init__(self):
        """Calculate derived properties."""
        if self.density is None:
            self.density = self.calculate_density()
    
    def calculate_density(self) -> float:
        """
        Calculate sludge density based on total solids content.
        
        Correlation from Lotito et al. (1997):
        ρ_sludge = ρ_water * (1 + 0.0075 * TS)
        
        Returns:
            Density [kg/m³]
        """
        # Water density at temperature (simplified correlation)
        T_C = self.temperature - 273.15
        rho_water = 999.8 - 0.0642 * T_C - 0.0085 * T_C**2 * 1e-3
        
        # Sludge density
        return rho_water * (1 + 0.0075 * self.total_solids)
    
    def viscosity_power_law(self, shear_rate: float) -> float:
        """
        Calculate apparent viscosity using power-law model.
        
        μ_app = K * γ^(n-1)
        
        where K and n are functions of TS content.
        
        Args:
            shear_rate: Shear rate [1/s]
            
        Returns:
            Apparent viscosity [Pa·s]
        """
        # Correlation for consistency index K (Slatter, 1997)
        # K = exp(a + b*TS) where a, b are temperature-dependent
        T_C = self.temperature - 273.15
        
        if T_C < 20:
            a, b = -6.5, 0.35
        elif T_C < 35:
            a, b = -6.8, 0.32
        else:
            a, b = -7.1, 0.30
        
        K = np.exp(a + b * self.total_solids)
        
        # Correlation for flow behavior index n (Moeller & Torres, 1997)
        # n decreases with increasing TS (more shear-thinning)
        n = 1.0 - 0.15 * np.log10(self.total_solids + 1)
        n = max(0.2, min(1.0, n))  # Bound between 0.2 and 1.0
        
        if shear_rate <= 0:
            # Return high viscosity at zero shear (yield behavior)
            return K * 1000
        
        return K * shear_rate**(n - 1)
    
    def viscosity_herschel_bulkley(self, shear_rate: float) -> float:
        """
        Calculate apparent viscosity using Herschel-Bulkley model.
        
        τ = τ_0 + K * γ^n
        μ_app = τ_0/γ + K * γ^(n-1)
        
        Most accurate for digested sludge with yield stress.
        
        Args:
            shear_rate: Shear rate [1/s]
            
        Returns:
            Apparent viscosity [Pa·s]
        """
        # Yield stress correlation (Eshtiaghi et al., 2013)
        # τ_0 = exp(a + b*TS^c)
        tau_0 = np.exp(-3.2 + 0.75 * self.total_solids**0.8)
        
        # Consistency and flow indices
        K = 0.05 * self.total_solids**1.5
        n = 0.6 - 0.05 * self.total_solids
        n = max(0.3, min(0.8, n))
        
        if shear_rate <= 1e-6:
            # Return very high viscosity below yield
            return 1000.0
        
        # Apparent viscosity
        mu_app = tau_0 / shear_rate + K * shear_rate**(n - 1)
        
        # Limit maximum viscosity
        return min(mu_app, 1000.0)
    
    def viscosity_casson(self, shear_rate: float) -> float:
        """
        Calculate apparent viscosity using Casson model.
        
        √τ = √τ_0 + √(μ_∞ * γ)
        
        Good for moderate TS content (2-6%).
        
        Args:
            shear_rate: Shear rate [1/s]
            
        Returns:
            Apparent viscosity [Pa·s]
        """
        # Casson yield stress
        tau_0 = 0.5 * self.total_solids**1.2
        
        # Infinite shear viscosity
        mu_inf = 0.001 * (1 + 0.5 * self.total_solids)
        
        if shear_rate <= 1e-6:
            return 1000.0
        
        # Casson model
        sqrt_tau = np.sqrt(tau_0) + np.sqrt(mu_inf * shear_rate)
        tau = sqrt_tau**2
        
        return tau / shear_rate
    
    def get_viscosity(self, shear_rate: float, 
                     model: RheologyModel = RheologyModel.HERSCHEL_BULKLEY) -> float:
        """
        Get viscosity using specified rheological model.
        
        Args:
            shear_rate: Shear rate [1/s]
            model: Rheological model to use
            
        Returns:
            Apparent viscosity [Pa·s]
        """
        if model == RheologyModel.NEWTONIAN:
            # Simple correlation for low TS
            return 0.001 * (1 + 0.5 * self.total_solids)
        
        elif model == RheologyModel.POWER_LAW:
            return self.viscosity_power_law(shear_rate)
        
        elif model == RheologyModel.HERSCHEL_BULKLEY:
            return self.viscosity_herschel_bulkley(shear_rate)
        
        elif model == RheologyModel.CASSON:
            return self.viscosity_casson(shear_rate)
        
        elif model == RheologyModel.BINGHAM:
            # Simplified Bingham (special case of H-B with n=1)
            tau_0 = 0.5 * self.total_solids**1.2
            mu_p = 0.001 * (1 + self.total_solids)
            
            if shear_rate <= 1e-6:
                return 1000.0
            
            return tau_0 / shear_rate + mu_p
        
        else:
            raise ValueError(f"Unknown rheology model: {model}")
    
    def mixing_reynolds(self, velocity: float, length: float, 
                       shear_rate: Optional[float] = None) -> float:
        """
        Calculate Reynolds number for non-Newtonian fluid.
        
        Re_eff = ρ * V * L / μ_eff
        
        Args:
            velocity: Characteristic velocity [m/s]
            length: Characteristic length [m]
            shear_rate: Representative shear rate [1/s]
            
        Returns:
            Effective Reynolds number [-]
        """
        if shear_rate is None:
            # Estimate shear rate from velocity and length
            shear_rate = velocity / length
        
        mu_eff = self.get_viscosity(shear_rate)
        
        return self.density * velocity * length / mu_eff
    
    def power_consumption_factor(self) -> float:
        """
        Calculate power consumption correction factor for non-Newtonian behavior.
        
        Based on Metzner-Otto correlation for mixing.
        
        Returns:
            Power factor [-]
        """
        # Higher TS requires more power due to yield stress and viscosity
        base_factor = 1.0
        
        # Yield stress contribution
        if self.total_solids > 2:
            base_factor *= (1 + 0.1 * (self.total_solids - 2))
        
        # Shear-thinning benefit (reduces power at high shear)
        if self.total_solids > 3:
            base_factor *= 0.95
        
        return base_factor
    
    def settling_velocity(self, particle_size: float, 
                         particle_density: float) -> float:
        """
        Calculate particle settling velocity in sludge.
        
        Uses modified Stokes law for non-Newtonian fluids.
        
        Args:
            particle_size: Particle diameter [m]
            particle_density: Particle density [kg/m³]
            
        Returns:
            Settling velocity [m/s]
        """
        g = 9.81
        
        # Estimate shear rate around particle
        shear_rate = 1.0  # Typical value for settling
        mu_eff = self.get_viscosity(shear_rate)
        
        # Check if yield stress prevents settling
        tau_0 = 0.5 * self.total_solids**1.2  # Yield stress
        critical_size = 2 * tau_0 / (g * abs(particle_density - self.density))
        
        if particle_size < critical_size:
            return 0.0  # Particle won't settle
        
        # Modified Stokes velocity
        v_stokes = (particle_density - self.density) * g * particle_size**2 / (18 * mu_eff)
        
        # Correction for non-Newtonian effects
        Re_p = self.density * abs(v_stokes) * particle_size / mu_eff
        if Re_p > 1:
            # Transition regime correction
            v_stokes *= (1 + 0.15 * Re_p**0.687)**(-1)
        
        return v_stokes


class TemperatureEffects:
    """
    Temperature-dependent properties and effects in anaerobic digestion.
    """
    
    @staticmethod
    def viscosity_temperature_correction(T: float, T_ref: float = 308.15) -> float:
        """
        Calculate temperature correction factor for viscosity.
        
        Based on Arrhenius-type relationship.
        
        Args:
            T: Temperature [K]
            T_ref: Reference temperature [K]
            
        Returns:
            Correction factor [-]
        """
        # Activation energy for viscous flow (typical for sludge)
        E_a = 20000  # J/mol
        R = 8.314  # J/(mol·K)
        
        return np.exp(E_a / R * (1/T - 1/T_ref))
    
    @staticmethod
    def biogas_production_rate(T: float, operation_mode: str = "mesophilic") -> float:
        """
        Calculate relative biogas production rate vs temperature.
        
        Args:
            T: Temperature [K]
            operation_mode: "mesophilic" or "thermophilic"
            
        Returns:
            Relative production rate [-]
        """
        T_C = T - 273.15
        
        if operation_mode == "mesophilic":
            # Optimal at 35°C
            T_opt = 35
            if 30 <= T_C <= 40:
                return 1.0 - 0.05 * abs(T_C - T_opt)
            else:
                return max(0, 1.0 - 0.15 * abs(T_C - T_opt))
        
        elif operation_mode == "thermophilic":
            # Optimal at 55°C
            T_opt = 55
            if 50 <= T_C <= 60:
                return 1.0 - 0.03 * abs(T_C - T_opt)
            else:
                return max(0, 1.0 - 0.10 * abs(T_C - T_opt))
        
        else:
            raise ValueError(f"Unknown operation mode: {operation_mode}")
    
    @staticmethod
    def mixing_energy_requirement(T: float, TS: float) -> float:
        """
        Calculate relative mixing energy requirement.
        
        Higher temperature reduces viscosity, reducing energy need.
        
        Args:
            T: Temperature [K]
            TS: Total solids [%]
            
        Returns:
            Relative energy factor [-]
        """
        T_C = T - 273.15
        
        # Base energy at 35°C
        base_factor = 1.0
        
        # Temperature effect (lower T = higher viscosity = more energy)
        temp_factor = 1.0 + 0.02 * (35 - T_C)
        
        # TS effect (exponential increase)
        ts_factor = np.exp(0.15 * (TS - 3))
        
        return base_factor * temp_factor * ts_factor


class MixingRegimeAnalysis:
    """
    Analyze mixing regime and performance for anaerobic digesters.
    """
    
    @staticmethod
    def regime_classification(Re: float, Fr: float = None) -> str:
        """
        Classify mixing regime based on dimensionless numbers.
        
        Args:
            Re: Reynolds number
            Fr: Froude number (optional)
            
        Returns:
            Mixing regime description
        """
        if Re < 10:
            regime = "Laminar/Creeping flow"
        elif Re < 100:
            regime = "Transitional laminar"
        elif Re < 10000:
            regime = "Transitional turbulent"
        else:
            regime = "Fully turbulent"
        
        if Fr is not None:
            if Fr < 0.1:
                regime += " (Viscous dominated)"
            elif Fr < 1.0:
                regime += " (Mixed viscous-inertial)"
            else:
                regime += " (Inertia dominated)"
        
        return regime
    
    @staticmethod
    def dead_zone_prediction(velocity: float, min_velocity: float = 0.05) -> bool:
        """
        Predict if location is in dead zone based on velocity.
        
        Args:
            velocity: Local velocity magnitude [m/s]
            min_velocity: Minimum velocity threshold [m/s]
            
        Returns:
            True if in dead zone
        """
        return velocity < min_velocity
    
    @staticmethod
    def short_circuiting_index(mean_residence_time: float, 
                              actual_residence_time: float) -> float:
        """
        Calculate short-circuiting index.
        
        SI = t_actual / t_theoretical
        
        SI < 0.3: Severe short-circuiting
        SI = 0.3-0.5: Moderate short-circuiting
        SI = 0.5-0.7: Mild short-circuiting
        SI > 0.7: Good mixing
        
        Args:
            mean_residence_time: Theoretical mean residence time [s]
            actual_residence_time: Measured residence time (t_10 or t_50) [s]
            
        Returns:
            Short-circuiting index [-]
        """
        if mean_residence_time <= 0:
            return 0.0
        
        return actual_residence_time / mean_residence_time