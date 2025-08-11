"""Interface definitions for physics models."""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

import numpy as np


class FluidModel(ABC):
    """Interface for fluid property models."""
    
    @abstractmethod
    def density(self, temperature: float, pressure: float = 101325) -> float:
        """
        Calculate fluid density.
        
        Args:
            temperature: Temperature [K]
            pressure: Pressure [Pa]
            
        Returns:
            Density [kg/m³]
        """
        pass
    
    @abstractmethod
    def viscosity(self, temperature: float, shear_rate: Optional[float] = None) -> float:
        """
        Calculate fluid viscosity.
        
        Args:
            temperature: Temperature [K]
            shear_rate: Shear rate [1/s] for non-Newtonian fluids
            
        Returns:
            Dynamic viscosity [Pa·s]
        """
        pass
    
    @abstractmethod
    def properties(self, temperature: float) -> Dict[str, float]:
        """
        Get all fluid properties.
        
        Args:
            temperature: Temperature [K]
            
        Returns:
            Dictionary of properties
        """
        pass


class TurbulenceModel(ABC):
    """Interface for turbulence models."""
    
    @abstractmethod
    def eddy_viscosity(self, k: float, epsilon: float) -> float:
        """
        Calculate turbulent eddy viscosity.
        
        Args:
            k: Turbulent kinetic energy [m²/s²]
            epsilon: Dissipation rate [m²/s³]
            
        Returns:
            Eddy viscosity [m²/s]
        """
        pass
    
    @abstractmethod
    def production_rate(self, grad_u: np.ndarray, **kwargs) -> float:
        """
        Calculate turbulence production rate.
        
        Args:
            grad_u: Velocity gradient tensor [1/s]
            **kwargs: Additional model parameters
            
        Returns:
            Production rate [m²/s³]
        """
        pass
    
    @abstractmethod
    def dissipation_rate(self, k: float, length_scale: float) -> float:
        """
        Calculate turbulent dissipation rate.
        
        Args:
            k: Turbulent kinetic energy [m²/s²]
            length_scale: Turbulent length scale [m]
            
        Returns:
            Dissipation rate [m²/s³]
        """
        pass
    
    @abstractmethod
    def initial_conditions(self, U_ref: float, L_ref: float, 
                         intensity: float = 0.05) -> Dict[str, float]:
        """
        Calculate initial turbulence conditions.
        
        Args:
            U_ref: Reference velocity [m/s]
            L_ref: Reference length [m]
            intensity: Turbulence intensity [-]
            
        Returns:
            Dictionary with k, epsilon, nu_t
        """
        pass


class RheologyModel(ABC):
    """Interface for rheological models."""
    
    @abstractmethod
    def apparent_viscosity(self, shear_rate: float) -> float:
        """
        Calculate apparent viscosity.
        
        Args:
            shear_rate: Shear rate [1/s]
            
        Returns:
            Apparent viscosity [Pa·s]
        """
        pass
    
    @abstractmethod
    def shear_stress(self, shear_rate: float) -> float:
        """
        Calculate shear stress.
        
        Args:
            shear_rate: Shear rate [1/s]
            
        Returns:
            Shear stress [Pa]
        """
        pass
    
    @abstractmethod
    def model_parameters(self) -> Dict[str, float]:
        """
        Get model parameters.
        
        Returns:
            Dictionary of model parameters
        """
        pass


class MixingModel(ABC):
    """Interface for mixing models."""
    
    @abstractmethod
    def mixing_time(self, volume: float, flow_rate: float, **kwargs) -> float:
        """
        Calculate mixing time.
        
        Args:
            volume: Tank volume [m³]
            flow_rate: Flow rate [m³/s]
            **kwargs: Additional parameters
            
        Returns:
            Mixing time [s]
        """
        pass
    
    @abstractmethod
    def velocity_gradient(self, power: float, volume: float, viscosity: float) -> float:
        """
        Calculate velocity gradient (G-value).
        
        Args:
            power: Power dissipation [W]
            volume: Volume [m³]
            viscosity: Dynamic viscosity [Pa·s]
            
        Returns:
            Velocity gradient [1/s]
        """
        pass
    
    @abstractmethod
    def dead_zone_fraction(self, velocity_field: np.ndarray, 
                          threshold: float = 0.05) -> float:
        """
        Calculate dead zone fraction.
        
        Args:
            velocity_field: Velocity field data
            threshold: Velocity threshold [m/s]
            
        Returns:
            Dead zone fraction [-]
        """
        pass
    
    @abstractmethod
    def mixing_efficiency(self, actual: float, theoretical: float) -> float:
        """
        Calculate mixing efficiency.
        
        Args:
            actual: Actual mixing parameter
            theoretical: Theoretical mixing parameter
            
        Returns:
            Efficiency [-]
        """
        pass