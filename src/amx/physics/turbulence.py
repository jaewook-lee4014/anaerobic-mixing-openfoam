"""Academically rigorous turbulence models based on RANS equations."""

import numpy as np
from typing import Dict, Optional
from abc import ABC, abstractmethod


class TurbulenceModel(ABC):
    """
    Abstract base class for turbulence models.
    
    Based on Reynolds-Averaged Navier-Stokes (RANS) approach.
    """
    
    def __init__(self, nu: float, rho: float = 998.0):
        """
        Initialize turbulence model.
        
        Args:
            nu: Kinematic viscosity [m²/s]
            rho: Density [kg/m³]
        """
        self.nu = nu
        self.rho = rho
        self.mu = rho * nu  # Dynamic viscosity
    
    @abstractmethod
    def calculate_eddy_viscosity(self, k: float, epsilon: float) -> float:
        """Calculate turbulent eddy viscosity."""
        pass
    
    @abstractmethod
    def calculate_production(self, grad_u: np.ndarray) -> float:
        """Calculate turbulence production term."""
        pass


class RANS_kEpsilon(TurbulenceModel):
    """
    Standard k-ε turbulence model.
    
    References:
    - Launder, B.E. & Spalding, D.B. (1974). The numerical computation of turbulent flows.
    - Jones, W.P. & Launder, B.E. (1972). The prediction of laminarization with a two-equation model.
    - Wilcox, D.C. (2006). Turbulence Modeling for CFD.
    """
    
    # Model constants (standard values from Launder & Spalding, 1974)
    C_mu = 0.09      # Eddy viscosity constant
    C_1e = 1.44      # ε equation constant
    C_2e = 1.92      # ε equation constant  
    C_3e = 0.0       # Buoyancy constant (0 for non-buoyant flows)
    sigma_k = 1.0    # Prandtl number for k
    sigma_e = 1.3    # Prandtl number for ε
    
    def calculate_eddy_viscosity(self, k: float, epsilon: float) -> float:
        """
        Calculate turbulent eddy viscosity.
        
        ν_t = C_μ * k² / ε
        
        Args:
            k: Turbulent kinetic energy [m²/s²]
            epsilon: Dissipation rate [m²/s³]
            
        Returns:
            Eddy viscosity [m²/s]
        """
        if epsilon <= 0 or k <= 0:
            return 0.0
        
        return self.C_mu * k**2 / epsilon
    
    def calculate_production(self, grad_u: np.ndarray, k: float = None, 
                           epsilon: float = None, nu_t: float = None) -> float:
        """
        Calculate turbulence production term.
        
        P_k = -ρ * <u'_i * u'_j> * ∂U_i/∂x_j
        
        For incompressible flow:
        P_k = ν_t * S²
        
        where S = √(2S_ij*S_ij) is strain rate magnitude
        
        Args:
            grad_u: Velocity gradient tensor ∂u_i/∂x_j [1/s]
            k: Turbulent kinetic energy [m²/s²] (optional)
            epsilon: Dissipation rate [m²/s³] (optional)
            nu_t: Eddy viscosity [m²/s] (optional, will be calculated if k and epsilon provided)
            
        Returns:
            Production rate [m²/s³]
        """
        # Strain rate tensor S_ij = 0.5 * (∂u_i/∂x_j + ∂u_j/∂x_i)
        S = 0.5 * (grad_u + grad_u.T)
        
        # Strain rate magnitude
        S_mag = np.sqrt(2 * np.sum(S * S))
        
        # Get eddy viscosity
        if nu_t is None:
            if k is not None and epsilon is not None:
                nu_t = self.calculate_eddy_viscosity(k, epsilon)
            else:
                # Use correlation for high Reynolds number turbulent flow
                # ν_t ≈ 0.01 * U * L for fully developed turbulence
                # This should be overridden with actual values in production
                raise ValueError("Either provide nu_t or both k and epsilon for production calculation")
        
        return nu_t * S_mag**2
    
    def calculate_dissipation(self, k: float, length_scale: float) -> float:
        """
        Calculate dissipation rate.
        
        ε = C_μ^(3/4) * k^(3/2) / l
        
        Based on dimensional analysis and mixing length hypothesis.
        
        Args:
            k: Turbulent kinetic energy [m²/s²]
            length_scale: Turbulent length scale [m]
            
        Returns:
            Dissipation rate [m²/s³]
        """
        if length_scale <= 0 or k <= 0:
            return 1e-10  # Small positive value to avoid division by zero
        
        return self.C_mu**(3/4) * k**(3/2) / length_scale
    
    def calculate_length_scale(self, k: float, epsilon: float) -> float:
        """
        Calculate turbulent length scale.
        
        l = C_μ^(3/4) * k^(3/2) / ε
        
        Args:
            k: Turbulent kinetic energy [m²/s²]
            epsilon: Dissipation rate [m²/s³]
            
        Returns:
            Length scale [m]
        """
        if epsilon <= 0 or k <= 0:
            return 0.0
        
        return self.C_mu**(3/4) * k**(3/2) / epsilon
    
    def calculate_time_scale(self, k: float, epsilon: float) -> float:
        """
        Calculate turbulent time scale.
        
        τ = k / ε
        
        Args:
            k: Turbulent kinetic energy [m²/s²]
            epsilon: Dissipation rate [m²/s³]
            
        Returns:
            Time scale [s]
        """
        if epsilon <= 0:
            return float('inf')
        
        return k / epsilon
    
    def calculate_reynolds_stress(self, k: float, grad_u: np.ndarray, 
                                 nu_t: float) -> np.ndarray:
        """
        Calculate Reynolds stress tensor using Boussinesq hypothesis.
        
        -<u'_i * u'_j> = ν_t * (∂U_i/∂x_j + ∂U_j/∂x_i) - (2/3) * k * δ_ij
        
        Args:
            k: Turbulent kinetic energy [m²/s²]
            grad_u: Velocity gradient tensor [1/s]
            nu_t: Eddy viscosity [m²/s]
            
        Returns:
            Reynolds stress tensor [m²/s²]
        """
        # Strain rate tensor
        S = 0.5 * (grad_u + grad_u.T)
        
        # Identity matrix
        I = np.eye(3)
        
        # Reynolds stress
        tau = 2 * nu_t * S - (2/3) * k * I
        
        return tau
    
    def y_plus_calculation(self, y: float, u_tau: float) -> float:
        """
        Calculate dimensionless wall distance y+.
        
        y+ = y * u_τ / ν
        
        where u_τ = √(τ_w / ρ) is friction velocity
        
        Args:
            y: Wall distance [m]
            u_tau: Friction velocity [m/s]
            
        Returns:
            y+ value [-]
        """
        return y * u_tau / self.nu
    
    def wall_function_u_plus(self, y_plus: float) -> float:
        """
        Law of the wall for velocity.
        
        Viscous sublayer (y+ < 5):     u+ = y+
        Buffer layer (5 < y+ < 30):     Blending function
        Log layer (y+ > 30):           u+ = (1/κ) * ln(E * y+)
        
        where κ = 0.41 (von Karman constant), E = 9.8
        
        Args:
            y_plus: Dimensionless wall distance [-]
            
        Returns:
            Dimensionless velocity u+ [-]
        """
        kappa = 0.41  # von Karman constant
        E = 9.8       # Roughness parameter
        
        if y_plus <= 5:
            # Viscous sublayer
            return y_plus
        elif y_plus <= 30:
            # Buffer layer - blend between viscous and log
            u_visc = y_plus
            u_log = (1/kappa) * np.log(E * y_plus)
            # Smooth blending
            blend = (y_plus - 5) / 25
            return (1 - blend) * u_visc + blend * u_log
        else:
            # Log layer
            return (1/kappa) * np.log(E * y_plus)
    
    def estimate_initial_k_epsilon(self, U_ref: float, L_ref: float, 
                                  turbulence_intensity: float = 0.05) -> Dict[str, float]:
        """
        Estimate initial k and ε values for simulation.
        
        Based on:
        k = (3/2) * (U * I)²
        ε = C_μ^(3/4) * k^(3/2) / l
        
        Args:
            U_ref: Reference velocity [m/s]
            L_ref: Reference length scale [m]
            turbulence_intensity: Turbulence intensity (typically 0.01-0.10)
            
        Returns:
            Dict with k and epsilon values
        """
        # Turbulent kinetic energy
        k = 1.5 * (U_ref * turbulence_intensity)**2
        
        # Turbulent length scale (typically 0.07 * L_ref for internal flows)
        l = 0.07 * L_ref
        
        # Dissipation rate
        epsilon = self.C_mu**(3/4) * k**(3/2) / l
        
        return {
            'k': k,
            'epsilon': epsilon,
            'nu_t': self.calculate_eddy_viscosity(k, epsilon),
            'length_scale': l,
            'time_scale': self.calculate_time_scale(k, epsilon)
        }