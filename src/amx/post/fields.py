"""Process velocity and turbulence fields."""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pyvista as pv


class FieldProcessor:
    """Process OpenFOAM field data."""

    def __init__(self, mesh: pv.UnstructuredGrid):
        """
        Initialize field processor.
        
        Args:
            mesh: PyVista mesh with field data
        """
        self.mesh = mesh

    def get_velocity_magnitude(self) -> np.ndarray:
        """Get velocity magnitude field."""
        if "U" not in self.mesh.array_names:
            raise ValueError("Velocity field 'U' not found in mesh")
        
        U = self.mesh["U"]
        return np.linalg.norm(U, axis=1)

    def get_turbulent_kinetic_energy(self) -> Optional[np.ndarray]:
        """Get turbulent kinetic energy field."""
        if "k" in self.mesh.array_names:
            return self.mesh["k"]
        return None

    def get_dissipation_rate(self) -> Optional[np.ndarray]:
        """Get turbulent dissipation rate."""
        if "epsilon" in self.mesh.array_names:
            return self.mesh["epsilon"]
        return None

    def get_eddy_viscosity(self) -> Optional[np.ndarray]:
        """Get turbulent eddy viscosity."""
        if "nut" in self.mesh.array_names:
            return self.mesh["nut"]
        return None

    def calculate_vorticity(self) -> np.ndarray:
        """Calculate vorticity magnitude from velocity field."""
        if "U" not in self.mesh.array_names:
            raise ValueError("Velocity field 'U' not found in mesh")
        
        # Compute gradients
        gradients = self.mesh.compute_derivative("U", gradient=True)
        grad_U = gradients["gradient"]
        
        # Vorticity = curl(U)
        # For structured data, PyVista can compute this directly
        # For unstructured, we approximate
        vorticity = np.zeros_like(self.mesh["U"])
        
        # ωx = ∂w/∂y - ∂v/∂z
        vorticity[:, 0] = grad_U[:, 8] - grad_U[:, 5]  # dwdy - dvdz
        # ωy = ∂u/∂z - ∂w/∂x
        vorticity[:, 1] = grad_U[:, 2] - grad_U[:, 6]  # dudz - dwdx
        # ωz = ∂v/∂x - ∂u/∂y
        vorticity[:, 2] = grad_U[:, 3] - grad_U[:, 1]  # dvdx - dudy
        
        return np.linalg.norm(vorticity, axis=1)

    def calculate_q_criterion(self) -> np.ndarray:
        """Calculate Q-criterion for vortex identification."""
        if "U" not in self.mesh.array_names:
            raise ValueError("Velocity field 'U' not found in mesh")
        
        # Compute velocity gradient tensor
        gradients = self.mesh.compute_derivative("U", gradient=True)
        grad_U = gradients["gradient"].reshape(-1, 3, 3)
        
        # Q = 0.5 * (||Ω||² - ||S||²)
        # where Ω is vorticity tensor and S is strain rate tensor
        
        Q = np.zeros(len(grad_U))
        for i, grad in enumerate(grad_U):
            # Strain rate tensor S = 0.5 * (grad_U + grad_U.T)
            S = 0.5 * (grad + grad.T)
            # Vorticity tensor Ω = 0.5 * (grad_U - grad_U.T)
            Omega = 0.5 * (grad - grad.T)
            
            # Q-criterion
            Q[i] = 0.5 * (np.linalg.norm(Omega)**2 - np.linalg.norm(S)**2)
        
        return Q

    def calculate_strain_rate(self) -> np.ndarray:
        """Calculate strain rate magnitude."""
        if "U" not in self.mesh.array_names:
            raise ValueError("Velocity field 'U' not found in mesh")
        
        # Compute velocity gradient tensor
        gradients = self.mesh.compute_derivative("U", gradient=True)
        grad_U = gradients["gradient"].reshape(-1, 3, 3)
        
        strain_rate = np.zeros(len(grad_U))
        for i, grad in enumerate(grad_U):
            # Strain rate tensor S = 0.5 * (grad_U + grad_U.T)
            S = 0.5 * (grad + grad.T)
            # Strain rate magnitude = sqrt(2 * S:S)
            strain_rate[i] = np.sqrt(2 * np.sum(S * S))
        
        return strain_rate

    def extract_region(self, bounds: Tuple[float, ...]) -> pv.UnstructuredGrid:
        """
        Extract a region of the mesh.
        
        Args:
            bounds: (xmin, xmax, ymin, ymax, zmin, zmax)
            
        Returns:
            Extracted mesh region
        """
        return self.mesh.clip_box(bounds, invert=False)

    def extract_slice(self, normal: Tuple[float, float, float], 
                     origin: Optional[Tuple[float, float, float]] = None) -> pv.PolyData:
        """
        Extract a slice through the mesh.
        
        Args:
            normal: Slice normal vector
            origin: Slice origin (mesh center if None)
            
        Returns:
            Slice mesh
        """
        if origin is None:
            origin = self.mesh.center
        
        return self.mesh.slice(normal=normal, origin=origin)

    def compute_statistics(self, field_name: str) -> Dict[str, float]:
        """
        Compute statistics for a field.
        
        Args:
            field_name: Name of the field
            
        Returns:
            Dictionary of statistics
        """
        if field_name not in self.mesh.array_names:
            raise ValueError(f"Field '{field_name}' not found in mesh")
        
        field = self.mesh[field_name]
        
        # Handle vector fields
        if len(field.shape) > 1 and field.shape[1] > 1:
            # Use magnitude for vector fields
            field = np.linalg.norm(field, axis=1)
        
        # Get cell volumes for weighted statistics
        volumes = self.mesh.compute_cell_sizes(length=False, area=False, volume=True)["Volume"]
        
        # Volume-weighted statistics
        total_volume = np.sum(volumes)
        if total_volume > 0:
            weighted_mean = np.sum(field * volumes) / total_volume
        else:
            weighted_mean = np.mean(field)
        
        # Variance
        if total_volume > 0:
            weighted_var = np.sum((field - weighted_mean)**2 * volumes) / total_volume
        else:
            weighted_var = np.var(field)
        weighted_std = np.sqrt(weighted_var)
        
        return {
            "mean": weighted_mean,
            "std": weighted_std,
            "min": np.min(field),
            "max": np.max(field),
            "median": np.median(field),
            "variance": weighted_var,
        }

    def identify_dead_zones(self, velocity_threshold: float = 0.05) -> Tuple[pv.UnstructuredGrid, float]:
        """
        Identify dead zones based on velocity magnitude.
        
        Args:
            velocity_threshold: Threshold velocity (m/s)
            
        Returns:
            Tuple of (dead zone mesh, dead zone volume fraction)
        """
        U_mag = self.get_velocity_magnitude()
        
        # Create threshold filter
        self.mesh["U_mag"] = U_mag
        dead_zones = self.mesh.threshold(value=velocity_threshold, scalars="U_mag", invert=True)
        
        # Calculate volume fraction
        if dead_zones.n_cells > 0:
            dead_volume = np.sum(dead_zones.compute_cell_sizes()["Volume"])
            total_volume = np.sum(self.mesh.compute_cell_sizes()["Volume"])
            if total_volume > 0:
                dead_fraction = dead_volume / total_volume
            else:
                dead_fraction = 0.0
        else:
            dead_fraction = 0.0
        
        return dead_zones, dead_fraction