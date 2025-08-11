"""Calculate mixing performance metrics."""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pyvista as pv

from amx.config import Config
from amx.post.fields import FieldProcessor


class MixingMetrics:
    """Calculate mixing performance metrics."""

    def __init__(self, mesh: pv.UnstructuredGrid, config: Optional[Config] = None):
        """
        Initialize mixing metrics calculator.
        
        Args:
            mesh: PyVista mesh with field data
            config: Simulation configuration
        """
        self.mesh = mesh
        self.config = config
        self.processor = FieldProcessor(mesh)

    def calculate_mean_velocity(self) -> float:
        """
        Calculate volume-weighted mean velocity magnitude.
        
        Returns:
            Mean velocity in m/s
        """
        U_mag = self.processor.get_velocity_magnitude()
        
        try:
            volumes = self.mesh.compute_cell_sizes()["Volume"]
            
            # Check if we need to convert point data to cell data
            if len(U_mag) != len(volumes):
                if len(U_mag) == self.mesh.n_points and len(volumes) == self.mesh.n_cells:
                    # Convert point data to cell data by averaging
                    mesh_with_cell_data = self.mesh.point_data_to_cell_data()
                    if "U" in mesh_with_cell_data.array_names:
                        U = mesh_with_cell_data["U"]
                        U_mag = np.linalg.norm(U, axis=1)
            
            # Volume-weighted average if shapes match
            total_volume = np.sum(volumes)
            if total_volume > 0 and len(U_mag) == len(volumes):
                mean_velocity = np.sum(U_mag * volumes) / total_volume
            else:
                mean_velocity = np.mean(U_mag)
        except Exception:
            # Fallback to simple mean
            mean_velocity = np.mean(U_mag)
        
        return float(mean_velocity)

    def calculate_velocity_variance(self) -> float:
        """
        Calculate velocity variance as mixing uniformity indicator.
        
        Returns:
            Velocity variance
        """
        stats = self.processor.compute_statistics("U")
        return stats["variance"]

    def calculate_mixing_intensity(self) -> float:
        """
        Calculate mixing intensity (turbulent to mean velocity ratio).
        
        Returns:
            Mixing intensity
        """
        mean_U = self.calculate_mean_velocity()
        k = self.processor.get_turbulent_kinetic_energy()
        
        if k is None or mean_U == 0:
            return 0.0
        
        # Turbulent velocity scale
        U_turb = np.sqrt(2/3 * np.mean(k))
        
        return float(U_turb / mean_U)

    def calculate_energy_density(self, power_w: Optional[float] = None) -> float:
        """
        Calculate energy density (W/m³).
        
        Args:
            power_w: Input power in watts
            
        Returns:
            Energy density in W/m³
        """
        if power_w is None and self.config and self.config.operation:
            # Calculate from pump parameters
            power_w = self._calculate_pump_power()
        
        if power_w is None:
            return 0.0
        
        total_volume = np.sum(self.mesh.compute_cell_sizes()["Volume"])
        if total_volume > 0:
            return power_w / total_volume
        else:
            # Use mesh bounds to estimate volume
            bounds = self.mesh.bounds
            volume = (bounds[1] - bounds[0]) * (bounds[3] - bounds[2]) * (bounds[5] - bounds[4])
            return power_w / max(volume, 1e-10)

    def calculate_g_value(self, mu: Optional[float] = None) -> float:
        """
        Calculate velocity gradient (G-value) for mixing.
        
        G = sqrt(P / (μ * V))
        
        Args:
            mu: Dynamic viscosity (Pa·s)
            
        Returns:
            G-value in s⁻¹
        """
        if mu is None and self.config:
            mu = self.config.fluid.mu
        
        if mu is None:
            return 0.0
        
        power_w = self._calculate_pump_power()
        if power_w is None:
            return 0.0
        
        total_volume = np.sum(self.mesh.compute_cell_sizes()["Volume"])
        if total_volume <= 0:
            # Use mesh bounds to estimate volume
            bounds = self.mesh.bounds
            total_volume = (bounds[1] - bounds[0]) * (bounds[3] - bounds[2]) * (bounds[5] - bounds[4])
        
        if total_volume > 0 and mu > 0:
            return float(np.sqrt(power_w / (mu * total_volume)))
        else:
            return 0.0

    def calculate_dead_zones(self, threshold_velocity: float = 0.05) -> Dict[str, float]:
        """
        Calculate dead zone metrics.
        
        Args:
            threshold_velocity: Velocity threshold for dead zones (m/s)
            
        Returns:
            Dictionary with dead zone metrics
        """
        dead_zones, dead_fraction = self.processor.identify_dead_zones(threshold_velocity)
        
        return {
            "volume_fraction": dead_fraction,
            "volume_percent": dead_fraction * 100,
            "n_cells": dead_zones.n_cells if dead_zones.n_cells > 0 else 0,
        }

    def calculate_regional_metrics(self, regions: Optional[List[Dict]] = None) -> Dict[str, Dict]:
        """
        Calculate metrics for specific regions.
        
        Args:
            regions: List of region definitions with bounds
            
        Returns:
            Dictionary of regional metrics
        """
        if regions is None:
            # Default regions: bottom, middle, top, corners
            H = self.mesh.bounds[5] - self.mesh.bounds[4]  # z_max - z_min
            L = self.mesh.bounds[1] - self.mesh.bounds[0]  # x_max - x_min
            W = self.mesh.bounds[3] - self.mesh.bounds[2]  # y_max - y_min
            
            regions = [
                {"name": "bottom", "bounds": [0, L, 0, W, 0, H/3]},
                {"name": "middle", "bounds": [0, L, 0, W, H/3, 2*H/3]},
                {"name": "top", "bounds": [0, L, 0, W, 2*H/3, H]},
                {"name": "corner_1", "bounds": [0, L/4, 0, W/4, 0, H/3]},
            ]
        
        regional_metrics = {}
        
        for region in regions:
            name = region["name"]
            bounds = region.get("bounds")
            
            if bounds:
                # Extract region
                region_mesh = self.processor.extract_region(bounds)
                
                if region_mesh.n_cells > 0:
                    # Create processor for region
                    region_processor = FieldProcessor(region_mesh)
                    
                    # Calculate metrics
                    U_mag = region_processor.get_velocity_magnitude()
                    volumes = region_mesh.compute_cell_sizes()["Volume"]
                    
                    total_vol = np.sum(volumes)
                    if total_vol > 0:
                        mean_U = np.sum(U_mag * volumes) / total_vol
                    else:
                        mean_U = np.mean(U_mag)
                    
                    regional_metrics[name] = {
                        "mean_velocity": float(mean_U),
                        "volume": float(np.sum(volumes)),
                        "n_cells": region_mesh.n_cells,
                    }
                else:
                    regional_metrics[name] = {
                        "mean_velocity": 0.0,
                        "volume": 0.0,
                        "n_cells": 0,
                    }
        
        return regional_metrics

    def calculate_mixing_time_proxy(self) -> float:
        """
        Calculate mixing time proxy based on circulation patterns.
        
        Returns:
            Estimated mixing time in seconds
        """
        mean_U = self.calculate_mean_velocity()
        
        if mean_U == 0:
            return float('inf')
        
        # Characteristic length (hydraulic diameter)
        bounds = self.mesh.bounds
        L = bounds[1] - bounds[0]
        W = bounds[3] - bounds[2]
        H = bounds[5] - bounds[4]
        
        # Hydraulic diameter for rectangular tank
        A = L * W
        P = 2 * (L + W)
        D_h = 4 * A / P
        
        # Circulation time scale
        t_circ = D_h / mean_U
        
        # Mixing time is typically 3-5 circulation times
        mixing_time = 4 * t_circ
        
        return float(mixing_time)

    def calculate_uniformity_index(self) -> float:
        """
        Calculate mixing uniformity index (0=poor, 1=perfect).
        
        Returns:
            Uniformity index
        """
        U_mag = self.processor.get_velocity_magnitude()
        mean_U = np.mean(U_mag)
        
        if mean_U == 0:
            return 0.0
        
        # Coefficient of variation
        cv = np.std(U_mag) / mean_U
        
        # Convert to uniformity index (lower CV = higher uniformity)
        uniformity = 1.0 / (1.0 + cv)
        
        return float(uniformity)

    def _calculate_pump_power(self) -> Optional[float]:
        """Calculate pump power from configuration."""
        if not self.config or not self.config.operation:
            return None
        
        # P = ρ * g * Q * H / η
        rho = self.config.fluid.rho
        g = 9.81
        Q = self.config.operation.pump_total_m3ph / 3600  # Convert to m³/s
        H = self.config.operation.head_m
        
        # Efficiency
        eta = 0.65  # Default
        if self.config.energy:
            eta = self.config.energy.pump_efficiency
        
        power_w = rho * g * Q * H / eta
        
        return power_w

    def get_summary_metrics(self) -> Dict[str, float]:
        """
        Get comprehensive summary of mixing metrics.
        
        Returns:
            Dictionary of all key metrics
        """
        metrics = {
            "mean_velocity_mps": self.calculate_mean_velocity(),
            "velocity_variance": self.calculate_velocity_variance(),
            "mixing_intensity": self.calculate_mixing_intensity(),
            "uniformity_index": self.calculate_uniformity_index(),
            "mixing_time_s": self.calculate_mixing_time_proxy(),
        }
        
        # Add energy metrics if configuration available
        if self.config:
            power_w = self._calculate_pump_power()
            if power_w:
                metrics["power_w"] = power_w
                metrics["energy_density_w_m3"] = self.calculate_energy_density(power_w)
                metrics["g_value_s-1"] = self.calculate_g_value()
        
        # Add dead zone metrics
        dead_zone_metrics = self.calculate_dead_zones()
        metrics.update({f"dead_zone_{k}": v for k, v in dead_zone_metrics.items()})
        
        return metrics