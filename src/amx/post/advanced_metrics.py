"""Advanced mixing metrics for industrial-scale analysis.

Provides comprehensive metrics for evaluating mixing performance
in anaerobic digesters based on CFD results.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import xarray as xr
from scipy import integrate, interpolate
from scipy.spatial import KDTree


@dataclass
class MixingMetricsAdvanced:
    """
    Advanced mixing performance metrics for anaerobic digesters.
    
    Based on industrial standards and research literature:
    - CSTR mixing efficiency
    - Dead zone quantification
    - Short-circuiting analysis
    - Energy dissipation distribution
    """
    
    def __init__(self, mesh_data: Dict, field_data: Dict):
        """
        Initialize with mesh and field data.
        
        Args:
            mesh_data: Dictionary with 'points' and 'cells'
            field_data: Dictionary with field arrays (U, k, epsilon, etc.)
        """
        self.mesh = mesh_data
        self.fields = field_data
        self.points = np.array(mesh_data['points'])
        self.n_cells = len(self.points)
        
        # Build KD-tree for spatial queries
        self.kdtree = KDTree(self.points)
    
    def calculate_velocity_statistics(self) -> Dict[str, float]:
        """
        Calculate comprehensive velocity field statistics.
        
        Returns:
            Dictionary with velocity metrics
        """
        U = self.fields.get('U', None)
        if U is None:
            raise ValueError("Velocity field 'U' not found")
        
        # Velocity magnitude
        U_mag = np.linalg.norm(U, axis=1)
        
        # Volume-weighted statistics
        volumes = self.mesh.get('volumes', np.ones(self.n_cells))
        total_volume = np.sum(volumes)
        
        # Mean velocity (volume-weighted)
        U_mean = np.sum(U_mag * volumes) / total_volume
        
        # RMS velocity
        U_rms = np.sqrt(np.sum(U_mag**2 * volumes) / total_volume)
        
        # Velocity variance
        U_var = np.sum((U_mag - U_mean)**2 * volumes) / total_volume
        
        # Coefficient of variation (mixing uniformity indicator)
        CoV = np.sqrt(U_var) / U_mean if U_mean > 0 else float('inf')
        
        # Percentiles
        U_10 = np.percentile(U_mag, 10)
        U_50 = np.percentile(U_mag, 50)
        U_90 = np.percentile(U_mag, 90)
        
        # Velocity gradients (for shear rate)
        grad_U = self._calculate_velocity_gradient(U)
        shear_rate = self._calculate_shear_rate(grad_U)
        mean_shear = np.mean(shear_rate)
        
        return {
            'mean_velocity': U_mean,
            'rms_velocity': U_rms,
            'velocity_variance': U_var,
            'coefficient_of_variation': CoV,
            'velocity_p10': U_10,
            'velocity_p50': U_50,
            'velocity_p90': U_90,
            'mean_shear_rate': mean_shear,
            'max_velocity': np.max(U_mag),
            'min_velocity': np.min(U_mag),
        }
    
    def calculate_dead_zones(self, threshold_velocity: float = 0.05) -> Dict[str, float]:
        """
        Identify and quantify dead zones based on velocity threshold.
        
        Args:
            threshold_velocity: Minimum velocity for active mixing [m/s]
            
        Returns:
            Dictionary with dead zone metrics
        """
        U = self.fields.get('U', None)
        if U is None:
            raise ValueError("Velocity field 'U' not found")
        
        U_mag = np.linalg.norm(U, axis=1)
        volumes = self.mesh.get('volumes', np.ones(self.n_cells))
        total_volume = np.sum(volumes)
        
        # Identify dead zone cells
        dead_zone_mask = U_mag < threshold_velocity
        dead_zone_volume = np.sum(volumes[dead_zone_mask])
        dead_zone_fraction = dead_zone_volume / total_volume
        
        # Stagnant zones (even lower velocity)
        stagnant_mask = U_mag < (threshold_velocity * 0.2)
        stagnant_volume = np.sum(volumes[stagnant_mask])
        stagnant_fraction = stagnant_volume / total_volume
        
        # Active mixing zones
        active_mask = U_mag >= threshold_velocity
        active_volume = np.sum(volumes[active_mask])
        active_fraction = active_volume / total_volume
        
        # Dead zone connectivity (largest connected dead zone)
        largest_dead_zone = self._find_largest_connected_region(dead_zone_mask)
        
        return {
            'dead_zone_fraction': dead_zone_fraction,
            'dead_zone_volume_m3': dead_zone_volume,
            'stagnant_fraction': stagnant_fraction,
            'stagnant_volume_m3': stagnant_volume,
            'active_fraction': active_fraction,
            'active_volume_m3': active_volume,
            'largest_dead_zone_fraction': largest_dead_zone / total_volume,
            'threshold_velocity_used': threshold_velocity,
        }
    
    def calculate_mixing_time(self, tracer_data: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Estimate mixing time from flow field or tracer data.
        
        Args:
            tracer_data: Optional tracer concentration over time
            
        Returns:
            Dictionary with mixing time estimates
        """
        if tracer_data is not None:
            # Direct calculation from tracer
            return self._mixing_time_from_tracer(tracer_data)
        
        # Estimate from flow field
        U = self.fields.get('U', None)
        if U is None:
            raise ValueError("Velocity field 'U' not found")
        
        U_mag = np.linalg.norm(U, axis=1)
        U_mean = np.mean(U_mag)
        
        # Tank dimensions
        bbox = self._get_bounding_box()
        L_char = np.max(bbox[1] - bbox[0])  # Characteristic length
        
        # Circulation time
        t_circ = L_char / U_mean if U_mean > 0 else float('inf')
        
        # Mixing time correlations
        # Method 1: Based on circulation time
        t_mix_circ = 4 * t_circ  # Rule of thumb: 4 circulations
        
        # Method 2: Based on turbulence
        k = self.fields.get('k', None)
        epsilon = self.fields.get('epsilon', None)
        
        if k is not None and epsilon is not None:
            k_mean = np.mean(k)
            eps_mean = np.mean(epsilon)
            
            # Turbulent time scale
            t_turb = k_mean / eps_mean if eps_mean > 0 else float('inf')
            
            # Mixing time from turbulence
            t_mix_turb = 10 * t_turb  # Typical factor for CSTR
        else:
            t_mix_turb = float('inf')
        
        # Method 3: Based on energy dissipation
        if epsilon is not None:
            # Average dissipation rate
            eps_avg = np.mean(epsilon)
            
            # Mixing time from Camp number
            # Gt = 10^4 for good mixing
            G = np.sqrt(eps_avg / 1e-6)  # Assume nu = 1e-6 m²/s
            t_mix_camp = 10000 / G if G > 0 else float('inf')
        else:
            t_mix_camp = float('inf')
        
        return {
            'circulation_time': t_circ,
            'mixing_time_circulation': t_mix_circ,
            'mixing_time_turbulence': t_mix_turb,
            'mixing_time_camp': t_mix_camp,
            'mixing_time_estimate': min(t_mix_circ, t_mix_turb, t_mix_camp),
        }
    
    def calculate_energy_metrics(self) -> Dict[str, float]:
        """
        Calculate energy-related mixing metrics.
        
        Returns:
            Dictionary with energy metrics
        """
        epsilon = self.fields.get('epsilon', None)
        if epsilon is None:
            raise ValueError("Dissipation field 'epsilon' not found")
        
        volumes = self.mesh.get('volumes', np.ones(self.n_cells))
        total_volume = np.sum(volumes)
        
        # Volume-averaged dissipation
        eps_avg = np.sum(epsilon * volumes) / total_volume
        
        # Dissipation variance (uniformity indicator)
        eps_var = np.sum((epsilon - eps_avg)**2 * volumes) / total_volume
        eps_std = np.sqrt(eps_var)
        
        # Camp velocity gradient G
        nu = 1e-6  # Kinematic viscosity (should come from config)
        G_avg = np.sqrt(eps_avg / nu)
        
        # Local G values
        G_local = np.sqrt(epsilon / nu)
        G_max = np.max(G_local)
        G_min = np.min(G_local)
        
        # Energy distribution uniformity
        uniformity = 1.0 - (eps_std / eps_avg) if eps_avg > 0 else 0.0
        
        # Power density
        rho = 1000  # Density (should come from config)
        power_density = rho * eps_avg  # W/m³
        total_power = power_density * total_volume  # W
        
        return {
            'avg_dissipation_rate': eps_avg,
            'dissipation_variance': eps_var,
            'avg_velocity_gradient_G': G_avg,
            'max_velocity_gradient_G': G_max,
            'min_velocity_gradient_G': G_min,
            'energy_uniformity': uniformity,
            'power_density_W_m3': power_density,
            'total_power_dissipation_W': total_power,
        }
    
    def calculate_mixing_indices(self) -> Dict[str, float]:
        """
        Calculate various mixing quality indices.
        
        Returns:
            Dictionary with mixing indices
        """
        # Uniformity index based on velocity
        U = self.fields.get('U', None)
        if U is None:
            raise ValueError("Velocity field 'U' not found")
        
        U_mag = np.linalg.norm(U, axis=1)
        U_mean = np.mean(U_mag)
        U_std = np.std(U_mag)
        
        # Coefficient of variation (lower is better)
        CoV = U_std / U_mean if U_mean > 0 else float('inf')
        
        # Mixing index (0-1, higher is better)
        mixing_index = 1.0 / (1.0 + CoV)
        
        # Turbulence intensity
        k = self.fields.get('k', None)
        if k is not None:
            k_mean = np.mean(k)
            turb_intensity = np.sqrt(2 * k_mean / 3) / U_mean if U_mean > 0 else 0
        else:
            turb_intensity = 0
        
        # Flow number (jet momentum vs buoyancy)
        # Simplified calculation
        jet_velocity = 4.0  # From config
        tank_height = 16.0  # From config
        g = 9.81
        
        Fr = jet_velocity / np.sqrt(g * tank_height)
        
        # Mixing effectiveness (combination of metrics)
        dead_zones = self.calculate_dead_zones()
        dead_fraction = dead_zones['dead_zone_fraction']
        
        effectiveness = (1 - dead_fraction) * mixing_index
        
        return {
            'coefficient_of_variation': CoV,
            'mixing_index': mixing_index,
            'turbulence_intensity': turb_intensity,
            'froude_number': Fr,
            'mixing_effectiveness': effectiveness,
            'quality_grade': self._grade_mixing_quality(effectiveness),
        }
    
    def calculate_residence_time_distribution(self, 
                                            n_particles: int = 1000,
                                            injection_points: Optional[np.ndarray] = None) -> Dict:
        """
        Estimate residence time distribution from velocity field.
        
        Args:
            n_particles: Number of tracer particles
            injection_points: Injection locations
            
        Returns:
            RTD statistics
        """
        U = self.fields.get('U', None)
        if U is None:
            raise ValueError("Velocity field 'U' not found")
        
        # This is a simplified approach
        # Full implementation would require particle tracking
        
        # Estimate from velocity field statistics
        U_mag = np.linalg.norm(U, axis=1)
        
        # Tank volume and flow rate
        volumes = self.mesh.get('volumes', np.ones(self.n_cells))
        total_volume = np.sum(volumes)
        
        # Theoretical mean residence time
        flow_rate = 430.0 / 3600  # m³/s (from config)
        tau_theoretical = total_volume / flow_rate
        
        # Estimate RTD spread from velocity variations
        U_mean = np.mean(U_mag)
        U_std = np.std(U_mag)
        
        # Peclet number (advection/dispersion)
        L_char = (total_volume)**(1/3)
        D_eff = 0.1 * U_std * L_char  # Effective dispersion
        Pe = U_mean * L_char / D_eff if D_eff > 0 else float('inf')
        
        # RTD variance for CSTR with non-ideal mixing
        sigma_squared = 2 / Pe if Pe > 0 else float('inf')
        
        return {
            'mean_residence_time': tau_theoretical,
            'peclet_number': Pe,
            'rtd_variance': sigma_squared,
            'mixing_model': 'CSTR with dispersion' if Pe < 100 else 'Near plug flow',
        }
    
    def _calculate_velocity_gradient(self, U: np.ndarray) -> np.ndarray:
        """Calculate velocity gradient tensor."""
        # Simplified finite difference approach
        # Full implementation would use mesh connectivity
        n_points = len(U)
        grad_U = np.zeros((n_points, 3, 3))
        
        # Use nearest neighbors for gradient estimation
        for i in range(n_points):
            # Find nearest neighbors
            distances, indices = self.kdtree.query(self.points[i], k=7)
            
            if len(indices) > 1:
                # Finite difference approximation
                for j in indices[1:]:  # Skip self
                    dx = self.points[j] - self.points[i]
                    dU = U[j] - U[i]
                    
                    # Update gradient components
                    for m in range(3):
                        for n in range(3):
                            if abs(dx[n]) > 1e-10:
                                grad_U[i, m, n] += dU[m] / dx[n] / len(indices)
        
        return grad_U
    
    def _calculate_shear_rate(self, grad_U: np.ndarray) -> np.ndarray:
        """Calculate shear rate from velocity gradient."""
        n_points = grad_U.shape[0]
        shear_rate = np.zeros(n_points)
        
        for i in range(n_points):
            # Strain rate tensor
            S = 0.5 * (grad_U[i] + grad_U[i].T)
            
            # Shear rate magnitude
            shear_rate[i] = np.sqrt(2 * np.sum(S * S))
        
        return shear_rate
    
    def _find_largest_connected_region(self, mask: np.ndarray) -> float:
        """Find largest connected region in masked cells."""
        # Simplified approach - returns fraction of total dead zone
        # Full implementation would use connectivity analysis
        return np.sum(mask) * 0.3  # Assume 30% is largest connected region
    
    def _get_bounding_box(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get bounding box of mesh."""
        min_coords = np.min(self.points, axis=0)
        max_coords = np.max(self.points, axis=0)
        return min_coords, max_coords
    
    def _mixing_time_from_tracer(self, tracer_data: np.ndarray) -> Dict[str, float]:
        """Calculate mixing time from tracer data."""
        # Placeholder for tracer-based calculation
        return {
            'mixing_time_95': 1800.0,  # 30 minutes
            'mixing_time_99': 2400.0,  # 40 minutes
        }
    
    def _grade_mixing_quality(self, effectiveness: float) -> str:
        """Grade mixing quality based on effectiveness."""
        if effectiveness > 0.9:
            return "Excellent"
        elif effectiveness > 0.75:
            return "Good"
        elif effectiveness > 0.6:
            return "Adequate"
        elif effectiveness > 0.4:
            return "Poor"
        else:
            return "Inadequate"
    
    def generate_report(self) -> Dict:
        """
        Generate comprehensive mixing analysis report.
        
        Returns:
            Complete analysis report
        """
        report = {
            'velocity_statistics': self.calculate_velocity_statistics(),
            'dead_zones': self.calculate_dead_zones(),
            'mixing_time': self.calculate_mixing_time(),
            'energy_metrics': self.calculate_energy_metrics(),
            'mixing_indices': self.calculate_mixing_indices(),
            'residence_time': self.calculate_residence_time_distribution(),
        }
        
        # Add summary and recommendations
        report['summary'] = self._generate_summary(report)
        report['recommendations'] = self._generate_recommendations(report)
        
        return report
    
    def _generate_summary(self, report: Dict) -> Dict:
        """Generate summary from report data."""
        return {
            'overall_performance': report['mixing_indices']['quality_grade'],
            'key_metrics': {
                'mean_velocity': report['velocity_statistics']['mean_velocity'],
                'dead_zone_fraction': report['dead_zones']['dead_zone_fraction'],
                'mixing_time': report['mixing_time']['mixing_time_estimate'],
                'energy_efficiency': report['energy_metrics']['energy_uniformity'],
            },
            'meets_targets': self._check_targets(report),
        }
    
    def _check_targets(self, report: Dict) -> Dict[str, bool]:
        """Check if performance meets targets."""
        return {
            'velocity_target': report['velocity_statistics']['mean_velocity'] >= 0.3,
            'dead_zone_target': report['dead_zones']['dead_zone_fraction'] < 0.1,
            'mixing_time_target': report['mixing_time']['mixing_time_estimate'] < 1800,
        }
    
    def _generate_recommendations(self, report: Dict) -> List[str]:
        """Generate recommendations based on analysis."""
        recommendations = []
        
        # Check dead zones
        if report['dead_zones']['dead_zone_fraction'] > 0.15:
            recommendations.append("High dead zone fraction - consider increasing jet velocity or adjusting nozzle angles")
        
        # Check mixing time
        if report['mixing_time']['mixing_time_estimate'] > 2400:
            recommendations.append("Long mixing time - increase flow rate or optimize nozzle configuration")
        
        # Check energy distribution
        if report['energy_metrics']['energy_uniformity'] < 0.5:
            recommendations.append("Non-uniform energy distribution - redistribute nozzles for better coverage")
        
        # Check velocity
        if report['velocity_statistics']['mean_velocity'] < 0.25:
            recommendations.append("Low mean velocity - increase pump power or optimize jet configuration")
        
        if not recommendations:
            recommendations.append("System performing within acceptable parameters")
        
        return recommendations