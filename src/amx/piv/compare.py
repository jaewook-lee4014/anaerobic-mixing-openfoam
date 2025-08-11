"""Compare CFD results with PIV measurements."""

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import interpolate, stats


class PIVComparison:
    """Compare CFD and PIV velocity fields."""

    def __init__(self, cfd_data: Dict, piv_data: Dict):
        """
        Initialize comparison.
        
        Args:
            cfd_data: CFD velocity field data
            piv_data: PIV velocity field data
        """
        self.cfd_data = cfd_data
        self.piv_data = piv_data

    def interpolate_to_common_grid(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Interpolate both fields to common grid.
        
        Returns:
            Tuple of (u_cfd, v_cfd, u_piv, v_piv) on common grid
        """
        # Get PIV grid (usually coarser)
        x_piv = self.piv_data["x"]
        y_piv = self.piv_data["y"]
        
        # Flatten if needed
        if x_piv.ndim > 1:
            x_piv_flat = x_piv.flatten()
            y_piv_flat = y_piv.flatten()
        else:
            x_piv_flat = x_piv
            y_piv_flat = y_piv
        
        # Get CFD data
        x_cfd = self.cfd_data["x"]
        y_cfd = self.cfd_data["y"]
        u_cfd = self.cfd_data["u"]
        v_cfd = self.cfd_data["v"]
        
        # Flatten CFD data
        if x_cfd.ndim > 1:
            x_cfd_flat = x_cfd.flatten()
            y_cfd_flat = y_cfd.flatten()
            u_cfd_flat = u_cfd.flatten()
            v_cfd_flat = v_cfd.flatten()
        else:
            x_cfd_flat = x_cfd
            y_cfd_flat = y_cfd
            u_cfd_flat = u_cfd
            v_cfd_flat = v_cfd
        
        # Interpolate CFD to PIV grid
        u_cfd_interp = interpolate.griddata(
            (x_cfd_flat, y_cfd_flat),
            u_cfd_flat,
            (x_piv_flat, y_piv_flat),
            method='linear',
            fill_value=np.nan
        )
        
        v_cfd_interp = interpolate.griddata(
            (x_cfd_flat, y_cfd_flat),
            v_cfd_flat,
            (x_piv_flat, y_piv_flat),
            method='linear',
            fill_value=np.nan
        )
        
        # Get PIV velocities
        u_piv = self.piv_data.get("u_mean", self.piv_data.get("u"))
        v_piv = self.piv_data.get("v_mean", self.piv_data.get("v"))
        
        if u_piv.ndim > 1:
            u_piv = u_piv.flatten()
            v_piv = v_piv.flatten()
        
        return u_cfd_interp, v_cfd_interp, u_piv, v_piv

    def calculate_correlation(self) -> Dict[str, float]:
        """
        Calculate correlation coefficients.
        
        Returns:
            Dictionary with correlation metrics
        """
        u_cfd, v_cfd, u_piv, v_piv = self.interpolate_to_common_grid()
        
        # Remove NaN values
        mask = ~(np.isnan(u_cfd) | np.isnan(v_cfd) | np.isnan(u_piv) | np.isnan(v_piv))
        u_cfd = u_cfd[mask]
        v_cfd = v_cfd[mask]
        u_piv = u_piv[mask]
        v_piv = v_piv[mask]
        
        if len(u_cfd) == 0:
            return {
                "u_correlation": 0.0,
                "v_correlation": 0.0,
                "magnitude_correlation": 0.0,
                "vector_correlation": 0.0,
            }
        
        # Component correlations
        r_u, p_u = stats.pearsonr(u_cfd, u_piv)
        r_v, p_v = stats.pearsonr(v_cfd, v_piv)
        
        # Magnitude correlation
        mag_cfd = np.sqrt(u_cfd**2 + v_cfd**2)
        mag_piv = np.sqrt(u_piv**2 + v_piv**2)
        r_mag, p_mag = stats.pearsonr(mag_cfd, mag_piv)
        
        # Vector correlation (considers both components)
        vector_corr = (r_u + r_v) / 2
        
        return {
            "u_correlation": r_u,
            "v_correlation": r_v,
            "magnitude_correlation": r_mag,
            "vector_correlation": vector_corr,
            "u_p_value": p_u,
            "v_p_value": p_v,
            "magnitude_p_value": p_mag,
        }

    def calculate_errors(self) -> Dict[str, float]:
        """
        Calculate error metrics.
        
        Returns:
            Dictionary with error metrics
        """
        u_cfd, v_cfd, u_piv, v_piv = self.interpolate_to_common_grid()
        
        # Remove NaN values
        mask = ~(np.isnan(u_cfd) | np.isnan(v_cfd) | np.isnan(u_piv) | np.isnan(v_piv))
        u_cfd = u_cfd[mask]
        v_cfd = v_cfd[mask]
        u_piv = u_piv[mask]
        v_piv = v_piv[mask]
        
        if len(u_cfd) == 0:
            return {
                "rmse_u": np.inf,
                "rmse_v": np.inf,
                "rmse_magnitude": np.inf,
                "mae_u": np.inf,
                "mae_v": np.inf,
                "mae_magnitude": np.inf,
                "max_error": np.inf,
                "relative_error": np.inf,
            }
        
        # Component errors
        error_u = u_cfd - u_piv
        error_v = v_cfd - v_piv
        
        # RMSE
        rmse_u = np.sqrt(np.mean(error_u**2))
        rmse_v = np.sqrt(np.mean(error_v**2))
        
        # MAE
        mae_u = np.mean(np.abs(error_u))
        mae_v = np.mean(np.abs(error_v))
        
        # Magnitude errors
        mag_cfd = np.sqrt(u_cfd**2 + v_cfd**2)
        mag_piv = np.sqrt(u_piv**2 + v_piv**2)
        error_mag = mag_cfd - mag_piv
        
        rmse_mag = np.sqrt(np.mean(error_mag**2))
        mae_mag = np.mean(np.abs(error_mag))
        
        # Maximum error
        max_error = np.max(np.sqrt(error_u**2 + error_v**2))
        
        # Relative error
        mean_mag_piv = np.mean(mag_piv)
        if mean_mag_piv > 0:
            relative_error = rmse_mag / mean_mag_piv
        else:
            relative_error = np.inf
        
        return {
            "rmse_u": rmse_u,
            "rmse_v": rmse_v,
            "rmse_magnitude": rmse_mag,
            "mae_u": mae_u,
            "mae_v": mae_v,
            "mae_magnitude": mae_mag,
            "max_error": max_error,
            "relative_error": relative_error,
        }

    def calculate_regional_comparison(self, regions: Optional[list] = None) -> Dict[str, Dict]:
        """
        Compare CFD and PIV in specific regions.
        
        Args:
            regions: List of region definitions
            
        Returns:
            Dictionary of regional comparisons
        """
        if regions is None:
            # Default regions based on data bounds
            x_piv = self.piv_data["x"]
            y_piv = self.piv_data["y"]
            
            x_min, x_max = np.min(x_piv), np.max(x_piv)
            y_min, y_max = np.min(y_piv), np.max(y_piv)
            
            x_mid = (x_min + x_max) / 2
            y_mid = (y_min + y_max) / 2
            
            regions = [
                {"name": "full", "bounds": [x_min, x_max, y_min, y_max]},
                {"name": "left", "bounds": [x_min, x_mid, y_min, y_max]},
                {"name": "right", "bounds": [x_mid, x_max, y_min, y_max]},
                {"name": "bottom", "bounds": [x_min, x_max, y_min, y_mid]},
                {"name": "top", "bounds": [x_min, x_max, y_mid, y_max]},
            ]
        
        results = {}
        
        for region in regions:
            name = region["name"]
            bounds = region["bounds"]
            
            # Filter data in region
            x_piv = self.piv_data["x"].flatten()
            y_piv = self.piv_data["y"].flatten()
            
            mask = (
                (x_piv >= bounds[0]) & (x_piv <= bounds[1]) &
                (y_piv >= bounds[2]) & (y_piv <= bounds[3])
            )
            
            if np.sum(mask) > 0:
                # Create subset comparison
                cfd_subset = {
                    "x": self.cfd_data["x"],
                    "y": self.cfd_data["y"],
                    "u": self.cfd_data["u"],
                    "v": self.cfd_data["v"],
                }
                
                piv_subset = {
                    "x": self.piv_data["x"].flatten()[mask],
                    "y": self.piv_data["y"].flatten()[mask],
                    "u": self.piv_data.get("u_mean", self.piv_data["u"]).flatten()[mask],
                    "v": self.piv_data.get("v_mean", self.piv_data["v"]).flatten()[mask],
                }
                
                # Compare subset
                comparison = PIVComparison(cfd_subset, piv_subset)
                
                results[name] = {
                    "correlation": comparison.calculate_correlation(),
                    "errors": comparison.calculate_errors(),
                    "n_points": np.sum(mask),
                }
        
        return results

    def generate_report(self) -> Dict:
        """
        Generate comprehensive comparison report.
        
        Returns:
            Dictionary with all comparison metrics
        """
        report = {
            "correlation": self.calculate_correlation(),
            "errors": self.calculate_errors(),
            "regional": self.calculate_regional_comparison(),
        }
        
        # Add summary statistics
        report["summary"] = {
            "overall_correlation": report["correlation"]["magnitude_correlation"],
            "overall_rmse": report["errors"]["rmse_magnitude"],
            "overall_relative_error": report["errors"]["relative_error"],
            "validation_passed": (
                report["correlation"]["magnitude_correlation"] > 0.85 and
                report["errors"]["relative_error"] < 0.15
            ),
        }
        
        return report

    def save_report(self, output_path: Path) -> None:
        """
        Save comparison report to file.
        
        Args:
            output_path: Output file path
        """
        import json
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        report = self.generate_report()
        
        # Convert numpy types to Python types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(v) for v in obj]
            return obj
        
        report = convert_types(report)
        
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)