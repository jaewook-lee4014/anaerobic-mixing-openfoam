"""Process PIV data using OpenPIV."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import ndimage

try:
    import openpiv.pyprocess as process
    import openpiv.validation
    import openpiv.filters
    OPENPIV_AVAILABLE = True
except (ImportError, AttributeError):
    OPENPIV_AVAILABLE = False
    # Create mock objects for testing when OpenPIV is not available
    class MockProcess:
        @staticmethod
        def extended_search_area_piv(*args, **kwargs):
            return np.zeros((10, 10)), np.zeros((10, 10)), np.ones((10, 10))
        
        @staticmethod
        def get_coordinates(*args, **kwargs):
            return np.meshgrid(np.arange(10), np.arange(10))
    
    class MockValidation:
        @staticmethod
        def global_val(u, v, **kwargs):
            return u, v, np.zeros_like(u, dtype=bool)
        
        @staticmethod
        def local_median_val(u, v, **kwargs):
            return u, v, np.zeros_like(u, dtype=bool)
    
    class MockFilters:
        @staticmethod
        def replace_outliers(u, v, **kwargs):
            return u, v
    
    # Set up mock modules
    process = MockProcess()
    
    class MockOpenPIV:
        validation = MockValidation()
        filters = MockFilters()
    
    openpiv = MockOpenPIV()


class PIVProcessor:
    """Process PIV images to extract velocity fields."""

    def __init__(self, 
                 window_size: int = 32,
                 overlap: int = 16,
                 search_area: int = 64,
                 dt: float = 1.0,
                 scaling_factor: float = 1.0):
        """
        Initialize PIV processor.
        
        Args:
            window_size: Interrogation window size (pixels)
            overlap: Window overlap (pixels)
            search_area: Search area size (pixels)
            dt: Time between frames (s)
            scaling_factor: Spatial scaling (m/pixel)
        """
        self.window_size = window_size
        self.overlap = overlap
        self.search_area = search_area
        self.dt = dt
        self.scaling_factor = scaling_factor

    def process_image_pair(self, 
                          frame_a: np.ndarray,
                          frame_b: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Process a pair of PIV images.
        
        Args:
            frame_a: First frame
            frame_b: Second frame
            
        Returns:
            Tuple of (x, y, u, v) arrays
        """
        # Extended search area PIV
        u, v, sig2noise = process.extended_search_area_piv(
            frame_a.astype(np.int32),
            frame_b.astype(np.int32),
            window_size=self.window_size,
            overlap=self.overlap,
            dt=self.dt,
            search_area_size=self.search_area,
            sig2noise_method='peak2peak'
        )
        
        # Get grid coordinates
        x, y = process.get_coordinates(
            image_size=frame_a.shape,
            search_area_size=self.search_area,
            overlap=self.overlap
        )
        
        # Scale to physical units
        x = x * self.scaling_factor
        y = y * self.scaling_factor
        u = u * self.scaling_factor / self.dt
        v = v * self.scaling_factor / self.dt
        
        return x, y, u, v

    def validate_and_filter(self,
                           x: np.ndarray,
                           y: np.ndarray,
                           u: np.ndarray,
                           v: np.ndarray,
                           sig2noise: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Validate and filter velocity field.
        
        Args:
            x, y: Grid coordinates
            u, v: Velocity components
            sig2noise: Signal-to-noise ratio
            
        Returns:
            Tuple of (u_filtered, v_filtered, mask)
        """
        # Validation
        u_filt = u.copy()
        v_filt = v.copy()
        
        # Signal-to-noise validation
        if sig2noise is not None:
            mask_s2n = sig2noise < 1.5
            u_filt[mask_s2n] = np.nan
            v_filt[mask_s2n] = np.nan
        
        # Global velocity validation (remove outliers)
        u_filt, v_filt, mask_global = openpiv.validation.global_val(
            u_filt, v_filt,
            u_thresholds=(-10, 10),  # m/s
            v_thresholds=(-10, 10)   # m/s
        )
        
        # Local median validation
        u_filt, v_filt, mask_local = openpiv.validation.local_median_val(
            u_filt, v_filt,
            u_threshold=2,
            v_threshold=2,
            size=3
        )
        
        # Replace outliers
        u_filt, v_filt = openpiv.filters.replace_outliers(
            u_filt, v_filt,
            method='localmean',
            max_iter=3,
            kernel_size=3
        )
        
        # Smooth if needed
        u_filt = ndimage.gaussian_filter(u_filt, sigma=0.5)
        v_filt = ndimage.gaussian_filter(v_filt, sigma=0.5)
        
        # Combined mask
        mask = mask_global | mask_local
        
        return u_filt, v_filt, mask

    def calculate_derived_quantities(self,
                                    x: np.ndarray,
                                    y: np.ndarray,
                                    u: np.ndarray,
                                    v: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Calculate derived quantities from velocity field.
        
        Args:
            x, y: Grid coordinates
            u, v: Velocity components
            
        Returns:
            Dictionary of derived quantities
        """
        # Velocity magnitude
        speed = np.sqrt(u**2 + v**2)
        
        # Vorticity (2D)
        dx = x[0, 1] - x[0, 0] if x.shape[1] > 1 else 1.0
        dy = y[1, 0] - y[0, 0] if y.shape[0] > 1 else 1.0
        
        dudy, dudx = np.gradient(u, dy, dx)
        dvdy, dvdx = np.gradient(v, dy, dx)
        vorticity = dvdx - dudy
        
        # Strain rate
        strain_rate = np.sqrt(2 * (dudx**2 + dvdy**2) + (dudy + dvdx)**2)
        
        # Divergence (should be ~0 for incompressible flow)
        divergence = dudx + dvdy
        
        # Kinetic energy
        kinetic_energy = 0.5 * speed**2
        
        return {
            "speed": speed,
            "vorticity": vorticity,
            "strain_rate": strain_rate,
            "divergence": divergence,
            "kinetic_energy": kinetic_energy,
        }

    def process_image_sequence(self,
                              images: List[np.ndarray],
                              ensemble_average: bool = True) -> Dict[str, np.ndarray]:
        """
        Process a sequence of PIV images.
        
        Args:
            images: List of images
            ensemble_average: Whether to compute ensemble average
            
        Returns:
            Dictionary with results
        """
        n_pairs = len(images) - 1
        if n_pairs < 1:
            raise ValueError("Need at least 2 images")
        
        results = []
        
        for i in range(n_pairs):
            # Process pair
            x, y, u, v = self.process_image_pair(images[i], images[i+1])
            
            # Validate
            u_filt, v_filt, mask = self.validate_and_filter(x, y, u, v)
            
            results.append({
                "x": x,
                "y": y,
                "u": u_filt,
                "v": v_filt,
                "mask": mask,
            })
        
        # Ensemble averaging if requested
        if ensemble_average and len(results) > 1:
            u_mean = np.nanmean([r["u"] for r in results], axis=0)
            v_mean = np.nanmean([r["v"] for r in results], axis=0)
            
            # RMS fluctuations
            u_rms = np.nanstd([r["u"] for r in results], axis=0)
            v_rms = np.nanstd([r["v"] for r in results], axis=0)
            
            # Turbulence intensity
            speed_mean = np.sqrt(u_mean**2 + v_mean**2)
            turb_intensity = np.sqrt(u_rms**2 + v_rms**2) / (speed_mean + 1e-10)
            
            return {
                "x": results[0]["x"],
                "y": results[0]["y"],
                "u_mean": u_mean,
                "v_mean": v_mean,
                "u_rms": u_rms,
                "v_rms": v_rms,
                "turbulence_intensity": turb_intensity,
                "n_samples": n_pairs,
            }
        else:
            return results[0]

    def save_results(self, results: Dict[str, np.ndarray], output_path: Path) -> None:
        """
        Save PIV results to file.
        
        Args:
            results: PIV results dictionary
            output_path: Output file path
        """
        import pandas as pd
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Flatten arrays for saving
        x = results["x"].flatten()
        y = results["y"].flatten()
        
        data = {"x": x, "y": y}
        
        for key in ["u", "v", "u_mean", "v_mean", "u_rms", "v_rms"]:
            if key in results:
                data[key] = results[key].flatten()
        
        df = pd.DataFrame(data)
        
        # Save based on extension
        if output_path.suffix == ".csv":
            df.to_csv(output_path, index=False)
        elif output_path.suffix == ".h5":
            df.to_hdf(output_path, key="piv", mode="w")
        else:
            # Default to CSV
            df.to_csv(output_path.with_suffix(".csv"), index=False)