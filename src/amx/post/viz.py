"""Visualization utilities for CFD results."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from matplotlib import cm

from amx.post.fields import FieldProcessor


class Visualizer:
    """Create visualizations of CFD results."""

    def __init__(self, mesh: pv.UnstructuredGrid):
        """
        Initialize visualizer.
        
        Args:
            mesh: PyVista mesh with field data
        """
        self.mesh = mesh
        self.processor = FieldProcessor(mesh)

    def plot_velocity_contours(self, 
                              plane_normal: str = "z",
                              plane_value: Optional[float] = None,
                              save_path: Optional[Path] = None) -> pv.Plotter:
        """
        Plot velocity magnitude contours on a plane.
        
        Args:
            plane_normal: Normal direction ('x', 'y', or 'z')
            plane_value: Position along normal (center if None)
            save_path: Path to save image
            
        Returns:
            PyVista plotter
        """
        # Get velocity magnitude
        U_mag = self.processor.get_velocity_magnitude()
        self.mesh["U_mag"] = U_mag
        
        # Create slice
        normal = {"x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1]}[plane_normal]
        
        if plane_value is None:
            origin = self.mesh.center
        else:
            origin = list(self.mesh.center)
            idx = {"x": 0, "y": 1, "z": 2}[plane_normal]
            origin[idx] = plane_value
        
        slice_mesh = self.mesh.slice(normal=normal, origin=origin)
        
        # Create plot
        plotter = pv.Plotter(off_screen=save_path is not None)
        plotter.add_mesh(
            slice_mesh,
            scalars="U_mag",
            cmap="jet",
            scalar_bar_args={"title": "Velocity (m/s)"},
        )
        plotter.add_title(f"Velocity Contours ({plane_normal}={plane_value:.2f}m)")
        plotter.show_axes()
        plotter.view_xy() if plane_normal == "z" else plotter.view_xz()
        
        if save_path:
            plotter.screenshot(str(save_path))
        
        return plotter

    def plot_velocity_vectors(self,
                            plane_normal: str = "z",
                            plane_value: Optional[float] = None,
                            scale_factor: float = 0.1,
                            save_path: Optional[Path] = None) -> pv.Plotter:
        """
        Plot velocity vectors on a plane.
        
        Args:
            plane_normal: Normal direction ('x', 'y', or 'z')
            plane_value: Position along normal
            scale_factor: Vector scaling factor
            save_path: Path to save image
            
        Returns:
            PyVista plotter
        """
        # Create slice
        normal = {"x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1]}[plane_normal]
        
        if plane_value is None:
            origin = self.mesh.center
        else:
            origin = list(self.mesh.center)
            idx = {"x": 0, "y": 1, "z": 2}[plane_normal]
            origin[idx] = plane_value
        
        slice_mesh = self.mesh.slice(normal=normal, origin=origin)
        
        # Decimate for cleaner vector plot
        if slice_mesh.n_points > 1000:
            slice_mesh = slice_mesh.decimate(0.1)
        
        # Create plot
        plotter = pv.Plotter(off_screen=save_path is not None)
        
        # Add velocity magnitude as background
        U_mag = np.linalg.norm(slice_mesh["U"], axis=1)
        plotter.add_mesh(
            slice_mesh,
            scalars=U_mag,
            cmap="Blues",
            opacity=0.5,
            show_scalar_bar=False,
        )
        
        # Add vectors
        plotter.add_arrows(
            slice_mesh.points,
            slice_mesh["U"],
            mag=scale_factor,
            color="black",
        )
        
        plotter.add_title(f"Velocity Vectors ({plane_normal}={plane_value:.2f}m)")
        plotter.show_axes()
        plotter.view_xy() if plane_normal == "z" else plotter.view_xz()
        
        if save_path:
            plotter.screenshot(str(save_path))
        
        return plotter

    def plot_streamlines(self,
                        n_streamlines: int = 50,
                        save_path: Optional[Path] = None) -> pv.Plotter:
        """
        Plot 3D streamlines.
        
        Args:
            n_streamlines: Number of streamlines
            save_path: Path to save image
            
        Returns:
            PyVista plotter
        """
        # Create seed points
        bounds = self.mesh.bounds
        seeds = pv.PolyData(np.random.default_rng().uniform(
            [bounds[0], bounds[2], bounds[4]],
            [bounds[1], bounds[3], bounds[5]],
            (n_streamlines, 3)
        ))
        
        # Generate streamlines
        streamlines = self.mesh.streamlines(
            vectors="U",
            source=seeds,
            max_time=100,
            integration_direction="both",
        )
        
        # Get velocity magnitude for coloring
        U_mag = np.linalg.norm(streamlines["U"], axis=1)
        
        # Create plot
        plotter = pv.Plotter(off_screen=save_path is not None)
        
        # Add tank outline
        plotter.add_mesh(
            self.mesh.outline(),
            color="black",
            line_width=2,
        )
        
        # Add streamlines
        plotter.add_mesh(
            streamlines,
            scalars=U_mag,
            cmap="jet",
            line_width=2,
            scalar_bar_args={"title": "Velocity (m/s)"},
        )
        
        plotter.add_title("Flow Streamlines")
        plotter.show_axes()
        plotter.view_isometric()
        
        if save_path:
            plotter.screenshot(str(save_path))
        
        return plotter

    def plot_turbulence(self,
                       field: str = "k",
                       plane_normal: str = "z",
                       plane_value: Optional[float] = None,
                       save_path: Optional[Path] = None) -> pv.Plotter:
        """
        Plot turbulence field.
        
        Args:
            field: Turbulence field ('k' or 'epsilon')
            plane_normal: Normal direction
            plane_value: Position along normal
            save_path: Path to save image
            
        Returns:
            PyVista plotter
        """
        if field not in self.mesh.array_names:
            raise ValueError(f"Field '{field}' not found in mesh")
        
        # Create slice
        normal = {"x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1]}[plane_normal]
        
        if plane_value is None:
            origin = self.mesh.center
        else:
            origin = list(self.mesh.center)
            idx = {"x": 0, "y": 1, "z": 2}[plane_normal]
            origin[idx] = plane_value
        
        slice_mesh = self.mesh.slice(normal=normal, origin=origin)
        
        # Create plot
        plotter = pv.Plotter(off_screen=save_path is not None)
        
        title_map = {
            "k": "Turbulent Kinetic Energy (m²/s²)",
            "epsilon": "Dissipation Rate (m²/s³)",
            "nut": "Eddy Viscosity (m²/s)",
        }
        
        plotter.add_mesh(
            slice_mesh,
            scalars=field,
            cmap="viridis",
            scalar_bar_args={"title": title_map.get(field, field)},
        )
        
        plotter.add_title(f"{title_map.get(field, field)} ({plane_normal}={plane_value:.2f}m)")
        plotter.show_axes()
        plotter.view_xy() if plane_normal == "z" else plotter.view_xz()
        
        if save_path:
            plotter.screenshot(str(save_path))
        
        return plotter

    def plot_dead_zones(self,
                       threshold: float = 0.05,
                       save_path: Optional[Path] = None) -> pv.Plotter:
        """
        Plot dead zones in 3D.
        
        Args:
            threshold: Velocity threshold for dead zones (m/s)
            save_path: Path to save image
            
        Returns:
            PyVista plotter
        """
        dead_zones, fraction = self.processor.identify_dead_zones(threshold)
        
        # Create plot
        plotter = pv.Plotter(off_screen=save_path is not None)
        
        # Add tank outline
        plotter.add_mesh(
            self.mesh.outline(),
            color="black",
            line_width=2,
        )
        
        # Add dead zones
        if dead_zones.n_cells > 0:
            plotter.add_mesh(
                dead_zones,
                color="red",
                opacity=0.7,
                label=f"Dead zones ({fraction*100:.1f}%)",
            )
        
        # Add active zones
        active_zones = self.mesh.threshold(value=threshold, scalars="U_mag", invert=False)
        if active_zones.n_cells > 0:
            plotter.add_mesh(
                active_zones,
                color="blue",
                opacity=0.3,
                label="Active zones",
            )
        
        plotter.add_title(f"Dead Zones (|U| < {threshold} m/s)")
        plotter.add_legend()
        plotter.show_axes()
        plotter.view_isometric()
        
        if save_path:
            plotter.screenshot(str(save_path))
        
        return plotter

    def create_figure_set(self, output_dir: Path, format: str = "png") -> List[Path]:
        """
        Create a standard set of figures.
        
        Args:
            output_dir: Output directory
            format: Image format
            
        Returns:
            List of created file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        figures = []
        
        # Velocity contours at different heights
        for z_rel in [0.1, 0.5, 0.9]:
            z = self.mesh.bounds[4] + z_rel * (self.mesh.bounds[5] - self.mesh.bounds[4])
            path = output_dir / f"velocity_z{z_rel:.1f}.{format}"
            self.plot_velocity_contours("z", z, path)
            figures.append(path)
        
        # Streamlines
        path = output_dir / f"streamlines.{format}"
        self.plot_streamlines(save_path=path)
        figures.append(path)
        
        # Dead zones
        path = output_dir / f"dead_zones.{format}"
        self.plot_dead_zones(save_path=path)
        figures.append(path)
        
        # Turbulence if available
        if "k" in self.mesh.array_names:
            z = self.mesh.center[2]
            path = output_dir / f"turbulence_k.{format}"
            self.plot_turbulence("k", "z", z, path)
            figures.append(path)
        
        return figures


def create_matplotlib_figures(data: Dict, output_dir: Path) -> List[Path]:
    """
    Create matplotlib figures from metrics data.
    
    Args:
        data: Dictionary of metrics
        output_dir: Output directory
        
    Returns:
        List of created file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    figures = []
    
    # Time series plot if available
    if "time_series" in data:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        time = data["time_series"]["time"]
        
        # Mean velocity
        if "mean_velocity" in data["time_series"]:
            axes[0, 0].plot(time, data["time_series"]["mean_velocity"])
            axes[0, 0].set_xlabel("Time (s)")
            axes[0, 0].set_ylabel("Mean Velocity (m/s)")
            axes[0, 0].grid(True)
            axes[0, 0].set_title("Mean Velocity Evolution")
        
        # Dead zones
        if "dead_zone_fraction" in data["time_series"]:
            axes[0, 1].plot(time, np.array(data["time_series"]["dead_zone_fraction"]) * 100)
            axes[0, 1].set_xlabel("Time (s)")
            axes[0, 1].set_ylabel("Dead Zone (%)")
            axes[0, 1].grid(True)
            axes[0, 1].set_title("Dead Zone Evolution")
        
        # Energy
        if "energy_density" in data["time_series"]:
            axes[1, 0].plot(time, data["time_series"]["energy_density"])
            axes[1, 0].set_xlabel("Time (s)")
            axes[1, 0].set_ylabel("Energy Density (W/m³)")
            axes[1, 0].grid(True)
            axes[1, 0].set_title("Energy Density")
        
        # G-value
        if "g_value" in data["time_series"]:
            axes[1, 1].plot(time, data["time_series"]["g_value"])
            axes[1, 1].set_xlabel("Time (s)")
            axes[1, 1].set_ylabel("G-value (s⁻¹)")
            axes[1, 1].grid(True)
            axes[1, 1].set_title("Velocity Gradient")
        
        plt.tight_layout()
        path = output_dir / "time_series.png"
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        figures.append(path)
    
    # Bar chart of final metrics
    if "summary" in data:
        metrics = data["summary"]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        keys = list(metrics.keys())[:8]  # Top 8 metrics
        values = [metrics[k] for k in keys]
        
        bars = ax.bar(range(len(keys)), values)
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels(keys, rotation=45, ha="right")
        ax.set_ylabel("Value")
        ax.set_title("Mixing Performance Metrics")
        ax.grid(True, axis="y")
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        path = output_dir / "metrics_summary.png"
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        figures.append(path)
    
    return figures