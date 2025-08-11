"""Configuration management using Pydantic models."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from pydantic import BaseModel, Field, field_validator


class PathConfig(BaseModel):
    """Path configuration."""

    case_root: Path
    template_root: Path = Path("case_templates")
    output_root: Path = Path("data/processed")

    @field_validator("case_root", "template_root", "output_root", mode="before")
    def convert_to_path(cls, v):
        return Path(v) if not isinstance(v, Path) else v


class TankConfig(BaseModel):
    """Tank geometry configuration."""

    L: float = Field(gt=0, description="Tank length (m)")
    W: float = Field(gt=0, description="Tank width (m)")
    H: float = Field(gt=0, description="Tank height (m)")

    @property
    def volume(self) -> float:
        """Calculate tank volume in m³."""
        return self.L * self.W * self.H


class NozzleConfig(BaseModel):
    """Nozzle configuration."""

    count: int = Field(gt=0, description="Number of nozzles")
    throat_diameter_mm: float = Field(gt=0, description="Nozzle throat diameter (mm)")
    pitch_deg: float = Field(ge=0, le=90, description="Upward pitch angle (degrees)")
    array: Tuple[int, int] = Field(description="Nozzle array arrangement [rows, cols]")
    start_offset_m: Tuple[float, float] = Field(description="Offset from walls [x, y]")
    spacing_m: Optional[Tuple[float, float]] = Field(None, description="Spacing between nozzles")
    zone_radius_m: float = Field(default=0.2, gt=0, description="Momentum source zone radius")
    zone_height_m: float = Field(default=0.5, gt=0, description="Momentum source zone height")

    @property
    def throat_area_m2(self) -> float:
        """Calculate nozzle throat area in m²."""
        d_m = self.throat_diameter_mm / 1000
        return 3.14159 * (d_m / 2) ** 2


class ScreenConfig(BaseModel):
    """Screen/porous zone configuration."""

    type: str = "porous"
    center: Tuple[float, float, float] = Field(description="Center position [x, y, z]")
    radius: float = Field(gt=0, description="Screen radius (m)")
    height: float = Field(gt=0, description="Screen height (m)")
    darcy: Tuple[float, float, float] = Field(description="Darcy coefficients [dx, dy, dz]")
    forchheimer: Tuple[float, float, float] = Field(
        description="Forchheimer coefficients [fx, fy, fz]"
    )


class MeshConfig(BaseModel):
    """Mesh configuration."""

    base_cell_size: float = Field(gt=0, description="Base cell size (m)")
    refinement_levels: int = Field(ge=0, le=5, default=2)
    wall_layers: int = Field(ge=0, default=3)
    refinement_zones: Optional[List[Dict[str, Any]]] = None


class FluidConfig(BaseModel):
    """Fluid properties configuration."""

    T: float = Field(gt=0, description="Temperature (K)")
    rho: float = Field(gt=0, description="Density (kg/m³)")
    mu: float = Field(gt=0, description="Dynamic viscosity (Pa·s)")
    nu: float = Field(gt=0, description="Kinematic viscosity (m²/s)")

    @field_validator("nu")
    def check_nu_consistency(cls, v, info):
        """Check that nu = mu/rho."""
        if "mu" in info.data and "rho" in info.data:
            expected_nu = info.data["mu"] / info.data["rho"]
            # Allow 1% tolerance for floating point differences
            if abs(v - expected_nu) / expected_nu > 0.01:
                raise ValueError(f"Inconsistent nu: expected {expected_nu}, got {v}")
        return v


class OperationConfig(BaseModel):
    """Operation parameters configuration."""

    pump_total_m3ph: float = Field(gt=0, description="Total pump flow rate (m³/h)")
    head_m: float = Field(gt=0, description="Total dynamic head (m)")
    nozzle_split: str = Field(default="equal", description="Flow distribution method")
    nozzle_flow_m3ph: Optional[float] = None
    jet_velocity_mps: Optional[float] = None
    schedule: Optional[Dict[str, Any]] = None

    def calculate_nozzle_flow(self, nozzle_count: int) -> float:
        """Calculate flow per nozzle."""
        if self.nozzle_flow_m3ph is not None:
            return self.nozzle_flow_m3ph
        return self.pump_total_m3ph / nozzle_count if nozzle_count > 0 else 0


class SolverConfig(BaseModel):
    """Solver configuration."""

    name: str = Field(default="pimpleFoam", description="OpenFOAM solver name")
    dt: float = Field(gt=0, description="Time step (s)")
    endTime: float = Field(gt=0, description="End time (s)")
    writeInterval: int = Field(gt=0, description="Write interval (timesteps)")
    adjustTimeStep: bool = Field(default=False)
    maxCo: float = Field(gt=0, default=1.0, description="Maximum Courant number")


class TurbulenceConfig(BaseModel):
    """Turbulence model configuration."""

    model: str = Field(default="kEpsilon", description="Turbulence model")
    k_init: float = Field(gt=0, default=0.01, description="Initial k value")
    epsilon_init: float = Field(gt=0, default=0.001, description="Initial epsilon value")


class ExportConfig(BaseModel):
    """Export configuration."""

    planes: Optional[List[Dict[str, float]]] = None
    fields: List[str] = Field(default=["U", "k", "epsilon", "nut", "p_rgh"])
    sample_interval: int = Field(gt=0, default=100)
    format: str = Field(default="vtk")


class TargetConfig(BaseModel):
    """Target performance metrics."""

    mean_velocity_mps: float = Field(gt=0, description="Target mean velocity (m/s)")
    mlss_deviation_pct: float = Field(gt=0, description="Target MLSS deviation (%)")
    mixing_time_min: float = Field(gt=0, description="Target mixing time (minutes)")


class EnergyConfig(BaseModel):
    """Energy calculation parameters."""

    pump_efficiency: float = Field(gt=0, le=1, default=0.65)
    motor_efficiency: float = Field(gt=0, le=1, default=0.90)


class GeometryConfig(BaseModel):
    """Complete geometry configuration."""

    tank: TankConfig
    nozzle: Optional[NozzleConfig] = None
    screen: Optional[ScreenConfig] = None
    mesh: Optional[MeshConfig] = None


class Config(BaseModel):
    """Complete simulation configuration."""

    project: str
    paths: PathConfig
    geometry: GeometryConfig
    fluid: FluidConfig
    operation: Optional[OperationConfig] = None
    solver: SolverConfig
    turbulence: Optional[TurbulenceConfig] = None
    export: Optional[ExportConfig] = None
    targets: Optional[TargetConfig] = None
    energy: Optional[EnergyConfig] = None


def load_config(config_path: Path) -> Config:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        data = yaml.safe_load(f)
    return Config(**data)


def save_config(config: Config, output_path: Path) -> None:
    """Save configuration to YAML file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(config.model_dump(), f, default_flow_style=False, sort_keys=False)