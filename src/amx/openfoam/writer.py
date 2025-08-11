"""OpenFOAM dictionary file writer using Jinja2 templates."""

import shutil
from pathlib import Path
from typing import Any, Dict, Optional

from jinja2 import Environment, FileSystemLoader, Template

from amx.config import Config


class DictWriter:
    """Write OpenFOAM dictionary files from templates."""

    def __init__(self, template_dir: Path):
        """
        Initialize dictionary writer.
        
        Args:
            template_dir: Path to template directory
        """
        self.template_dir = Path(template_dir)
        self.env = Environment(
            loader=FileSystemLoader(str(template_dir)),
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def write_control_dict(self, case_dir: Path, config: Config) -> None:
        """Write controlDict file."""
        template = self.env.get_template("system/controlDict")
        
        # Prepare plane sampling data
        planes = []
        if config.export and config.export.planes:
            for plane in config.export.planes:
                planes.append({"z": plane.get("z", 0)})
        
        context = {
            "solver": config.solver.name,
            "endTime": config.solver.endTime,
            "deltaT": config.solver.dt,
            "writeInterval": config.solver.writeInterval,
            "adjustTimeStep": "yes" if config.solver.adjustTimeStep else "no",
            "maxCo": config.solver.maxCo,
            "planes": planes,
            "sampleInterval": config.export.sample_interval if config.export else 100,
        }
        
        output_path = case_dir / "system" / "controlDict"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(template.render(**context))

    def write_fv_schemes(self, case_dir: Path) -> None:
        """Copy fvSchemes file."""
        src = self.template_dir / "system" / "fvSchemes"
        dst = case_dir / "system" / "fvSchemes"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    def write_fv_solution(self, case_dir: Path) -> None:
        """Copy fvSolution file."""
        src = self.template_dir / "system" / "fvSolution"
        dst = case_dir / "system" / "fvSolution"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    def write_turbulence_properties(self, case_dir: Path, config: Config) -> None:
        """Write turbulenceProperties file."""
        template = self.env.get_template("constant/turbulenceProperties")
        
        ras_model = "kEpsilon"
        if config.turbulence:
            ras_model = config.turbulence.model
        
        context = {"RASModel": ras_model}
        
        output_path = case_dir / "constant" / "turbulenceProperties"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(template.render(**context))

    def write_transport_properties(self, case_dir: Path, config: Config) -> None:
        """Write transportProperties file."""
        template = self.env.get_template("constant/transportProperties")
        
        context = {
            "nu": config.fluid.nu,
            "rho": config.fluid.rho,
        }
        
        output_path = case_dir / "constant" / "transportProperties"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(template.render(**context))

    def write_g(self, case_dir: Path) -> None:
        """Copy gravity file."""
        src = self.template_dir / "constant" / "g"
        dst = case_dir / "constant" / "g"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    def write_initial_conditions(self, case_dir: Path, config: Config) -> None:
        """Write initial condition files."""
        zero_dir = case_dir / "0"
        zero_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy field files
        for field in ["U", "p_rgh", "k", "epsilon", "nut"]:
            src = self.template_dir / "0" / field
            if src.exists():
                dst = zero_dir / field
                shutil.copy2(src, dst)

    def write_all_dicts(self, case_dir: Path, config: Config) -> None:
        """Write all dictionary files for a case."""
        # System files
        self.write_control_dict(case_dir, config)
        self.write_fv_schemes(case_dir)
        self.write_fv_solution(case_dir)
        
        # Constant files
        self.write_turbulence_properties(case_dir, config)
        self.write_transport_properties(case_dir, config)
        self.write_g(case_dir)
        
        # Initial conditions
        self.write_initial_conditions(case_dir, config)