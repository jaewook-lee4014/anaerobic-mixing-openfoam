"""Run OpenFOAM solvers and monitor progress."""

import logging
import subprocess
import time
from pathlib import Path
from typing import Dict, Optional

from amx.config import Config

logger = logging.getLogger(__name__)


class CaseRunner:
    """Run OpenFOAM cases and monitor progress."""

    def __init__(self, config: Config, case_dir: Path):
        """
        Initialize case runner.
        
        Args:
            config: Simulation configuration
            case_dir: Case directory path
        """
        self.config = config
        self.case_dir = Path(case_dir)
        self.solver_name = config.solver.name
        self.log_file = self.case_dir / f"log.{self.solver_name}"

    def check_case_setup(self) -> bool:
        """Check if case is properly set up."""
        required_dirs = [
            self.case_dir / "0",
            self.case_dir / "constant",
            self.case_dir / "system",
        ]
        
        required_files = [
            self.case_dir / "system" / "controlDict",
            self.case_dir / "system" / "fvSchemes",
            self.case_dir / "system" / "fvSolution",
            self.case_dir / "constant" / "polyMesh" / "points",
        ]
        
        for dir_path in required_dirs:
            if not dir_path.exists():
                logger.error(f"Missing directory: {dir_path}")
                return False
        
        for file_path in required_files:
            if not file_path.exists():
                logger.error(f"Missing file: {file_path}")
                return False
        
        return True

    def run_solver(self, parallel: bool = False, num_procs: int = 4) -> bool:
        """
        Run the OpenFOAM solver.
        
        Args:
            parallel: Run in parallel mode
            num_procs: Number of processors for parallel run
            
        Returns:
            True if successful, False otherwise
        """
        if not self.check_case_setup():
            logger.error("Case setup check failed")
            return False
        
        # Prepare command
        if parallel:
            # Decompose case
            if not self._decompose_case(num_procs):
                return False
            
            cmd = [
                "mpirun",
                "-np",
                str(num_procs),
                self.solver_name,
                "-parallel",
                "-case",
                str(self.case_dir),
            ]
        else:
            cmd = [self.solver_name, "-case", str(self.case_dir)]
        
        # Run solver
        logger.info(f"Running {self.solver_name}...")
        start_time = time.time()
        
        try:
            with open(self.log_file, "w") as log:
                process = subprocess.Popen(
                    cmd,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                
                # Monitor progress
                while process.poll() is None:
                    time.sleep(5)
                    self._print_progress()
                
                # Check return code
                if process.returncode != 0:
                    logger.error(f"{self.solver_name} failed with code {process.returncode}")
                    self._print_last_errors()
                    return False
        
        except Exception as e:
            logger.error(f"Error running solver: {e}")
            return False
        
        elapsed = time.time() - start_time
        logger.info(f"Solver completed in {elapsed:.1f} seconds")
        
        # Reconstruct if parallel
        if parallel:
            return self._reconstruct_case()
        
        return True

    def _decompose_case(self, num_procs: int) -> bool:
        """Decompose case for parallel run."""
        # Write decomposeParDict
        decompose_dict = self.case_dir / "system" / "decomposeParDict"
        decompose_dict.write_text(f"""
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      decomposeParDict;
}}

numberOfSubdomains {num_procs};
method          simple;
simpleCoeffs
{{
    n               ({num_procs} 1 1);
    delta           0.001;
}}
""")
        
        # Run decomposePar
        try:
            result = subprocess.run(
                ["decomposePar", "-case", str(self.case_dir)],
                capture_output=True,
                text=True,
                check=True,
            )
            logger.info("Case decomposed successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"decomposePar failed: {e.stderr}")
            return False

    def _reconstruct_case(self) -> bool:
        """Reconstruct case after parallel run."""
        try:
            result = subprocess.run(
                ["reconstructPar", "-case", str(self.case_dir)],
                capture_output=True,
                text=True,
                check=True,
            )
            logger.info("Case reconstructed successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"reconstructPar failed: {e.stderr}")
            return False

    def _print_progress(self) -> None:
        """Print current simulation progress."""
        if not self.log_file.exists():
            return
        
        # Read last few lines of log
        with open(self.log_file, "r") as f:
            lines = f.readlines()
            
        # Look for time step info
        for line in reversed(lines[-20:]):
            if "Time =" in line:
                logger.info(f"Progress: {line.strip()}")
                break

    def _print_last_errors(self) -> None:
        """Print last errors from log file."""
        if not self.log_file.exists():
            return
        
        with open(self.log_file, "r") as f:
            lines = f.readlines()
        
        # Look for error messages
        error_lines = []
        for line in lines[-50:]:
            if "error" in line.lower() or "fatal" in line.lower():
                error_lines.append(line.strip())
        
        if error_lines:
            logger.error("Last errors from log:")
            for line in error_lines[-10:]:
                logger.error(f"  {line}")

    def run_post_processing(self) -> bool:
        """Run post-processing utilities."""
        utilities = [
            ("sample", "-latestTime"),
            ("foamToVTK", "-latestTime"),
        ]
        
        for utility, *args in utilities:
            cmd = [utility, "-case", str(self.case_dir)] + list(args)
            
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=True,
                )
                logger.info(f"{utility} completed successfully")
            except subprocess.CalledProcessError as e:
                logger.warning(f"{utility} failed: {e.stderr}")
                # Continue with other utilities
        
        return True

    def get_results_info(self) -> Dict:
        """Get information about simulation results."""
        info = {
            "case_dir": str(self.case_dir),
            "solver": self.solver_name,
            "completed": False,
            "final_time": 0,
            "time_steps": [],
        }
        
        # Check for time directories
        time_dirs = []
        for item in self.case_dir.iterdir():
            if item.is_dir() and item.name.replace(".", "").isdigit():
                try:
                    time_val = float(item.name)
                    time_dirs.append(time_val)
                except ValueError:
                    pass
        
        if time_dirs:
            time_dirs.sort()
            info["time_steps"] = time_dirs
            info["final_time"] = time_dirs[-1]
            info["completed"] = info["final_time"] >= self.config.solver.endTime * 0.99
        
        return info