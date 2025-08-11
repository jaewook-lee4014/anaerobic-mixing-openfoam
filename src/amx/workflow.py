"""Main workflow orchestration."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from amx.config import Config, load_config
from amx.energy import EnergyCalculator
from amx.openfoam import CaseRunner, DictWriter, FvOptionsWriter, MeshGenerator
from amx.piv import PIVComparison, PIVProcessor
from amx.post import MixingMetrics, VTKReader, Visualizer, create_matplotlib_figures
from amx.utils import create_case_structure, get_latest_time

logger = logging.getLogger(__name__)


def run_full_case(config: Config, output_dir: Path) -> Dict:
    """
    Run complete OpenFOAM case from configuration.
    
    Args:
        config: Simulation configuration
        output_dir: Output directory
        
    Returns:
        Dictionary with results
    """
    output_dir = Path(output_dir)
    case_dir = output_dir / "case"
    
    logger.info(f"Setting up case in {case_dir}")
    
    # Create case structure
    create_case_structure(case_dir)
    
    # Write dictionaries
    logger.info("Writing OpenFOAM dictionaries")
    writer = DictWriter(config.paths.template_root)
    writer.write_all_dicts(case_dir, config)
    
    # Write fvOptions
    logger.info("Writing fvOptions for momentum sources")
    fv_writer = FvOptionsWriter(config)
    fv_writer.write(case_dir)
    
    # Generate mesh
    logger.info("Generating mesh")
    mesher = MeshGenerator(config)
    if not mesher.generate_mesh(case_dir):
        raise RuntimeError("Mesh generation failed")
    
    # Run solver
    logger.info(f"Running {config.solver.name}")
    runner = CaseRunner(config, case_dir)
    if not runner.run_solver():
        raise RuntimeError("Solver failed")
    
    # Post-processing
    logger.info("Running post-processing")
    runner.run_post_processing()
    
    # Get results info
    results = runner.get_results_info()
    
    # Save configuration
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config.model_dump(), f, indent=2, default=str)
    
    results["config_path"] = str(config_path)
    results["output_dir"] = str(output_dir)
    
    return results


def analyze_mixing(case_dir: Path, output_dir: Path) -> Dict:
    """
    Analyze mixing performance from simulation results.
    
    Args:
        case_dir: Case directory with results
        output_dir: Output directory for analysis
        
    Returns:
        Dictionary with metrics
    """
    case_dir = Path(case_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Loading simulation results")
    
    # Load configuration if available
    config = None
    config_path = case_dir.parent / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config_data = json.load(f)
            config = Config(**config_data)
    
    # Read VTK data
    reader = VTKReader(case_dir / "case")
    
    # Get latest time
    latest_time = get_latest_time(case_dir / "case")
    if latest_time is None:
        raise ValueError("No results found in case directory")
    
    logger.info(f"Analyzing results at t={latest_time}s")
    
    # Load mesh
    mesh = reader.read_time_step(latest_time)
    
    # Calculate metrics
    metrics_calc = MixingMetrics(mesh, config)
    metrics = metrics_calc.get_summary_metrics()
    
    # Add regional metrics
    regional = metrics_calc.calculate_regional_metrics()
    metrics["regional"] = regional
    
    # Save metrics
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    
    logger.info(f"Metrics saved to {metrics_path}")
    
    # Create visualizations
    logger.info("Creating visualizations")
    viz = Visualizer(mesh)
    figures = viz.create_figure_set(output_dir / "figures")
    
    # Create matplotlib figures
    if "time_series" in metrics:
        create_matplotlib_figures(metrics, output_dir / "figures")
    
    metrics["figures"] = [str(f) for f in figures]
    
    return metrics


def piv_validation(config_path: Path, cfd_dir: Path, piv_dir: Path, 
                   output_dir: Optional[Path] = None) -> Dict:
    """
    Validate CFD results against PIV measurements.
    
    Args:
        config_path: PIV configuration file
        cfd_dir: CFD results directory
        piv_dir: PIV data directory
        output_dir: Output directory
        
    Returns:
        Validation results
    """
    config = load_config(config_path)
    
    if output_dir is None:
        output_dir = Path(cfd_dir) / "piv_validation"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Loading CFD results")
    
    # Load CFD data
    reader = VTKReader(cfd_dir)
    mesh = reader.read_time_step()
    
    # Extract velocity on PIV plane
    # This is simplified - actual implementation would need proper plane extraction
    cfd_data = {
        "x": mesh.points[:, 0],
        "y": mesh.points[:, 1],
        "u": mesh["U"][:, 0],
        "v": mesh["U"][:, 1],
    }
    
    logger.info("Loading PIV data")
    
    # Load PIV data (simplified - actual would read from files)
    piv_data_path = Path(piv_dir) / "piv_results.csv"
    if piv_data_path.exists():
        import pandas as pd
        df = pd.read_csv(piv_data_path)
        piv_data = {
            "x": df["x"].values,
            "y": df["y"].values,
            "u": df["u"].values,
            "v": df["v"].values,
        }
    else:
        # No PIV data available
        logger.warning("No PIV data found at %s", piv_data_path)
        logger.warning("PIV validation requires experimental data")
        return {
            "status": "no_data",
            "message": "PIV validation requires experimental data file",
            "required_file": str(piv_data_path)
        }
    
    logger.info("Comparing CFD and PIV")
    
    # Compare
    comparison = PIVComparison(cfd_data, piv_data)
    report = comparison.generate_report()
    
    # Save report
    report_path = output_dir / "validation_report.json"
    comparison.save_report(report_path)
    
    logger.info(f"Validation report saved to {report_path}")
    
    return report


def generate_figures(metrics_path: Path, output_dir: Path, format: str = "png") -> List[Path]:
    """
    Generate figures from analysis results.
    
    Args:
        metrics_path: Path to metrics file
        output_dir: Output directory for figures
        format: Output format
        
    Returns:
        List of generated figure paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load metrics
    with open(metrics_path) as f:
        metrics = json.load(f)
    
    # Create matplotlib figures
    figures = create_matplotlib_figures({"summary": metrics}, output_dir)
    
    return figures


def build_report(config_path: Path, metrics_path: Path, 
                output_path: Path, format: str = "md") -> None:
    """
    Build comprehensive report from simulation results.
    
    Args:
        config_path: Configuration file path
        metrics_path: Metrics file path
        output_path: Output report path
        format: Report format (md, html, pdf)
    """
    # Load data
    config = load_config(config_path)
    
    with open(metrics_path) as f:
        metrics = json.load(f)
    
    # Create report content
    report = f"""# Anaerobic Digester Mixing Analysis Report

## Project: {config.project}

### Configuration Summary
- Tank dimensions: {config.geometry.tank.L} × {config.geometry.tank.W} × {config.geometry.tank.H} m
- Tank volume: {config.geometry.tank.volume:.1f} m³
- Fluid temperature: {config.fluid.T - 273.15:.1f} °C
- Fluid viscosity: {config.fluid.mu:.4f} Pa·s
"""
    
    if config.operation:
        report += f"""
### Operation Parameters
- Total flow rate: {config.operation.pump_total_m3ph:.1f} m³/h
- Total dynamic head: {config.operation.head_m:.1f} m
- Nozzle configuration: {config.geometry.nozzle.count if config.geometry.nozzle else 0} nozzles
"""
    
    report += f"""
### Performance Metrics
- Mean velocity: {metrics.get('mean_velocity_mps', 0):.3f} m/s
- Velocity variance: {metrics.get('velocity_variance', 0):.4f}
- Uniformity index: {metrics.get('uniformity_index', 0):.3f}
- Dead zone fraction: {metrics.get('dead_zone_volume_percent', 0):.1f}%
- Mixing time estimate: {metrics.get('mixing_time_s', 0)/60:.1f} minutes
"""
    
    if 'power_w' in metrics:
        report += f"""
### Energy Metrics
- Pump power: {metrics['power_w']/1000:.1f} kW
- Energy density: {metrics.get('energy_density_w_m3', 0):.2f} W/m³
- G-value: {metrics.get('g_value_s-1', 0):.1f} s⁻¹
"""
    
    # Add regional analysis if available
    if 'regional' in metrics:
        report += "\n### Regional Analysis\n"
        for region, data in metrics['regional'].items():
            report += f"- {region}: {data.get('mean_velocity', 0):.3f} m/s\n"
    
    report += """
### Conclusions
"""
    
    # Check against targets
    if config.targets:
        if metrics.get('mean_velocity_mps', 0) >= config.targets.mean_velocity_mps:
            report += f"✓ Mean velocity target ({config.targets.mean_velocity_mps} m/s) achieved\n"
        else:
            report += f"✗ Mean velocity below target ({config.targets.mean_velocity_mps} m/s)\n"
        
        if metrics.get('dead_zone_volume_percent', 100) <= 10:
            report += "✓ Dead zone fraction acceptable (<10%)\n"
        else:
            report += "✗ Excessive dead zones (>10%)\n"
    
    report += "\n---\n*Generated with Anaerobic Mixing OpenFOAM*\n"
    
    # Save report
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == "md":
        output_path.write_text(report)
    elif format == "html":
        import markdown
        html = markdown.markdown(report)
        output_path.write_text(html)
    elif format == "pdf":
        # Would need pandoc or similar for PDF conversion
        logger.warning("PDF output not implemented, saving as markdown")
        output_path.with_suffix(".md").write_text(report)
    
    logger.info(f"Report saved to {output_path}")