"""Command-line interface for anaerobic mixing simulations."""

import logging
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.logging import RichHandler

from amx.config import load_config
from amx.utils.logging import setup_logging

app = typer.Typer(
    name="amx",
    help="Anaerobic Mixing with OpenFOAM - CLI for CFD simulations",
    add_completion=False,
)
console = Console()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(console=console, rich_tracebacks=True)],
)
logger = logging.getLogger(__name__)


@app.command()
def run_case(
    config: Path = typer.Option(..., "--config", "-c", help="Configuration file path"),
    output: Path = typer.Option(..., "--out", "-o", help="Output directory"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Perform dry run without execution"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
) -> None:
    """Run a complete OpenFOAM case from configuration."""
    setup_logging(verbose)
    
    try:
        console.print(f"[bold green]Loading configuration from {config}[/bold green]")
        cfg = load_config(config)
        
        console.print(f"[cyan]Project: {cfg.project}[/cyan]")
        console.print(f"[cyan]Output: {output}[/cyan]")
        
        if dry_run:
            console.print("[yellow]DRY RUN - No actual execution[/yellow]")
            return
            
        # Import here to avoid circular imports
        from amx.workflow import run_full_case as run_workflow
        
        run_workflow(cfg, output)
        console.print("[bold green]✓ Case completed successfully![/bold green]")
        
    except Exception as e:
        logger.error(f"Error running case: {e}")
        raise typer.Exit(code=1)


@app.command()
def analyze_mix(
    input_dir: Path = typer.Option(..., "--in", "-i", help="Input case directory"),
    output: Path = typer.Option(..., "--out", "-o", help="Output directory for results"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
) -> None:
    """Analyze mixing performance from simulation results."""
    setup_logging(verbose)
    
    try:
        console.print(f"[bold green]Analyzing mixing from {input_dir}[/bold green]")
        
        # Import here to avoid circular imports
        from amx.workflow import analyze_mixing
        
        metrics = analyze_mixing(input_dir, output)
        
        console.print("[cyan]Mixing Analysis Results:[/cyan]")
        console.print(f"  Mean velocity: {metrics.get('mean_velocity', 0):.3f} m/s")
        console.print(f"  Velocity variance: {metrics.get('velocity_variance', 0):.4f}")
        console.print(f"  Dead zones: {metrics.get('dead_zones_pct', 0):.1f}%")
        
        console.print("[bold green]✓ Analysis completed![/bold green]")
        
    except Exception as e:
        logger.error(f"Error analyzing mixing: {e}")
        raise typer.Exit(code=1)


@app.command()
def analyze_piv(
    config: Path = typer.Option(..., "--config", "-c", help="PIV configuration file"),
    cfd: Path = typer.Option(..., "--cfd", help="CFD results directory"),
    piv: Path = typer.Option(..., "--piv", help="PIV data directory"),
    output: Optional[Path] = typer.Option(None, "--out", "-o", help="Output directory"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
) -> None:
    """Compare CFD results with PIV measurements."""
    setup_logging(verbose)
    
    try:
        console.print("[bold green]PIV Validation[/bold green]")
        console.print(f"  Config: {config}")
        console.print(f"  CFD: {cfd}")
        console.print(f"  PIV: {piv}")
        
        # Import here to avoid circular imports
        from amx.workflow import piv_validation
        
        results = piv_validation(config, cfd, piv, output)
        
        console.print("[cyan]Validation Results:[/cyan]")
        console.print(f"  Correlation: {results.get('correlation', 0):.3f}")
        console.print(f"  RMSE: {results.get('rmse', 0):.4f} m/s")
        console.print(f"  Max error: {results.get('max_error', 0):.4f} m/s")
        
        console.print("[bold green]✓ PIV validation completed![/bold green]")
        
    except Exception as e:
        logger.error(f"Error in PIV validation: {e}")
        raise typer.Exit(code=1)


@app.command()
def make_figures(
    metrics: Path = typer.Option(..., "--metrics", "-m", help="Metrics data file"),
    output: Path = typer.Option(..., "--out", "-o", help="Output directory for figures"),
    format: str = typer.Option("png", "--format", "-f", help="Output format (png, pdf, svg)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
) -> None:
    """Generate figures from analysis results."""
    setup_logging(verbose)
    
    try:
        console.print(f"[bold green]Generating figures from {metrics}[/bold green]")
        
        # Import here to avoid circular imports
        from amx.workflow import generate_figures
        
        figures = generate_figures(metrics, output, format)
        
        console.print(f"[cyan]Generated {len(figures)} figures:[/cyan]")
        for fig in figures:
            console.print(f"  • {fig}")
            
        console.print("[bold green]✓ Figures generated![/bold green]")
        
    except Exception as e:
        logger.error(f"Error generating figures: {e}")
        raise typer.Exit(code=1)


@app.command()
def build_report(
    config: Path = typer.Option(..., "--config", "-c", help="Configuration file"),
    metrics: Path = typer.Option(..., "--metrics", "-m", help="Metrics data file"),
    output: Path = typer.Option(..., "--out", "-o", help="Output report file"),
    format: str = typer.Option("md", "--format", "-f", help="Output format (md, html, pdf)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
) -> None:
    """Build a comprehensive report from simulation results."""
    setup_logging(verbose)
    
    try:
        console.print("[bold green]Building report[/bold green]")
        console.print(f"  Config: {config}")
        console.print(f"  Metrics: {metrics}")
        console.print(f"  Output: {output}")
        
        # Import here to avoid circular imports
        from amx.workflow import build_report as build
        
        build(config, metrics, output, format)
        
        console.print(f"[bold green]✓ Report saved to {output}[/bold green]")
        
    except Exception as e:
        logger.error(f"Error building report: {e}")
        raise typer.Exit(code=1)


@app.command()
def version():
    """Show version information."""
    from amx import __version__
    
    console.print(f"[bold cyan]Anaerobic Mixing OpenFOAM v{__version__}[/bold cyan]")


if __name__ == "__main__":
    app()