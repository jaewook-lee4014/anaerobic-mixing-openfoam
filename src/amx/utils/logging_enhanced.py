"""Enhanced logging system with structured logging and CFD-specific features."""

import logging
import sys
import json
import time
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
from contextlib import contextmanager


class CFDLogHandler(logging.Handler):
    """Custom handler for CFD simulation logs."""
    
    def __init__(self, log_dir: Path):
        """Initialize CFD log handler."""
        super().__init__()
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Separate files for different log types
        self.files = {
            'convergence': self.log_dir / 'convergence.log',
            'residuals': self.log_dir / 'residuals.json',
            'performance': self.log_dir / 'performance.json',
            'errors': self.log_dir / 'errors.log'
        }
        
        self.iteration_data = []
        self.performance_data = []
    
    def emit(self, record: logging.LogRecord):
        """Handle log record based on type."""
        try:
            # Check for CFD-specific data
            if hasattr(record, 'cfd_type'):
                self._handle_cfd_record(record)
            elif record.levelno >= logging.ERROR:
                self._handle_error(record)
        except Exception:
            self.handleError(record)
    
    def _handle_cfd_record(self, record: logging.LogRecord):
        """Handle CFD-specific log records."""
        cfd_type = record.cfd_type
        
        if cfd_type == 'iteration':
            self.iteration_data.append({
                'time': record.created,
                'iteration': record.iteration,
                'residuals': record.residuals,
                'continuity': record.continuity
            })
            
            # Write every 10 iterations
            if record.iteration % 10 == 0:
                self._save_residuals()
        
        elif cfd_type == 'convergence':
            with open(self.files['convergence'], 'a') as f:
                f.write(f"{datetime.now()}: {record.getMessage()}\n")
        
        elif cfd_type == 'performance':
            self.performance_data.append({
                'time': record.created,
                'metric': record.metric,
                'value': record.value
            })
    
    def _handle_error(self, record: logging.LogRecord):
        """Log errors to separate file."""
        with open(self.files['errors'], 'a') as f:
            f.write(f"{datetime.now()}: {self.format(record)}\n")
    
    def _save_residuals(self):
        """Save residual history to JSON."""
        if self.iteration_data:
            with open(self.files['residuals'], 'w') as f:
                json.dump(self.iteration_data, f, indent=2)
    
    def close(self):
        """Save all data on close."""
        self._save_residuals()
        
        if self.performance_data:
            with open(self.files['performance'], 'w') as f:
                json.dump(self.performance_data, f, indent=2)
        
        super().close()


class SimulationLogger:
    """Logger for CFD simulation progress and results."""
    
    def __init__(self, case_name: str, log_dir: Optional[Path] = None):
        """
        Initialize simulation logger.
        
        Args:
            case_name: Name of simulation case
            log_dir: Directory for log files
        """
        self.case_name = case_name
        self.logger = logging.getLogger(f'amx.simulation.{case_name}')
        
        if log_dir:
            handler = CFDLogHandler(log_dir)
            self.logger.addHandler(handler)
        
        self.start_time = None
        self.metrics = {}
    
    def start_simulation(self):
        """Log simulation start."""
        self.start_time = time.time()
        self.logger.info(f"Starting simulation: {self.case_name}")
        self.logger.info(f"Timestamp: {datetime.now()}")
    
    def log_iteration(self, iteration: int, residuals: Dict[str, float], 
                     continuity: float):
        """
        Log solver iteration.
        
        Args:
            iteration: Iteration number
            residuals: Dictionary of residuals
            continuity: Continuity error
        """
        max_res = max(residuals.values())
        
        # Log with CFD-specific data
        self.logger.debug(
            f"Iter {iteration}: Max res = {max_res:.2e}, Cont = {continuity:.2e}",
            extra={
                'cfd_type': 'iteration',
                'iteration': iteration,
                'residuals': residuals,
                'continuity': continuity
            }
        )
        
        # Check convergence
        if max_res < 1e-5:
            self.logger.info(
                f"Convergence achieved at iteration {iteration}",
                extra={'cfd_type': 'convergence'}
            )
    
    def log_performance(self, metric: str, value: float):
        """
        Log performance metric.
        
        Args:
            metric: Metric name
            value: Metric value
        """
        self.metrics[metric] = value
        
        self.logger.info(
            f"Performance: {metric} = {value:.4f}",
            extra={
                'cfd_type': 'performance',
                'metric': metric,
                'value': value
            }
        )
    
    def end_simulation(self, success: bool = True):
        """
        Log simulation end.
        
        Args:
            success: Whether simulation completed successfully
        """
        if self.start_time:
            elapsed = time.time() - self.start_time
            hours = int(elapsed // 3600)
            minutes = int((elapsed % 3600) // 60)
            seconds = elapsed % 60
            
            status = "COMPLETED" if success else "FAILED"
            
            self.logger.info(
                f"Simulation {status}: {self.case_name}",
                extra={'elapsed_time': elapsed}
            )
            self.logger.info(
                f"Total time: {hours:02d}:{minutes:02d}:{seconds:05.2f}"
            )
            
            if self.metrics:
                self.logger.info("Final metrics:")
                for metric, value in self.metrics.items():
                    self.logger.info(f"  {metric}: {value:.4f}")


@contextmanager
def timed_operation(logger: logging.Logger, operation: str):
    """
    Context manager for timing operations.
    
    Usage:
        with timed_operation(logger, "mesh generation"):
            generate_mesh()
    """
    logger.info(f"Starting: {operation}")
    start = time.perf_counter()
    
    try:
        yield
        elapsed = time.perf_counter() - start
        logger.info(f"Completed: {operation} ({elapsed:.2f}s)")
    except Exception as e:
        elapsed = time.perf_counter() - start
        logger.error(f"Failed: {operation} ({elapsed:.2f}s): {e}")
        raise


def setup_simulation_logging(case_dir: Path, level: str = "INFO") -> SimulationLogger:
    """
    Set up logging for a simulation case.
    
    Args:
        case_dir: Case directory
        level: Logging level
        
    Returns:
        Configured SimulationLogger
    """
    log_dir = case_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure base logging
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s [%(levelname)s] %(name)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_dir / "simulation.log")
        ]
    )
    
    # Create simulation logger
    case_name = case_dir.name
    sim_logger = SimulationLogger(case_name, log_dir)
    
    return sim_logger