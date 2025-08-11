"""Base classes for the AMX framework."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from pydantic import BaseModel as PydanticModel


class BaseModel(PydanticModel):
    """Enhanced Pydantic base model with common functionality."""
    
    class Config:
        """Pydantic configuration."""
        arbitrary_types_allowed = True
        use_enum_values = True
        validate_assignment = True
        extra = "forbid"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary."""
        return self.model_dump()
    
    def to_json(self) -> str:
        """Convert model to JSON string."""
        return self.model_dump_json(indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """Create model from dictionary."""
        return cls(**data)


@dataclass
class SimulationResult:
    """Container for simulation results."""
    
    success: bool
    case_dir: Path
    end_time: float
    fields: Dict[str, np.ndarray] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def has_errors(self) -> bool:
        """Check if simulation has errors."""
        return len(self.errors) > 0
    
    @property
    def has_warnings(self) -> bool:
        """Check if simulation has warnings."""
        return len(self.warnings) > 0
    
    def add_field(self, name: str, data: np.ndarray) -> None:
        """Add field data to results."""
        self.fields[name] = data
    
    def add_metric(self, name: str, value: float) -> None:
        """Add metric to results."""
        self.metrics[name] = value
    
    def add_error(self, message: str) -> None:
        """Add error message."""
        self.errors.append(message)
        self.success = False
    
    def add_warning(self, message: str) -> None:
        """Add warning message."""
        self.warnings.append(message)
    
    def summary(self) -> Dict[str, Any]:
        """Get summary of results."""
        return {
            "success": self.success,
            "case_dir": str(self.case_dir),
            "end_time": self.end_time,
            "n_fields": len(self.fields),
            "n_metrics": len(self.metrics),
            "n_errors": len(self.errors),
            "n_warnings": len(self.warnings),
            "timestamp": self.timestamp.isoformat(),
        }


class BaseSimulation(ABC):
    """Abstract base class for simulations."""
    
    def __init__(self, config: BaseModel, output_dir: Path):
        """
        Initialize simulation.
        
        Args:
            config: Simulation configuration
            output_dir: Output directory
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._result = None
    
    @abstractmethod
    def setup(self) -> None:
        """Set up simulation case."""
        pass
    
    @abstractmethod
    def run(self) -> SimulationResult:
        """Run simulation."""
        pass
    
    @abstractmethod
    def post_process(self) -> None:
        """Post-process simulation results."""
        pass
    
    def validate(self) -> bool:
        """
        Validate simulation setup.
        
        Returns:
            True if valid
        """
        # Default implementation - override in subclasses
        return True
    
    def clean(self) -> None:
        """Clean up simulation artifacts."""
        # Default implementation - override if needed
        pass
    
    @property
    def result(self) -> Optional[SimulationResult]:
        """Get simulation result."""
        return self._result
    
    def execute(self) -> SimulationResult:
        """
        Execute complete simulation workflow.
        
        Returns:
            Simulation result
        """
        try:
            # Validate
            if not self.validate():
                result = SimulationResult(
                    success=False,
                    case_dir=self.output_dir,
                    end_time=0.0
                )
                result.add_error("Validation failed")
                return result
            
            # Setup
            self.setup()
            
            # Run
            result = self.run()
            
            # Post-process if successful
            if result.success:
                self.post_process()
            
            self._result = result
            return result
            
        except Exception as e:
            result = SimulationResult(
                success=False,
                case_dir=self.output_dir,
                end_time=0.0
            )
            result.add_error(f"Simulation failed: {str(e)}")
            self._result = result
            return result
        
        finally:
            self.clean()


class BaseAnalyzer(ABC):
    """Abstract base class for analysis modules."""
    
    def __init__(self, data: Dict[str, Any]):
        """
        Initialize analyzer.
        
        Args:
            data: Input data for analysis
        """
        self.data = data
        self._results = {}
    
    @abstractmethod
    def analyze(self) -> Dict[str, Any]:
        """
        Perform analysis.
        
        Returns:
            Analysis results
        """
        pass
    
    @abstractmethod
    def validate_data(self) -> bool:
        """
        Validate input data.
        
        Returns:
            True if data is valid
        """
        pass
    
    def get_results(self) -> Dict[str, Any]:
        """Get analysis results."""
        if not self._results:
            self._results = self.analyze()
        return self._results
    
    def export_results(self, output_path: Path, format: str = "json") -> None:
        """
        Export results to file.
        
        Args:
            output_path: Output file path
            format: Export format (json, csv, etc.)
        """
        import json
        
        results = self.get_results()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "json":
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2, default=str)
        else:
            raise NotImplementedError(f"Export format {format} not supported")