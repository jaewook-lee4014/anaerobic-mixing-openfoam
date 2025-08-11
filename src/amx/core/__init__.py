"""Core base classes and interfaces for the AMX framework."""

from .base import (
    BaseModel,
    BaseSimulation,
    BaseAnalyzer,
    SimulationResult,
)
from .interfaces import (
    FluidModel,
    TurbulenceModel,
    RheologyModel,
    MixingModel,
)
from .exceptions import (
    AMXError,
    ConfigurationError,
    SimulationError,
    AnalysisError,
)

__all__ = [
    # Base classes
    "BaseModel",
    "BaseSimulation",
    "BaseAnalyzer",
    "SimulationResult",
    # Interfaces
    "FluidModel",
    "TurbulenceModel",
    "RheologyModel",
    "MixingModel",
    # Exceptions
    "AMXError",
    "ConfigurationError",
    "SimulationError",
    "AnalysisError",
]