"""Custom exceptions for the AMX framework."""


class AMXError(Exception):
    """Base exception for AMX framework."""
    
    def __init__(self, message: str, details: dict = None):
        """
        Initialize AMX error.
        
        Args:
            message: Error message
            details: Additional error details
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}
    
    def __str__(self) -> str:
        """String representation."""
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} ({details_str})"
        return self.message


class ConfigurationError(AMXError):
    """Configuration-related errors."""
    pass


class SimulationError(AMXError):
    """Simulation execution errors."""
    pass


class AnalysisError(AMXError):
    """Analysis and post-processing errors."""
    pass


class ValidationError(AMXError):
    """Data validation errors."""
    pass


class ConvergenceError(SimulationError):
    """Convergence-related errors."""
    pass


class MeshError(SimulationError):
    """Mesh generation errors."""
    pass


class IOError(AMXError):
    """Input/output errors."""
    pass