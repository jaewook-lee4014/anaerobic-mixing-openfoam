"""
Anaerobic Mixing with OpenFOAM
A comprehensive CFD simulation framework for anaerobic digester mixing.
"""

__version__ = "0.1.0"
__author__ = "Engineering Team"

from amx.config import Config, load_config
from amx.geometry import TankGeometry, NozzleArray

__all__ = [
    "Config",
    "load_config",
    "TankGeometry",
    "NozzleArray",
]