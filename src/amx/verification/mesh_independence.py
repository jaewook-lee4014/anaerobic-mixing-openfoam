"""Mesh independence study following ASME V&V 20-2009 standard."""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class MeshStudyResult:
    """Results from mesh independence study."""
    
    mesh_sizes: List[int]
    solutions: List[float]
    gci_fine: float  # Grid Convergence Index for fine mesh
    gci_medium: float  # GCI for medium mesh
    asymptotic_range: float
    richardson_extrapolation: float
    apparent_order: float
    converged: bool
    confidence_level: float


class MeshIndependenceStudy:
    """
    Mesh independence study following Richardson extrapolation method.
    
    References:
    - Roache, P.J. (1998). Verification and Validation in Computational Science and Engineering.
    - ASME V&V 20-2009. Standard for Verification and Validation in CFD and Heat Transfer.
    - Celik et al. (2008). Procedure for Estimation of Discretization Error in CFD. 
    """
    
    def __init__(self, safety_factor: float = 1.25):
        """
        Initialize mesh independence study.
        
        Args:
            safety_factor: Factor of safety for GCI (1.25 for 3+ grids, 3.0 for 2 grids)
        """
        self.Fs = safety_factor  # Factor of safety
        
    def richardson_extrapolation(self, 
                                h1: float, h2: float, h3: float,
                                f1: float, f2: float, f3: float) -> Dict:
        """
        Perform Richardson extrapolation for three mesh levels.
        
        Following ASME V&V 20-2009 procedure:
        1. Calculate refinement ratios
        2. Estimate apparent order of convergence
        3. Calculate extrapolated value
        4. Compute Grid Convergence Index (GCI)
        
        Args:
            h1, h2, h3: Representative mesh sizes (h1 < h2 < h3)
            f1, f2, f3: Solution values on respective meshes
            
        Returns:
            Dictionary with extrapolation results
        """
        # Check mesh ordering
        if not (h1 < h2 < h3):
            raise ValueError("Mesh sizes must be ordered: h1 < h2 < h3 (fine to coarse)")
        
        # Refinement ratios
        r21 = h2 / h1  # Should be > 1
        r32 = h3 / h2
        
        # Solution differences
        epsilon32 = f3 - f2
        epsilon21 = f2 - f1
        
        # Check for oscillatory convergence
        if epsilon32 * epsilon21 < 0:
            logger.warning("Oscillatory convergence detected")
            convergence_type = "oscillatory"
        else:
            convergence_type = "monotonic"
        
        # Apparent order of convergence (p)
        # Solve: epsilon32/epsilon21 = r21^p - 1 / r32^p - 1
        if abs(epsilon21) > 1e-10:
            # Iterative solution for p
            p = self._solve_for_order(r21, r32, epsilon21, epsilon32)
        else:
            logger.warning("epsilon21 too small, assuming theoretical order")
            p = 2.0  # Assume second-order
        
        # Richardson extrapolation to zero mesh size
        f_exact = f1 + epsilon21 / (r21**p - 1)
        
        # Relative errors
        e21_a = abs((f2 - f1) / f1) if f1 != 0 else 0
        e32_a = abs((f3 - f2) / f2) if f2 != 0 else 0
        
        # Grid Convergence Index (Roache, 1998)
        GCI_fine = self.Fs * e21_a / (r21**p - 1)
        GCI_medium = self.Fs * e32_a / (r32**p - 1)
        
        # Check asymptotic range
        # Should be approximately 1.0 for asymptotic convergence
        asymptotic_ratio = GCI_medium / (r21**p * GCI_fine)
        
        in_asymptotic = 0.95 < asymptotic_ratio < 1.05
        
        return {
            "apparent_order": p,
            "extrapolated_value": f_exact,
            "relative_error_21": e21_a * 100,  # Percent
            "relative_error_32": e32_a * 100,
            "GCI_fine": GCI_fine * 100,  # Percent
            "GCI_medium": GCI_medium * 100,
            "asymptotic_ratio": asymptotic_ratio,
            "in_asymptotic_range": in_asymptotic,
            "convergence_type": convergence_type,
            "refinement_ratio_21": r21,
            "refinement_ratio_32": r32
        }
    
    def _solve_for_order(self, r21: float, r32: float, 
                        eps21: float, eps32: float,
                        max_iter: int = 100) -> float:
        """
        Solve for apparent order p using fixed-point iteration.
        
        From: ln(eps32/eps21) = p * ln(r32) + ln((r21^p - 1)/(r32^p - 1))
        """
        # Initial guess based on theory
        p = 2.0
        
        # Avoid division by zero
        if abs(eps21) < 1e-10 or abs(eps32) < 1e-10:
            return p
        
        s = np.sign(eps32/eps21)
        ratio = abs(eps32/eps21)
        
        for i in range(max_iter):
            p_old = p
            
            # Fixed-point iteration
            f = (r21**p - s) / (r32**p - s) - ratio
            
            # Newton-Raphson update
            if abs(r32**p - s) > 1e-10:
                df_dp = (r21**p * np.log(r21) * (r32**p - s) - 
                        r32**p * np.log(r32) * (r21**p - s)) / (r32**p - s)**2
                
                if abs(df_dp) > 1e-10:
                    p = p_old - f / df_dp
                else:
                    break
            
            # Bound p to reasonable values
            p = max(0.5, min(p, 4.0))
            
            # Check convergence
            if abs(p - p_old) < 1e-6:
                break
        
        return p
    
    def run_study(self, mesh_cases: List[Dict]) -> MeshStudyResult:
        """
        Run complete mesh independence study.
        
        Args:
            mesh_cases: List of dictionaries with 'cells' and 'result' keys
            
        Returns:
            MeshStudyResult with complete analysis
        """
        if len(mesh_cases) < 3:
            raise ValueError("At least 3 mesh levels required for study")
        
        # Sort by mesh size (fine to coarse)
        cases = sorted(mesh_cases, key=lambda x: x['cells'], reverse=True)
        
        # Extract data
        cells = [c['cells'] for c in cases[:3]]
        results = [c['result'] for c in cases[:3]]
        
        # Representative mesh size h ~ N^(-1/3) for 3D
        h = [float(cells[0]/c)**(1/3) for c in cells[:3]]
        
        # Richardson extrapolation
        extrap = self.richardson_extrapolation(
            h[2], h[1], h[0],  # h1, h2, h3 (fine to coarse)
            results[2], results[1], results[0]  # f1, f2, f3
        )
        
        # Determine convergence
        converged = (
            extrap['in_asymptotic_range'] and
            extrap['GCI_fine'] < 5.0 and  # < 5% error
            extrap['convergence_type'] == 'monotonic'
        )
        
        # Confidence level based on GCI
        if extrap['GCI_fine'] < 1:
            confidence = 99
        elif extrap['GCI_fine'] < 3:
            confidence = 95
        elif extrap['GCI_fine'] < 5:
            confidence = 90
        else:
            confidence = 100 - extrap['GCI_fine']
        
        return MeshStudyResult(
            mesh_sizes=cells[:3],
            solutions=results[:3],
            gci_fine=extrap['GCI_fine'],
            gci_medium=extrap['GCI_medium'],
            asymptotic_range=extrap['asymptotic_ratio'],
            richardson_extrapolation=extrap['extrapolated_value'],
            apparent_order=extrap['apparent_order'],
            converged=converged,
            confidence_level=confidence
        )
    
    def recommend_mesh_size(self, target_accuracy: float = 3.0) -> int:
        """
        Recommend mesh size for target accuracy.
        
        Based on GCI < target_accuracy criterion.
        
        Args:
            target_accuracy: Target GCI in percent
            
        Returns:
            Recommended number of cells
        """
        # Based on typical convergence rates
        # N ~ (1/error)^(3/p) where p is order of accuracy
        
        # Assuming second-order method (p=2)
        # and current GCI of 5% at 320,000 cells
        current_cells = 320000
        current_gci = 5.0
        
        # Scale cells to achieve target
        required_cells = current_cells * (current_gci / target_accuracy)**(3/2)
        
        # Round to nearest practical value
        practical_cells = [
            50000, 100000, 200000, 320000, 500000,
            1000000, 2000000, 5000000, 10000000
        ]
        
        for cells in practical_cells:
            if cells >= required_cells:
                return cells
        
        return int(required_cells)


def generate_mesh_cases(base_config: Dict) -> List[Dict]:
    """
    Generate mesh cases for independence study.
    
    Following best practices:
    - Refinement ratio ~ 1.3-2.0
    - At least 3 levels (coarse, medium, fine)
    - Systematic refinement in all directions
    
    Args:
        base_config: Base configuration dictionary
        
    Returns:
        List of mesh configurations
    """
    # Base mesh (medium)
    base_cells = base_config.get('cells', [100, 40, 80])
    
    # Refinement ratio
    r = 1.5  # Recommended: 1.3-2.0
    
    mesh_cases = [
        {
            "name": "coarse",
            "cells": [int(n/r) for n in base_cells],
            "total_cells": int(np.prod([n/r for n in base_cells])),
            "level": 0
        },
        {
            "name": "medium", 
            "cells": base_cells,
            "total_cells": int(np.prod(base_cells)),
            "level": 1
        },
        {
            "name": "fine",
            "cells": [int(n*r) for n in base_cells],
            "total_cells": int(np.prod([n*r for n in base_cells])),
            "level": 2
        },
        {
            "name": "very_fine",
            "cells": [int(n*r*r) for n in base_cells],
            "total_cells": int(np.prod([n*r*r for n in base_cells])),
            "level": 3
        }
    ]
    
    return mesh_cases