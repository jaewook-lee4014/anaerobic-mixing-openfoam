"""Uncertainty quantification following ASME V&V standards."""

import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass 
class UncertaintyResult:
    """Complete uncertainty quantification results."""
    
    # Input uncertainties
    input_uncertainties: Dict[str, float]
    
    # Numerical uncertainties
    discretization_error: float  # From GCI
    iterative_error: float  # Convergence
    round_off_error: float  # Machine precision
    
    # Model uncertainties  
    turbulence_model_error: float
    boundary_condition_error: float
    
    # Validation uncertainties
    experimental_uncertainty: float
    
    # Combined uncertainties
    numerical_uncertainty: float
    model_uncertainty: float
    total_uncertainty: float
    
    # Confidence intervals
    confidence_level: float
    lower_bound: float
    upper_bound: float


class UncertaintyQuantification:
    """
    Comprehensive uncertainty quantification for CFD.
    
    References:
    - ASME V&V 20-2009: Standard for Verification and Validation
    - Coleman & Steele (2009): Experimentation, Validation, and Uncertainty
    - Oberkampf & Roy (2010): Verification and Validation in Scientific Computing
    """
    
    def __init__(self):
        """Initialize uncertainty quantification."""
        self.confidence_level = 0.95  # 95% confidence
        
    def propagate_uncertainty(self, 
                            inputs: Dict[str, Tuple[float, float]],
                            sensitivity: Dict[str, float]) -> float:
        """
        Propagate input uncertainties using Taylor series method.
        
        U_R = sqrt(Σ (∂R/∂x_i * U_xi)²)
        
        Args:
            inputs: Dict of (value, uncertainty) for each input
            sensitivity: Sensitivity coefficients ∂R/∂x_i
            
        Returns:
            Propagated uncertainty
        """
        variance = 0.0
        
        for param, (value, uncertainty) in inputs.items():
            if param in sensitivity:
                # Contribution to variance
                variance += (sensitivity[param] * uncertainty)**2
        
        return np.sqrt(variance)
    
    def sensitivity_analysis(self, 
                           base_case: Dict,
                           perturbations: Dict[str, float],
                           model_func) -> Dict[str, float]:
        """
        Perform sensitivity analysis using finite differences.
        
        Args:
            base_case: Base parameter values
            perturbations: Relative perturbation for each parameter
            model_func: Function that runs model with given parameters
            
        Returns:
            Sensitivity coefficients
        """
        base_result = model_func(base_case)
        sensitivities = {}
        
        for param, perturb in perturbations.items():
            if param in base_case:
                # Perturbed case
                perturbed = base_case.copy()
                perturbed[param] *= (1 + perturb)
                
                # Run perturbed case
                perturbed_result = model_func(perturbed)
                
                # Sensitivity coefficient
                delta_param = base_case[param] * perturb
                delta_result = perturbed_result - base_result
                
                if delta_param != 0:
                    sensitivities[param] = delta_result / delta_param
                else:
                    sensitivities[param] = 0.0
        
        return sensitivities
    
    def numerical_uncertainty(self,
                            gci: float,
                            iterative_error: float = 1e-5,
                            round_off: float = 1e-10) -> float:
        """
        Calculate combined numerical uncertainty.
        
        Following ASME V&V 20-2009:
        U_num = sqrt(U_disc² + U_iter² + U_round²)
        
        Args:
            gci: Grid Convergence Index (from mesh study)
            iterative_error: Iterative convergence error
            round_off: Round-off error estimate
            
        Returns:
            Combined numerical uncertainty
        """
        # Convert GCI percentage to fraction
        disc_error = gci / 100.0
        
        # Combine using root-sum-square
        u_num = np.sqrt(disc_error**2 + iterative_error**2 + round_off**2)
        
        return u_num
    
    def model_form_uncertainty(self,
                             model_type: str = "RANS_k_epsilon") -> float:
        """
        Estimate model form uncertainty based on literature.
        
        Args:
            model_type: Type of turbulence model
            
        Returns:
            Model uncertainty estimate
        """
        # Based on literature assessments
        model_uncertainties = {
            "RANS_k_epsilon": 0.15,  # 15% for standard k-ε
            "RANS_realizable": 0.12,  # 12% for realizable k-ε
            "RANS_k_omega_SST": 0.10,  # 10% for SST
            "LES": 0.05,  # 5% for LES
            "DNS": 0.01,  # 1% for DNS
            "mixing_length": 0.25,  # 25% for simple models
        }
        
        return model_uncertainties.get(model_type, 0.20)
    
    def validation_uncertainty(self,
                             simulation: float,
                             experiment: float,
                             exp_uncertainty: float) -> Dict:
        """
        Calculate validation uncertainty and metrics.
        
        Following ASME V&V 20-2009 validation approach.
        
        Args:
            simulation: Simulation result
            experiment: Experimental result
            exp_uncertainty: Experimental uncertainty
            
        Returns:
            Validation metrics
        """
        # Comparison error
        E = abs(simulation - experiment)
        
        # Relative error
        relative_error = E / experiment if experiment != 0 else 0
        
        # Validation uncertainty (assuming 95% confidence)
        u_val = 2 * exp_uncertainty  # k=2 for 95% confidence
        
        # Validation metric
        if u_val > 0:
            validation_metric = E / u_val
        else:
            validation_metric = float('inf')
        
        # Assessment
        if validation_metric < 1:
            assessment = "Validated"
        elif validation_metric < 2:
            assessment = "Marginally validated"
        else:
            assessment = "Not validated"
        
        return {
            "comparison_error": E,
            "relative_error": relative_error * 100,  # Percent
            "validation_uncertainty": u_val,
            "validation_metric": validation_metric,
            "assessment": assessment
        }
    
    def monte_carlo_uncertainty(self,
                              inputs: Dict[str, Dict],
                              model_func,
                              n_samples: int = 1000) -> Dict:
        """
        Monte Carlo uncertainty propagation.
        
        Args:
            inputs: Dict with 'mean', 'std', 'distribution' for each input
            model_func: Function to evaluate
            n_samples: Number of Monte Carlo samples
            
        Returns:
            Statistics from Monte Carlo analysis
        """
        results = []
        
        for i in range(n_samples):
            # Sample inputs
            sample = {}
            for param, specs in inputs.items():
                if specs['distribution'] == 'normal':
                    sample[param] = np.random.normal(
                        specs['mean'], specs['std']
                    )
                elif specs['distribution'] == 'uniform':
                    sample[param] = np.random.uniform(
                        specs['min'], specs['max']
                    )
                else:
                    sample[param] = specs['mean']
            
            # Run model
            result = model_func(sample)
            results.append(result)
        
        results = np.array(results)
        
        # Statistics
        return {
            "mean": np.mean(results),
            "std": np.std(results),
            "min": np.min(results),
            "max": np.max(results),
            "percentile_5": np.percentile(results, 5),
            "percentile_95": np.percentile(results, 95),
            "coefficient_of_variation": np.std(results) / np.mean(results)
        }
    
    def combine_uncertainties(self,
                            numerical: float,
                            model: float,
                            experimental: float = 0) -> float:
        """
        Combine different uncertainty sources.
        
        Using root-sum-square method:
        U_total = sqrt(U_num² + U_model² + U_exp²)
        
        Args:
            numerical: Numerical uncertainty
            model: Model form uncertainty
            experimental: Experimental uncertainty
            
        Returns:
            Combined uncertainty
        """
        return np.sqrt(numerical**2 + model**2 + experimental**2)
    
    def confidence_interval(self,
                          mean: float,
                          uncertainty: float,
                          confidence: float = 0.95) -> Tuple[float, float]:
        """
        Calculate confidence interval.
        
        Args:
            mean: Mean value
            uncertainty: Standard uncertainty
            confidence: Confidence level (0-1)
            
        Returns:
            (lower_bound, upper_bound)
        """
        # Coverage factor for normal distribution
        k = stats.norm.ppf((1 + confidence) / 2)
        
        expanded_uncertainty = k * uncertainty
        
        return (mean - expanded_uncertainty, mean + expanded_uncertainty)
    
    def report_uncertainty(self,
                         value: float,
                         numerical_unc: float,
                         model_unc: float,
                         exp_unc: float = 0) -> UncertaintyResult:
        """
        Generate complete uncertainty report.
        
        Args:
            value: Computed value
            numerical_unc: Numerical uncertainty
            model_unc: Model uncertainty
            exp_unc: Experimental uncertainty
            
        Returns:
            Complete uncertainty analysis
        """
        # Relative uncertainties
        rel_numerical = numerical_unc / value if value != 0 else 0
        rel_model = model_unc / value if value != 0 else 0
        rel_exp = exp_unc / value if value != 0 else 0
        
        # Combined uncertainty
        total_unc = self.combine_uncertainties(
            numerical_unc, model_unc, exp_unc
        )
        rel_total = total_unc / value if value != 0 else 0
        
        # Confidence interval
        lower, upper = self.confidence_interval(value, total_unc)
        
        return UncertaintyResult(
            input_uncertainties={},  # Would be filled from sensitivity analysis
            discretization_error=numerical_unc,
            iterative_error=1e-5,  # Typical value
            round_off_error=1e-10,  # Machine precision
            turbulence_model_error=model_unc,
            boundary_condition_error=0.05,  # Typical estimate
            experimental_uncertainty=exp_unc,
            numerical_uncertainty=numerical_unc,
            model_uncertainty=model_unc,
            total_uncertainty=total_unc,
            confidence_level=self.confidence_level,
            lower_bound=lower,
            upper_bound=upper
        )


def format_uncertainty(value: float, uncertainty: float, 
                      confidence: float = 0.95) -> str:
    """
    Format value with uncertainty following standards.
    
    Args:
        value: Central value
        uncertainty: Standard uncertainty
        confidence: Confidence level
        
    Returns:
        Formatted string
    """
    # Determine significant figures
    if uncertainty > 0:
        # Round uncertainty to 2 significant figures
        exp = np.floor(np.log10(uncertainty))
        unc_rounded = np.round(uncertainty, -int(exp) + 1)
        
        # Round value to same decimal place
        val_rounded = np.round(value, -int(exp) + 1)
        
        # Format based on magnitude
        if abs(exp) < 3:
            return f"{val_rounded:.{max(0, -int(exp)+1)}f} ± {unc_rounded:.{max(0, -int(exp)+1)}f} ({confidence*100:.0f}% conf.)"
        else:
            return f"{val_rounded:.2e} ± {unc_rounded:.2e} ({confidence*100:.0f}% conf.)"
    else:
        return f"{value:.4f} (exact)"