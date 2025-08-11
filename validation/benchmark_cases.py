"""Benchmark cases for validation against literature data."""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class BenchmarkCase:
    """Benchmark case with experimental or literature data."""
    
    name: str
    reference: str
    description: str
    parameters: Dict
    results: Dict
    uncertainty: Dict


# Classic benchmark cases from literature
BENCHMARK_CASES = [
    BenchmarkCase(
        name="Rushton_Turbine_Standard",
        reference="Rushton et al. (1950) Chemical Engineering Progress",
        description="Standard 6-blade Rushton turbine in baffled tank",
        parameters={
            "tank_diameter": 1.0,  # m
            "impeller_diameter": 0.333,  # D/T = 1/3
            "blade_width": 0.067,  # W/D = 1/5
            "blade_height": 0.083,  # H/D = 1/4
            "rotation_speed": 200,  # rpm
            "fluid_density": 998,  # kg/m³
            "fluid_viscosity": 0.001,  # Pa·s
            "reynolds_number": 35000
        },
        results={
            "power_number": 5.0,
            "flow_number": 0.72,
            "mixing_time_dimensionless": 5.3,  # Nt_m
            "mean_velocity_ratio": 0.75,  # U_mean/U_tip
        },
        uncertainty={
            "power_number": 0.2,
            "flow_number": 0.05,
            "mixing_time": 0.5
        }
    ),
    
    BenchmarkCase(
        name="Jet_Mixing_Simon_2011",
        reference="Simon et al. (2011) Chemical Engineering Science",
        description="Multiple jet mixing in rectangular tank",
        parameters={
            "tank_length": 4.0,  # m
            "tank_width": 2.0,  # m
            "tank_height": 3.0,  # m
            "number_of_jets": 4,
            "jet_diameter": 0.05,  # m
            "jet_velocity": 5.0,  # m/s
            "jet_angle": 45,  # degrees
            "fluid_properties": "water_20C"
        },
        results={
            "mixing_time_95": 180,  # seconds
            "mean_velocity": 0.25,  # m/s
            "dead_zone_fraction": 0.08,
            "energy_dissipation": 15.2,  # W/m³
        },
        uncertainty={
            "mixing_time": 15,
            "mean_velocity": 0.03,
            "dead_zone": 0.02
        }
    ),
    
    BenchmarkCase(
        name="Anaerobic_Digester_EPA",
        reference="EPA Process Design Manual (1979)",
        description="Full-scale anaerobic digester with draft tube mixing",
        parameters={
            "tank_volume": 3000,  # m³
            "tank_diameter": 18,  # m
            "sidewall_depth": 12,  # m
            "mixing_type": "gas_recirculation",
            "gas_flow_rate": 0.005,  # m³/m³/min
            "temperature": 35,  # °C
            "solids_content": 4.0,  # %
        },
        results={
            "velocity_gradient": 50,  # s⁻¹
            "mixing_energy": 5.0,  # W/m³
            "turnover_time": 20,  # minutes
            "active_volume_fraction": 0.85,
        },
        uncertainty={
            "velocity_gradient": 10,
            "mixing_energy": 1.0,
            "turnover_time": 5
        }
    ),
    
    BenchmarkCase(
        name="CFD_Validation_Wu_2010",
        reference="Wu (2010) Bioresource Technology",
        description="CFD validation of mechanical mixing in digester",
        parameters={
            "tank_diameter": 10,  # m
            "liquid_depth": 8,  # m
            "impeller_type": "pitched_blade",
            "impeller_diameter": 2.5,  # m
            "rotation_speed": 30,  # rpm
            "number_of_impellers": 2,
            "fluid_viscosity": 0.03,  # Pa·s (sludge)
        },
        results={
            "power_consumption": 12.5,  # kW
            "mean_velocity": 0.18,  # m/s
            "dead_zones": 0.12,  # fraction
            "mixing_time": 25,  # minutes
        },
        uncertainty={
            "power": 1.5,
            "velocity": 0.02,
            "dead_zones": 0.03
        }
    ),
    
    BenchmarkCase(
        name="Jet_Array_Fossett_1949", 
        reference="Fossett & Prosser (1949) Trans. Inst. Chem. Eng.",
        description="Side-entering jets in rectangular tank",
        parameters={
            "tank_dimensions": [6, 3, 4],  # L, W, H in meters
            "jet_configuration": "opposed_wall_jets",
            "number_of_jets": 8,
            "jet_spacing": 1.5,  # m
            "jet_reynolds": 50000,
            "momentum_flux_ratio": 0.1,
        },
        results={
            "blend_time_correlation": "theta = 4.0 * V/Q",
            "circulation_pattern": "dual_loop",
            "velocity_decay_constant": 6.2,
            "entrainment_coefficient": 0.057,
        },
        uncertainty={
            "blend_time": 0.5,
            "velocity_decay": 0.3,
            "entrainment": 0.005
        }
    )
]


def get_validation_metrics(case_name: str, simulation_results: Dict) -> Dict:
    """
    Compare simulation results with benchmark data.
    
    Args:
        case_name: Name of benchmark case
        simulation_results: Dictionary of simulation results
        
    Returns:
        Validation metrics including errors and confidence
    """
    # Find benchmark case
    benchmark = None
    for case in BENCHMARK_CASES:
        if case.name == case_name:
            benchmark = case
            break
    
    if not benchmark:
        raise ValueError(f"Benchmark case '{case_name}' not found")
    
    validation = {}
    
    # Calculate relative errors
    for key, expected in benchmark.results.items():
        if key in simulation_results:
            actual = simulation_results[key]
            if isinstance(expected, (int, float)):
                error = abs(actual - expected) / expected * 100
                uncertainty = benchmark.uncertainty.get(key, 0)
                
                # Check if within uncertainty bounds
                within_bounds = error <= (uncertainty / expected * 100)
                
                validation[key] = {
                    "expected": expected,
                    "actual": actual,
                    "error_percent": error,
                    "uncertainty": uncertainty,
                    "within_bounds": within_bounds,
                    "confidence": max(0, 100 - error) if within_bounds else 0
                }
    
    # Overall validation score
    scores = [v["confidence"] for v in validation.values() if "confidence" in v]
    validation["overall_score"] = np.mean(scores) if scores else 0
    
    return validation


def get_recommended_cases(application: str) -> List[str]:
    """
    Get recommended benchmark cases for specific application.
    
    Args:
        application: Type of application (e.g., "anaerobic_digester", "jet_mixing")
        
    Returns:
        List of recommended benchmark case names
    """
    recommendations = {
        "anaerobic_digester": [
            "Anaerobic_Digester_EPA",
            "CFD_Validation_Wu_2010"
        ],
        "jet_mixing": [
            "Jet_Mixing_Simon_2011",
            "Jet_Array_Fossett_1949"
        ],
        "mechanical_mixing": [
            "Rushton_Turbine_Standard",
            "CFD_Validation_Wu_2010"
        ]
    }
    
    return recommendations.get(application, [])


def literature_correlations():
    """
    Collection of validated correlations from literature.
    """
    return {
        "mixing_time": {
            "Grenville_1996": {
                "equation": "theta_m = 5.4 * (T^3/Q)",
                "application": "jet_mixing",
                "range": "Re > 10000",
                "reference": "Grenville & Tilton (1996) AIChE J."
            },
            "Ruszkowski_1994": {
                "equation": "Nt_m = 5.3 * (T/D)^2.3 * (mu/rho)^0.1",
                "application": "mechanical_mixing",
                "range": "0.3 < D/T < 0.6",
                "reference": "Ruszkowski (1994) Chem. Eng. Sci."
            },
            "Norwood_1960": {
                "equation": "theta_m = C * (H/T)^0.5 * N^(-1)",
                "application": "unbaffled_tanks",
                "constant_C": 36,
                "reference": "Norwood & Metzner (1960) AIChE J."
            }
        },
        "power_number": {
            "Rushton_turbine": {
                "laminar": "Po = 64/Re",
                "transition": "Po = 10/Re^0.3",
                "turbulent": "Po = 5.0",
                "Re_transition": [10, 10000],
                "reference": "Rushton et al. (1950)"
            },
            "pitched_blade": {
                "turbulent_Po": 1.27,
                "pumping_number": 0.79,
                "reference": "Oldshue (1983)"
            }
        },
        "jet_decay": {
            "round_jet": {
                "velocity": "U/U_0 = 6.0 * (D/x)",
                "width": "b/x = 0.11",
                "reference": "Pope (2000) Turbulent Flows"
            },
            "plane_jet": {
                "velocity": "U/U_0 = 2.4 * sqrt(D/x)",
                "width": "b/x = 0.10",
                "reference": "Rajaratnam (1976)"
            }
        }
    }