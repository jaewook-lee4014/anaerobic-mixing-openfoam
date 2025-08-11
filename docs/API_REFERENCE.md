# API Reference

## Core Modules

### amx.physics

Academic physics models based on peer-reviewed literature.

#### JetModel
```python
from amx.physics import JetModel

jet = JetModel(
    diameter=0.035,      # Nozzle diameter [m]
    velocity=3.9,        # Exit velocity [m/s]
    angle=0.785,         # Discharge angle [rad]
    density=998,         # Fluid density [kg/m³]
    viscosity=0.00074    # Dynamic viscosity [Pa·s]
)

# Calculate jet properties
u_centerline = jet.centerline_velocity(x=1.0)  # Velocity at 1m
width = jet.jet_width(x=1.0)                    # Jet width at 1m
Re = jet.reynolds                               # Reynolds number
```

**References**:
- Rajaratnam, N. (1976). Turbulent Jets. Elsevier.
- Pope, S.B. (2000). Turbulent Flows. Cambridge University Press.

#### RANS_kEpsilon
```python
from amx.physics import RANS_kEpsilon

turb = RANS_kEpsilon(nu=7.4e-7, rho=998)

# Calculate turbulence properties
nu_t = turb.calculate_eddy_viscosity(k=0.01, epsilon=0.001)
l_scale = turb.calculate_length_scale(k=0.01, epsilon=0.001)

# Initial estimates
init = turb.estimate_initial_k_epsilon(
    U_ref=0.3,                  # Reference velocity
    L_ref=10,                   # Reference length
    turbulence_intensity=0.05   # 5% turbulence
)
```

**Model Constants** (Launder & Spalding, 1974):
- C_μ = 0.09
- C_1ε = 1.44
- C_2ε = 1.92
- σ_k = 1.0
- σ_ε = 1.3

#### MixingTheory
```python
from amx.physics import MixingTheory, CampNumber

mixing = MixingTheory(volume=2560, viscosity=0.00074, density=998)

# Calculate G-value
G = mixing.velocity_gradient_from_power(power=30000)

# Mixing time correlations
t_mix = mixing.mixing_time_correlation(flow_rate=0.119, method="grenville")

# Kolmogorov scales
scales = mixing.kolmogorov_scales(epsilon=0.01)
print(f"η = {scales['length']*1000:.3f} mm")

# Camp number
camp = CampNumber(G_value=119.4, time=120)
print(f"Gt = {camp.camp_number}")
```

### amx.verification

Verification and validation following ASME V&V 20-2009.

#### MeshIndependenceStudy
```python
from amx.verification import MeshIndependenceStudy

study = MeshIndependenceStudy(safety_factor=1.25)

# Run Richardson extrapolation
result = study.richardson_extrapolation(
    h1=0.1, h2=0.15, h3=0.225,  # Mesh sizes (fine to coarse)
    f1=0.34, f2=0.32, f3=0.30    # Solutions
)

print(f"Apparent order: {result['apparent_order']:.2f}")
print(f"GCI fine: {result['GCI_fine']:.1f}%")
print(f"Extrapolated value: {result['extrapolated_value']:.3f}")
```

### amx.uncertainty

Uncertainty quantification following ASME standards.

#### UncertaintyQuantification
```python
from amx.uncertainty import UncertaintyQuantification

uq = UncertaintyQuantification()

# Propagate uncertainties
inputs = {
    'velocity': (3.9, 0.1),   # (value, uncertainty)
    'diameter': (0.035, 0.001)
}
sensitivity = {
    'velocity': 0.087,  # ∂R/∂v
    'diameter': -2.5    # ∂R/∂d
}
u_total = uq.propagate_uncertainty(inputs, sensitivity)

# Monte Carlo analysis
mc_result = uq.monte_carlo_uncertainty(
    inputs={
        'velocity': {'mean': 3.9, 'std': 0.1, 'distribution': 'normal'},
        'diameter': {'mean': 0.035, 'std': 0.001, 'distribution': 'normal'}
    },
    model_func=lambda x: x['velocity'] * x['diameter'],
    n_samples=10000
)
```

### amx.validation

Benchmark cases from literature.

#### Benchmark Cases
```python
from amx.validation import get_validation_metrics, BENCHMARK_CASES

# Compare with benchmark
simulation_results = {
    'mean_velocity': 0.34,
    'dead_zones': 0.002,
    'mixing_time': 132
}

validation = get_validation_metrics(
    case_name="Jet_Mixing_Simon_2011",
    simulation_results=simulation_results
)

print(f"Overall validation score: {validation['overall_score']:.1f}%")
```

Available benchmark cases:
- `Rushton_Turbine_Standard` - Classical impeller mixing
- `Jet_Mixing_Simon_2011` - Multiple jet mixing
- `Anaerobic_Digester_EPA` - EPA design manual data
- `CFD_Validation_Wu_2010` - CFD validation data
- `Jet_Array_Fossett_1949` - Wall jet experiments

## Configuration

### YAML Configuration Structure
```yaml
# Complete configuration example
project: "digester_prod"

geometry:
  tank:
    L: 20.0      # Length [m]
    W: 8.0       # Width [m]
    H: 16.0      # Height [m]
  
  nozzle:
    count: 32
    throat_diameter_mm: 35.0
    pitch_deg: 45.0
    array: [4, 8]  # 4 rows × 8 columns

mesh:
  cells: [100, 40, 80]  # 320,000 cells
  y_plus_target: 30     # Wall function compatible

fluid:
  T: 308.15      # Temperature [K]
  rho: 998.0     # Density [kg/m³]
  mu: 0.00074    # Viscosity [Pa·s]
  nu: 7.4e-7     # Kinematic viscosity [m²/s]

operation:
  pump_total_m3ph: 430.0
  head_m: 15.0
  schedule:
    runs_per_day: 3
    hours_per_run: 2

solver:
  name: "pimpleFoam"
  dt: 0.05
  endTime: 1800
  maxCo: 2.0

energy:
  pump_efficiency: 0.65
  motor_efficiency: 0.90
```

## CLI Commands

### Basic Usage
```bash
# Check installation
amx --version

# Run simulation
amx run-case --config configs/case_prod.yaml --output runs/prod

# Analyze results  
amx analyze-mix --input runs/prod --output analysis/

# Generate report
amx build-report --config configs/case_prod.yaml \
                --metrics analysis/metrics.json \
                --output report.md

# Mesh independence study
amx mesh-study --base-config configs/case_prod.yaml \
              --levels 3 \
              --output mesh_study/
```

### Docker Execution
```bash
# Check OpenFOAM environment
bash scripts/check_openfoam.sh

# Run with Docker
bash scripts/run_with_docker.sh case_prod configs/case_prod.yaml
```

## Error Codes and Troubleshooting

### Common Errors

| Code | Description | Solution |
|------|-------------|----------|
| E001 | OpenFOAM not found | Install OpenFOAM or use Docker |
| E002 | Mesh generation failed | Check blockMeshDict syntax |
| E003 | Solver divergence | Reduce time step or relaxation factors |
| E004 | Memory insufficient | Reduce mesh size or use parallel |
| E005 | Invalid configuration | Check YAML syntax and required fields |

### Logging

Enhanced logging with structured output:

```python
from amx.utils.logging_enhanced import SimulationLogger, timed_operation

# Create logger
logger = SimulationLogger("my_case", log_dir=Path("logs/"))

# Log simulation progress
logger.start_simulation()
logger.log_iteration(100, {'U': 1e-4, 'p': 1e-5}, continuity=1e-6)
logger.log_performance("mean_velocity", 0.34)
logger.end_simulation(success=True)

# Time operations
with timed_operation(logger.logger, "mesh generation"):
    generate_mesh()
```

## Performance Guidelines

### Mesh Sizing
- Minimum: 100,000 cells (quick tests)
- Recommended: 300,000 - 1,000,000 cells
- Production: 2,000,000+ cells

### Convergence Criteria
- Residuals < 1e-5
- Continuity < 1e-6
- Monitor point stable (< 1% variation)

### Parallel Execution
```bash
# Decompose for 8 cores
decomposePar -case runs/case -numberOfSubdomains 8

# Run parallel
mpirun -np 8 pimpleFoam -parallel -case runs/case

# Reconstruct
reconstructPar -case runs/case
```