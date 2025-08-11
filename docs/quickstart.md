# Quick Start Guide

## 🚀 Installation

### Prerequisites
- **OpenFOAM v11** (required for CFD simulations)
- **Python 3.11+** (for framework)
- **16GB RAM** minimum (32GB recommended for 5M+ cells)
- **Docker** (optional, for containerized deployment)

### Install Options

#### Option 1: Local Installation
```bash
# Clone repository
git clone https://github.com/your-org/anaerobic-mixing-openfoam
cd anaerobic-mixing-openfoam

# Install Python package with dependencies
pip install -e .

# Or with development tools
pip install -e .[dev]

# Verify installation
amx --version

# Check OpenFOAM
./scripts/check_openfoam.sh
```

#### Option 2: Docker Installation
```bash
# Build Docker image
docker build -t amx-openfoam docker/

# Run container
docker run -it -v $(pwd):/app amx-openfoam

# Or use docker-compose
docker-compose up -d
```

#### Option 3: HPC/Singularity
```bash
# Build Singularity image
singularity build of11.sif docker/apptainer.def

# Run on HPC
singularity exec of11.sif amx run-case --config configs/case_prod.yaml
```

---

## 📊 Basic Workflow

### Step 1: Configure Your Digester

Create or modify a configuration file:

```yaml
# configs/my_digester.yaml
project: "my_digester_simulation"

geometry:
  tank:
    L: 20.0  # Length [m]
    W: 8.0   # Width [m]
    H: 16.0  # Height [m]
  nozzle:
    count: 32
    throat_diameter_mm: 35.0
    pitch_deg: 45.0  # Upward angle

fluid:
  type: "sludge"      # Use refactored model
  total_solids: 3.5   # [%]
  temperature: 308.15 # [K] 35°C

operation:
  pump_total_m3ph: 430.0
  head_m: 15.0

solver:
  name: "pimpleFoam"
  dt: 0.05
  endTime: 1800  # 30 minutes
```

### Step 2: Run Simulation

```bash
# Full simulation pipeline with refactored framework
amx run-case --config configs/my_digester.yaml --out runs/my_run

# Monitor progress
tail -f runs/my_run/case/log.pimpleFoam

# Check convergence
amx plot-residuals runs/my_run
```

### Step 3: Analyze Results with New Metrics

```bash
# Comprehensive analysis using advanced_metrics.py
amx analyze-mix --in runs/my_run --out data/processed/my_analysis

# View detailed metrics
amx show-metrics data/processed/my_analysis/metrics.json --detailed
```

### Step 4: Visualize

```bash
# 3D visualization
amx visualize --in runs/my_run --field U --time 1800

# Generate comparison plots
amx plot-metrics --metrics data/processed/my_analysis --out figures/
```

### Step 5: Generate Report

```bash
# Complete performance report with recommendations
amx build-report \
  --config configs/my_digester.yaml \
  --metrics data/processed/my_analysis/metrics.json \
  --out reports/performance_report.md
```

---

## 🎯 Common Use Cases

### 1. Quick Test with Newtonian Fluid
```yaml
# configs/test_water.yaml
fluid:
  type: "water"  # Simple Newtonian model
  temperature: 298.15
  density: 998.0
  dynamic_viscosity: 0.001
```

### 2. Industrial Sludge Simulation
```yaml
# configs/industrial_sludge.yaml
fluid:
  type: "sludge"
  total_solids: 4.0  # Higher TS
  rheology_model: "herschel_bulkley"  # Most accurate
```

### 3. Parameter Study with Factory Pattern
```python
from amx.factories import ModelFactory
from amx.config import load_config

# Load base configuration
config = load_config("configs/case_prod.yaml")

# Create models for different TS values
for ts in [2.0, 3.0, 4.0, 5.0]:
    config.fluid.total_solids = ts
    factory = ModelFactory(config)
    fluid = factory.fluid
    
    print(f"TS={ts}%: viscosity={fluid.viscosity(308.15, 10):.4f} Pa·s")
```

### 4. Energy Optimization
```python
from amx.energy import EnergyCalculator
from amx.physics.mixing_theory import MixingTheory

config = load_config("configs/case_prod.yaml")
energy_calc = EnergyCalculator(config)

# Find optimal G-value
optimal = energy_calc.optimize_operation(
    target_g_value=50,
    max_power_w=50000
)

print(f"Optimal flow rate: {optimal['required_flow_m3h']:.1f} m³/h")
print(f"Power consumption: {optimal['required_power_w']/1000:.1f} kW")
```

### 5. PIV Validation
```bash
# Compare CFD with experimental PIV data
amx analyze-piv \
  --config configs/piv_lab.yaml \
  --cfd runs/my_run \
  --piv data/raw/piv \
  --out validation/

# Generate validation report
amx piv-report validation/ --out piv_validation.md
```

---

## 📁 Output Structure (Refactored)

```
runs/my_run/
├── case/                 # OpenFOAM case
│   ├── 0/               # Initial conditions
│   ├── constant/        # Properties & fvOptions
│   │   ├── fvOptions    # Enhanced momentum sources
│   │   └── transportProperties
│   ├── system/          # Solver settings
│   └── 1800/           # Final time step
├── postProcessing/      # Sampled data
├── logs/               # Execution logs
├── models/             # Serialized model instances
│   ├── fluid.json      # Fluid model parameters
│   ├── turbulence.json # Turbulence model
│   └── mixing.json     # Mixing model
└── config.json         # Used configuration
```

Analysis results:
```
data/processed/my_analysis/
├── metrics.json        # Comprehensive metrics
├── advanced_metrics.json # Industrial-grade analysis
├── fields/            # Processed field data
├── figures/           # Generated plots
├── recommendations.txt # Auto-generated improvements
└── report.md          # Full report
```

---

## ⚙️ Using the Refactored Framework

### Model Creation with Factories
```python
from amx.factories import ModelFactory
from amx.config import load_config

# Load configuration
config = load_config("configs/case_prod.yaml")

# Create all models automatically
factory = ModelFactory(config)
models = factory.create_all()

# Access individual models
fluid = models['fluid']
turbulence = models['turbulence']
mixing = models['mixing']

# Get fluid properties
print(f"Density: {fluid.density(308.15)} kg/m³")
print(f"Viscosity at γ=10 s⁻¹: {fluid.viscosity(308.15, 10)} Pa·s")
```

### Advanced Metrics Analysis
```python
from amx.post.advanced_metrics import MixingMetricsAdvanced

# Load simulation results
mesh_data = load_mesh("runs/my_run/case/1800")
field_data = load_fields("runs/my_run/case/1800")

# Create analyzer
analyzer = MixingMetricsAdvanced(mesh_data, field_data)

# Generate comprehensive report
report = analyzer.generate_report()

print(f"Dead zones: {report['dead_zones']['dead_zone_fraction']*100:.1f}%")
print(f"Energy uniformity: {report['energy_metrics']['energy_uniformity']:.2f}")
print(f"Mixing quality: {report['mixing_indices']['quality_grade']}")
```

### Custom Physics Models
```python
from amx.core.interfaces import RheologyModel
from amx.physics.models import NonNewtonianFluid

# Create custom rheology
class MyCustomRheology(RheologyModel):
    def apparent_viscosity(self, shear_rate):
        # Your custom model
        return 0.001 * (1 + shear_rate**0.5)
    
    def shear_stress(self, shear_rate):
        return self.apparent_viscosity(shear_rate) * shear_rate
    
    def model_parameters(self):
        return {"type": "custom", "description": "My model"}

# Use in simulation
custom_fluid = NonNewtonianFluid(
    total_solids=3.5,
    model_type="custom"
)
```

---

## 🔍 Monitoring & Validation

### Real-time Monitoring
```bash
# Monitor solver progress
amx monitor runs/my_run --refresh 5

# Check convergence criteria
amx check-convergence runs/my_run --criteria 1e-4
```

### Performance Validation
```python
# Validate against targets
from amx.validation import PerformanceValidator

validator = PerformanceValidator(config)
results = validator.validate(metrics)

for criterion, passed in results.items():
    status = "✓" if passed else "✗"
    print(f"{status} {criterion}")
```

---

## 🐛 Troubleshooting

### Common Issues & Solutions

1. **Non-Newtonian Model Convergence**
```yaml
# Adjust solver settings for sludge
solver:
  relaxationFactors:
    U: 0.7      # Reduce from 0.9
    p: 0.3      # Reduce from 0.5
```

2. **High Dead Zones with Refactored Models**
```python
# Check jet momentum with new model
from amx.physics.jet_model import JetArray
jets = JetArray(jet_models, positions, tank_center)
velocity_field = jets.velocity_field(x, y, z)
```

3. **Factory Creation Errors**
```python
# Debug factory creation
from amx.factories import FluidFactory
try:
    fluid = FluidFactory.create("sludge", total_solids=3.5)
except ConfigurationError as e:
    print(f"Configuration error: {e.details}")
```

4. **Memory Issues with Advanced Metrics**
```python
# Use chunked processing
analyzer = MixingMetricsAdvanced(mesh_data, field_data)
analyzer.process_in_chunks(chunk_size=100000)
```

---

## 📚 Next Steps

1. **Explore Physics Models**: See [PHYSICS_MODELS.md](PHYSICS_MODELS.md)
2. **Understand Architecture**: Read [ARCHITECTURE.md](ARCHITECTURE.md)
3. **Follow Process Flow**: Study [PROCESS_FLOW.md](PROCESS_FLOW.md)
4. **Check File Mapping**: Reference [FILE_MAPPING.md](FILE_MAPPING.md)
5. **API Documentation**: Review [API_REFERENCE.md](API_REFERENCE.md)

### Advanced Topics
- Custom turbulence models
- Parallel processing optimization
- Machine learning integration
- Real-time optimization

### Support
- GitHub Issues: Report bugs and request features
- Documentation: Comprehensive guides in `docs/`
- Examples: Working examples in `examples/`
- Tests: Unit and integration tests in `tests/`