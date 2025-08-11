# Anaerobic Mixing OpenFOAM

Industrial-grade CFD simulation framework for anaerobic digester mixing with advanced non-Newtonian rheology, comprehensive validation, and optimization capabilities.

## 🚀 Overview

This project provides a production-ready solution for simulating and optimizing mixing in anaerobic digesters:

### Core Technologies
- **OpenFOAM v11** - High-fidelity CFD simulations
- **Python 3.11+** - Refactored framework with clean architecture
- **PyVista** - Advanced 3D visualization
- **OpenPIV** - Experimental validation

### Key Features

- ✅ **Industrial-Grade Physics**
  - Non-Newtonian sludge rheology (Herschel-Bulkley, Power-law, Casson)
  - Temperature-dependent properties
  - Validated against literature correlations
  
- ✅ **Advanced Mixing Analysis**
  - Dead zone quantification with connectivity analysis
  - Residence time distribution (RTD)
  - Energy uniformity assessment
  - Automatic performance grading
  
- ✅ **Clean Architecture**
  - Abstract base classes and interfaces
  - Factory pattern for model creation
  - Comprehensive error handling
  - Type hints throughout
  
- ✅ **Production Ready**
  - Docker/Singularity support
  - 5-7M cell meshes for industrial accuracy
  - Robust validation framework
  - Automated reporting

## Installation

### Prerequisites

- OpenFOAM v11
- Python 3.11+
- Docker (optional)

### Quick Start

```bash
# Clone repository
git clone https://github.com/your-org/anaerobic-mixing-openfoam
cd anaerobic-mixing-openfoam

# Install Python package
pip install -e .

# Or using Docker
docker build -t amx-openfoam docker/
docker run -it -v $(pwd):/app amx-openfoam
```

## Usage

### Command-Line Interface

```bash
# Run complete simulation
amx run-case --config configs/case_prod.yaml --out runs/prod

# Analyze mixing performance
amx analyze-mix --in runs/prod --out data/processed/prod_metrics

# PIV validation
amx analyze-piv --config configs/piv_lab.yaml --cfd runs/prod --piv data/raw/piv

# Generate figures
amx make-figures --metrics data/processed/prod_metrics --out docs/figures

# Build report
amx build-report --config configs/case_prod.yaml --metrics data/processed/prod_metrics --out report.md
```

### Configuration

Edit YAML configuration files in `configs/`:

```yaml
# configs/case_prod.yaml
project: "digester_prod"
geometry:
  tank:
    L: 20.0  # Length (m)
    W: 8.0   # Width (m)
    H: 16.0  # Height (m)
  nozzle:
    count: 32
    pitch_deg: 45.0  # Upward angle
    array: [4, 8]    # 4x8 array
operation:
  pump_total_m3ph: 430.0  # Total flow rate
  head_m: 15.0           # Pump head
```

## 📁 Project Structure (Refactored)

```
anaerobic-mixing-openfoam/
├── src/amx/                # Main Python package
│   ├── core/              # Core framework
│   │   ├── base.py        # Abstract base classes
│   │   ├── interfaces.py  # Model interfaces
│   │   └── exceptions.py  # Custom exceptions
│   ├── physics/           # Physics models
│   │   ├── models.py      # Unified physics models
│   │   ├── sludge_rheology.py  # Non-Newtonian rheology
│   │   ├── jet_model.py   # Jet dynamics
│   │   └── mixing_theory.py    # Mixing correlations
│   ├── openfoam/          # OpenFOAM interface
│   │   ├── fvoptions.py   # Enhanced momentum sources
│   │   ├── writer.py      # Dictionary generation
│   │   └── runner.py      # Solver execution
│   ├── post/              # Post-processing
│   │   ├── advanced_metrics.py  # Industrial metrics
│   │   └── viz.py         # Visualization
│   ├── factories.py       # Factory pattern implementation
│   └── workflow.py        # Main orchestration
├── configs/               # YAML configurations
├── case_templates/        # OpenFOAM templates
├── tests/                 # Comprehensive tests
└── docker/                # Container definitions
```

## 🎯 Performance Targets

| Metric | Target | Industrial Standard |
|--------|--------|-------------------|
| Mean velocity | ≥0.30 m/s | Ensures active mixing |
| Dead zones | <10% volume | Minimizes settling |
| Mixing time | ≤30 minutes | Optimal for digestion |
| Energy uniformity | >0.7 | Efficient power use |
| MLSS deviation | ≤5% | Homogeneous distribution |

## 🔬 Technical Specifications

### CFD Model
- **Solver**: pimpleFoam (unsteady, incompressible)
- **Turbulence**: k-ε with wall functions
- **Mesh**: 5-7M cells with refinement zones
- **Momentum sources**: 32 jets with Gaussian distribution

### Fluid Properties (Anaerobic Sludge)
- **Density**: 1015 kg/m³ (3.5% TS)
- **Rheology**: Herschel-Bulkley model
- **Viscosity**: 0.0035 Pa·s (apparent)
- **Temperature**: 35°C (mesophilic)

### Validation Metrics
- **PIV correlation**: R > 0.85
- **RMSE**: < 0.05 m/s
- **Literature validation**: ✓

## Development

### Running Tests

```bash
# Install dev dependencies
pip install -e .[dev]

# Run tests
pytest

# Run with coverage
pytest --cov=amx

# Linting
ruff check .
black --check .
```

### Building Documentation

```bash
mkdocs serve  # Local preview
mkdocs build  # Build static site
```

## HPC Deployment

For HPC clusters using Singularity/Apptainer:

```bash
# Build container
apptainer build of11.sif docker/apptainer.def

# Submit job
sbatch submit_job.sh
```

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing`)
5. Open Pull Request

## License

MIT License - see LICENSE file for details.

## 🔄 Recent Improvements

### Refactoring Highlights
1. **Clean Architecture** - Abstract base classes, interfaces, and factory patterns
2. **Industrial Physics** - Non-Newtonian rheology with literature-validated correlations
3. **Enhanced Metrics** - Comprehensive dead zone analysis, RTD, energy uniformity
4. **Robust Error Handling** - Custom exception hierarchy with detailed error messages
5. **Type Safety** - Full type hints for better IDE support and maintainability

### Performance Improvements
- **Mesh Quality**: Increased to 5-7M cells for industrial accuracy
- **Momentum Sources**: Gaussian distribution for realistic jet modeling
- **Fluid Properties**: Accurate sludge rheology (TS-dependent viscosity)
- **Analysis Speed**: Optimized post-processing with vectorized operations

## 📚 Citation

If you use this software in your research, please cite:

```bibtex
@software{anaerobic_mixing_openfoam,
  title = {Anaerobic Mixing OpenFOAM: Industrial-Grade CFD Framework},
  author = {Engineering Team},
  year = {2024},
  version = {2.0},
  url = {https://github.com/your-org/anaerobic-mixing-openfoam}
}
```

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/your-org/anaerobic-mixing-openfoam/issues)
- **Documentation**: [Read the Docs](https://anaerobic-mixing-openfoam.readthedocs.io)
- **Email**: support@example.com

## 🙏 Acknowledgments

This framework incorporates validated correlations from:
- Eshtiaghi et al. (2013) - Sludge rheology characterization
- Camp & Stein (1943) - Mixing theory fundamentals
- Pope (2000) - Turbulent flows
- Various industrial standards and best practices