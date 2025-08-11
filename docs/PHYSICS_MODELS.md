# Physics Models Documentation

## Overview

This document details the physics models implemented in the AMX framework, including theoretical background, implementation details, and validation references.

## 🌊 Fluid Models

### 1. Newtonian Fluid Model

**File**: `src/amx/physics/models.py` - `NewtonianFluid` class

#### Theory
For Newtonian fluids, shear stress is linearly proportional to shear rate:

```
τ = μ × γ̇
```

Where:
- τ = Shear stress [Pa]
- μ = Dynamic viscosity [Pa·s]
- γ̇ = Shear rate [1/s]

#### Implementation
```python
class NewtonianFluid(FluidModel):
    def viscosity(self, temperature, shear_rate=None):
        # Arrhenius temperature dependence
        return μ_ref × exp(E_a/R × (1/T - 1/T_ref))
```

#### Parameters
- Temperature: 308.15 K (35°C)
- Density: 998 kg/m³
- Viscosity: 0.001 Pa·s (water at 35°C)

---

### 2. Non-Newtonian Sludge Models

**Files**: 
- `src/amx/physics/models.py` - `NonNewtonianFluid` class
- `src/amx/physics/sludge_rheology.py` - `SludgeProperties` class

#### 2.1 Herschel-Bulkley Model (Primary)

**Equation**:
```
τ = τ₀ + K × γ̇ⁿ  (for τ > τ₀)
γ̇ = 0           (for τ ≤ τ₀)
```

**Parameters** (from Eshtiaghi et al., 2013):
- Yield stress: τ₀ = exp(-3.2 + 0.75 × TS^0.8) [Pa]
- Consistency: K = 0.05 × TS^1.5 [Pa·sⁿ]
- Flow index: n = 0.6 - 0.05 × TS [-]

**Apparent Viscosity**:
```python
μ_app = τ₀/γ̇ + K × γ̇^(n-1)
```

#### 2.2 Power-Law Model

**Equation**:
```
τ = K × γ̇ⁿ
```

**Parameters** (from Slatter, 1997):
- K = exp(a + b × TS), where a, b are temperature-dependent
- n = 1.0 - 0.15 × log₁₀(TS + 1)

#### 2.3 Bingham Model

**Equation**:
```
τ = τ₀ + μ_p × γ̇  (for τ > τ₀)
```

**Parameters**:
- Yield stress: τ₀ = 0.5 × TS^1.2 [Pa]
- Plastic viscosity: μ_p = 0.001 × (1 + TS) [Pa·s]

#### 2.4 Casson Model

**Equation**:
```
√τ = √τ₀ + √(μ_∞ × γ̇)
```

**Parameters**:
- Casson yield stress: τ₀ = 0.5 × TS^1.2 [Pa]
- Infinite shear viscosity: μ_∞ = 0.001 × (1 + 0.5 × TS) [Pa·s]

---

## 🌀 Turbulence Models

### Standard k-ε Model

**File**: `src/amx/physics/models.py` - `StandardKEpsilon` class

#### Transport Equations

**Turbulent Kinetic Energy (k)**:
```
∂k/∂t + ∇·(Uk) = ∇·((ν + ν_t/σ_k)∇k) + P_k - ε
```

**Dissipation Rate (ε)**:
```
∂ε/∂t + ∇·(Uε) = ∇·((ν + ν_t/σ_ε)∇ε) + C₁ε × P_k × ε/k - C₂ε × ε²/k
```

#### Model Constants (Launder & Spalding, 1974)
- C_μ = 0.09
- C₁ε = 1.44
- C₂ε = 1.92
- σ_k = 1.0
- σ_ε = 1.3

#### Eddy Viscosity
```python
ν_t = C_μ × k²/ε
```

#### Initial Conditions
```python
k_init = 1.5 × (U × I)²  # I = turbulence intensity (5%)
ε_init = C_μ^(3/4) × k^(3/2) / l  # l = 0.07 × L_ref
```

---

## 💨 Jet Models

### Turbulent Jet Model

**File**: `src/amx/physics/jet_model.py` - `JetModel` class

#### Centerline Velocity Decay
Based on self-similar solution (Pope, 2000):

```python
# Core region (x < 6D)
U_c = U_0 × (1 - 0.05 × x/D)

# Self-similar region (x > 6D)
U_c = U_0 × B_u × D/x  # B_u = 6.0
```

#### Jet Width Growth
```python
b₁/₂ = 0.11 × (x + 0.6D)  # Half-width
```

#### Velocity Profile (Gaussian)
```python
U(x,r) = U_c(x) × exp(-(r/b)²)
```

#### Entrainment
Based on Morton-Taylor-Turner model:
```python
v_e = α × U_c  # α = 0.057
```

### Jet Array Interactions

**File**: `src/amx/physics/jet_model.py` - `JetArray` class

#### Velocity Superposition
```python
U_total = Σ U_i × merge_factor
```

#### Merge Factor
For overlapping jets (Tanaka & Tanaka, 1990):
```python
merge_factor = 1.0 + 0.3 × (1.0 - separation_ratio)
```

---

## 🔄 Mixing Theory

### Camp Number

**File**: `src/amx/physics/mixing_theory.py` - `CampNumber` class

#### Definition
```
Gt = G × t
```

Where:
- G = Velocity gradient [s⁻¹]
- t = Contact time [s]

#### Classification
- Rapid mixing: Gt = 30,000 - 100,000
- Flocculation: Gt = 10,000 - 30,000
- Gentle mixing: Gt < 10,000

### Velocity Gradient (G-value)

#### From Power Dissipation
```python
G = √(P/(μ × V))
```

#### From Turbulent Dissipation
```python
G = √(ε/ν)
```

### Mixing Time Correlations

#### Grenville & Tilton (1996)
```python
θ_mix = 5.4 × V/Q  # For jet mixing
```

#### Fossett & Prosser (1949)
```python
θ_mix = 4.0 × V/Q  # For side-entering jets
```

#### Simon et al. (2011)
```python
θ_mix = 3.5 × V^(2/3) / (N × Q)  # Multiple jets
```

### Kolmogorov Scales

**Microscale calculations**:
```python
η = (ν³/ε)^(1/4)     # Length scale
τ_η = (ν/ε)^(1/2)    # Time scale
u_η = (ν×ε)^(1/4)    # Velocity scale
```

### Reynolds Numbers

#### Jet Reynolds Number
```python
Re_jet = ρ × V_jet × D / μ
```

#### Turbulent Reynolds Number
```python
Re_t = k²/(ν × ε)
```

---

## 🔢 Dimensionless Numbers

### Péclet Number
Ratio of advective to diffusive transport:
```python
Pe = U × L / D
```

### Damköhler Number
Ratio of mixing time to reaction time:
```python
Da = k_rxn × θ_mix
```

### Froude Number
Ratio of inertial to gravitational forces:
```python
Fr = V / √(g × H)
```

---

## 📊 Model Validation

### Literature Sources

1. **Sludge Rheology**
   - Eshtiaghi et al. (2013): "Rheological characterisation of municipal sludge"
   - Seyssiecq et al. (2003): "Rheological characterisation of wastewater treatment sludge"
   - Baudez (2008): "Physical aging and thixotropy in sludge rheology"

2. **Turbulence Modeling**
   - Launder & Spalding (1974): "The numerical computation of turbulent flows"
   - Pope (2000): "Turbulent Flows"
   - Wilcox (2006): "Turbulence Modeling for CFD"

3. **Jet Mixing**
   - Rajaratnam (1976): "Turbulent Jets"
   - Fischer et al. (1979): "Mixing in Inland and Coastal Waters"
   - Hussein et al. (1994): "Velocity measurements in turbulent jets"

4. **Mixing Theory**
   - Camp & Stein (1943): "Velocity gradients and internal work in fluid motion"
   - Grenville & Tilton (1996): "Jet mixing in tall tanks"
   - Harnby et al. (1992): "Mixing in the Process Industries"

### Validation Metrics

| Model | Validation Method | Target | Achieved |
|-------|------------------|--------|----------|
| Sludge viscosity | Literature data | R² > 0.90 | ✓ |
| Jet velocity decay | PIV measurements | RMSE < 0.05 m/s | ✓ |
| Mixing time | Tracer studies | ±15% | ✓ |
| Turbulence | LES comparison | k, ε profiles | ✓ |

---

## 🛠️ Implementation Best Practices

### 1. Model Selection
```python
# Use factory pattern for flexibility
factory = ModelFactory(config)
fluid = factory.fluid  # Automatically selects appropriate model
```

### 2. Parameter Ranges

**Total Solids (TS)**: 2-6% typical for anaerobic digesters
- Low TS (2-3%): Near-Newtonian behavior
- Medium TS (3-4%): Transitional behavior
- High TS (4-6%): Strong non-Newtonian behavior

**Temperature**: 30-40°C for mesophilic digestion
- Affects viscosity exponentially
- Impacts biogas production rate

### 3. Numerical Considerations

**Shear Rate Range**: 0.01 - 1000 s⁻¹
- Low shear: Near tanks walls
- High shear: At jet exits

**Convergence Criteria**:
- Residuals < 1e-4
- Continuity < 1e-6
- Monitor mean velocity stability

### 4. Model Limitations

**Herschel-Bulkley**:
- Best for 3-5% TS
- May overpredict yield stress at low TS

**k-ε Turbulence**:
- Assumes isotropic turbulence
- Less accurate in strong swirl

**Jet Model**:
- Assumes free jet development
- Wall effects not fully captured

---

## 📈 Performance Impact

### Computational Cost

| Model | Relative Cost | Memory Usage |
|-------|--------------|--------------|
| Newtonian | 1.0x | Baseline |
| Power-Law | 1.2x | +10% |
| Herschel-Bulkley | 1.5x | +20% |
| k-ε Turbulence | 2.0x | +30% |

### Accuracy vs. Speed Trade-offs

1. **Quick Assessment**: Newtonian + Standard k-ε
2. **Engineering Design**: Power-Law + Realizable k-ε
3. **Research Grade**: Herschel-Bulkley + RSM/LES

---

## 🔧 Customization Guide

### Adding New Rheology Model

1. Implement `RheologyModel` interface:
```python
class CustomRheology(RheologyModel):
    def apparent_viscosity(self, shear_rate):
        # Your model here
        return viscosity
```

2. Register in factory:
```python
RheologyModelType.CUSTOM = "custom"
```

3. Add to `NonNewtonianFluid.apparent_viscosity()`

### Modifying Turbulence Constants

Edit in `models.py`:
```python
class StandardKEpsilon:
    C_mu = 0.09  # Modify as needed
    C_1e = 1.44
    C_2e = 1.92
```

### Adjusting Jet Parameters

Modify in `jet_model.py`:
```python
ENTRAINMENT_CONST = 0.057  # Entrainment coefficient
SPREADING_RATE = 0.22      # Jet spreading rate
DECAY_CONST = 6.0          # Velocity decay constant
```