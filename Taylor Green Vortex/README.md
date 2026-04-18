# Taylor-Green Vortex — Residual Correction PINN

A Physics-Informed Neural Network (PINN) that learns to correct a low-fidelity (LF) flow field toward the high-fidelity (HF) solution for the 2D Taylor-Green Vortex (TGV) problem, with adaptive extrapolation beyond the training window.

---

## Problem Setup

The **Taylor-Green Vortex** is a classical benchmark for 2D viscous flow with a known analytical solution, making it ideal for validating machine learning approaches in fluid dynamics.

- **Domain**: `[0, L] × [0, L]`, `L = 4.0`
- **Viscosity**: `ν = 0.001`
- **Training range**: `0 ~ 10s`
- **Extrapolation range**: `10 ~ 30s`

---

## Data Definition

The model is trained to correct the LF approximation toward the HF ground truth.

### High-Fidelity (HF) — Analytical TGV Solution

$$u_{HF} = -\cos(\pi x)\sin(\pi y) e^{-2\pi^2 \nu t}$$

$$v_{HF} = \sin(\pi x)\cos(\pi y) e^{-2\pi^2 \nu t}$$

$$P_{HF} = -0.25\{\cos(2\pi x) + \cos(2\pi y)\} e^{-4\pi^2 \nu t}$$

### Unphysical Noise (simulating LF CFD error)

$$u_{Noise} = 0.1\sin(2\pi x)\sin(2\pi y) e^{-2\pi^2 \nu t}$$

$$v_{Noise} = 0.1\cos(2\pi x)\cos(2\pi y) e^{-2\pi^2 \nu t} $$

$$P_{Noise} = 0.025\sin(\pi x) e^{-4\pi^2 \nu t} $$

### Low-Fidelity (LF) — HF + Noise

$$u_{LF} = u_{HF} + u_{Noise}, \quad v_{LF} = v_{HF} + v_{Noise}, \quad P_{LF} = P_{HF} + P_{Noise}$$

|    HF Data    |        Noise        |    LF Data    |
| :-----------: | :-----------------: | :-----------: |
| ![HF](HF.png) | ![Noise](Noise.png) | ![LF](LF.png) |

---

## Method

### Model Architecture

- **Input**: `(x, y, t)` → 3D coordinate vector
- **Fourier Embedding**: Random Fourier Features to mitigate spectral bias
- **Network**: 5-layer MLP with SiLU activations (hidden dim: 128)
- **Output**: residual corrections `(δu, δv, δP)` added to the LF base solution

$$[\delta u,\, \delta v,\, \delta P] = \mathcal{F}(x, y, t;\, \theta), \qquad \hat{u} = u_{LF} + \delta u, \quad \hat{v} = v_{LF} + \delta v, \quad \hat{P} = P_{LF} + \delta P$$

### Phase 1: Training (0 ~ 10s) — 3-Step Optimization

| Step | Optimizer          | Loss                                                  |
| ---- | ------------------ | ----------------------------------------------------- |
| 1    | Adam               | $\mathcal{L}_{data}$ only                             |
| 2    | Adam               | $100\cdot\mathcal{L}_ {data} + \mathcal{L}_{pyhsics}$ |
| 3    | L-BFGS             | $100\cdot\mathcal{L}_ {data} + \mathcal{L}_{pyhsics}$ |

### Phase 2: Adaptive Extrapolation (10 ~ 30s)

The model extrapolates beyond the training window using a **PDE residual-based adaptive loop**:

- At each time step, $\mathcal{L}_{physics}$ is evaluated against an adaptive threshold
- If residual **exceeds** threshold → **BOOST**: inject HF data and retrain locally (Adam × 300 + L-BFGS × 50)
- If residual **within** threshold → **KEEP**: self-refine using PDE loss + replay buffer regularization (× 200)

This prevents catastrophic forgetting while ensuring physical consistency throughout extrapolation.

---

## Loss Functions

The model predicts residual corrections $(\delta u, \delta v, \delta P)$ between LF and HF:

$$[\delta u,\, \delta v,\, \delta P] = \mathcal{F}(x, y, t;\, \theta)$$

$$\hat{u} = u_{LF} + \delta u, \quad \hat{v} = v_{LF} + \delta v, \quad \hat{P} = P_{LF} + \delta P$$

### Data Loss

$$\mathcal{L}_{data} = \frac{1}{N}\sum_{i=1}^{N}\left[(\hat{u} - u_{HF})^2 + (\hat{v} - v_{HF})^2 + (\hat{P} - P_{HF})^2\right]$$

### Physics Loss (Navier-Stokes)

**Continuity equation:**

$$f_c = \frac{\partial \hat{u}}{\partial x} + \frac{\partial \hat{v}}{\partial y}, \qquad \mathcal{L}_{mass} = \frac{1}{N}\sum_{i=1}^{N} f_c(x_i, y_i, t_i)^2$$

**Momentum equations:**

$$f_u = \frac{\partial \hat{u}}{\partial t} + \hat{u}\frac{\partial \hat{u}}{\partial x} + \hat{v}\frac{\partial \hat{u}}{\partial y} + \frac{\partial \hat{P}}{\partial x} - \nu\left(\frac{\partial^2 \hat{u}}{\partial x^2} + \frac{\partial^2 \hat{u}}{\partial y^2}\right)$$

$$f_v = \frac{\partial \hat{v}}{\partial t} + \hat{u}\frac{\partial \hat{v}}{\partial x} + \hat{v}\frac{\partial \hat{v}}{\partial y} + \frac{\partial \hat{P}}{\partial y} - \nu\left(\frac{\partial^2 \hat{v}}{\partial x^2} + \frac{\partial^2 \hat{v}}{\partial y^2}\right)$$

$$\mathcal{L}_{momentum} = \frac{1}{N}\sum_{i=1}^{N}\left[f_u(x_i, y_i, t_i)^2 + f_v(x_i, y_i, t_i)^2\right]$$

**Total physics loss:**

$$\mathcal{L}_{physics} = \mathcal{L}_{mass} + \mathcal{L}_{momentum}$$

---

## Results

### MSE Comparison (Training + Extrapolation)

![MSE Comparison](mse_log.png)

- **Blue**: Model prediction error (LF + PINN correction)
- **Red dashed**: LF baseline error
- **Stars**: Time steps where HF data was injected (BOOST)
- **Dotted vertical line**: boundary between training (left) and extrapolation (right)

### Improvement Rate over LF Baseline

$$MSE_{LF} = \frac{1}{N}\sum_{i=1}^{N}(u_{LF} - u_{HF})^2, \qquad MSE_{pred} = \frac{1}{N}\sum_{i=1}^{N}(\hat{u} - u_{HF})^2$$

$$Improvement = \frac{MSE_{LF} - MSE_{pred}}{MSE_{LF}} \times 100\%$$

![Improvement Rate](imp_rate.png)

The model maintains **>95% improvement** over the LF baseline throughout the extrapolation range (10 ~ 30s).

---

## File Structure

```
Taylor Green Vortex/
├── train.py          # Phase 1 (training) + Phase 2 (adaptive extrapolation)
├── evaluate.py       # Evaluation and result visualization
├── mse_log.png
├── imp_rate.png
├── HF.png
├── Noise.png
├── LF.png
└── README.md
```

---

## Requirements

```
torch
numpy
matplotlib
pandas
```

```bash
pip install torch numpy matplotlib pandas
```

---

## Usage

```bash
# Step 1: Train the model (Phase 1 + Phase 2)
python train.py

# Step 2: Evaluate and generate result plots
python evaluate.py
```

> **Note**: Before running `evaluate.py`, manually update `manual_boost_times` in the file to match the BOOST time steps logged during Phase 2 training.
