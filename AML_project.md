# AML Projects — Precise Instructions
DTU 02460 Advanced Machine Learning | MSc Biomedical Engineering

---

## Repository Structure

```
AML_projects/
├── m1_exercise/
│   ├── partA_VAE/          # VAE with learnable priors (Gaussian / MoG / Flow)
│   │   ├── vae_bernoulli.py
│   │   ├── Training_log.csv
│   │   └── graphs/
│   └── partB_DDPM/         # DDPM + Latent DDPM
│       ├── ddpm.py
│       ├── latent_DDPM.py
│       ├── unet.py
│       ├── flow.py
│       ├── fid.py
│       ├── vae_bernoulli.py
│       └── visualize.py
└── m2_exercise/
    ├── w5.py               # Representation invariance & curve lengths
    └── w6.py               # Geodesic distances on Riemannian manifolds
```

---

## Module 1 — Mini Project 1

**Deadline context:** Week 4. Submitted report: `MiniProject1_AdvancedML__Copy_.pdf`
**Team:** Morten (s224022), Holger (s214776), Karol (s243920)
**Dataset:** MNIST (28×28 binary images, 60 000 train / 10 000 test)

---

### Part A — VAE with Learnable Priors (`m1_exercise/partA_VAE/`)

**Goal:** Replace the standard Gaussian prior with learnable alternatives and compare test ELBO.

#### Architecture (fixed for all experiments)

| Component | Layers |
|-----------|--------|
| Encoder | 784 → 512 → 512 → 2M (mean + log-std) |
| Decoder | M → 512 → 512 → 784 (Bernoulli logits) |

Activation: ReLU throughout. Latent dim `M = 10` (training), `M = 2` for visualisation.

#### The three priors to implement

**1. Standard Gaussian (`--prior gaussian`)**
```
p(z) = N(0, I)
KL[q(z|x) || p(z)] = closed-form Gaussian KL
```

**2. Mixture of Gaussians (`--prior mog`)**
```
p(z) = (1/K) * sum_k N(z | mu_k, I)
KL approximated via Monte Carlo: KL ≈ E_q[log q(z|x)] - E_q[log p(z)]
Learnable parameters: mu_k for K components
```

**3. Flow-based prior (`--prior flow`)**
```
p(z) = p_base(f^{-1}(z)) * |det df^{-1}/dz|
Use masked coupling layers (RealNVP-style)
Log-prob: log p(z) = log p_base(f^{-1}(z)) + sum log |s(z)|  (inverse coupling)
```

#### ELBO objective

```
ELBO(x) = E_{q(z|x)}[log p(x|z)] - KL[q(z|x) || p(z)]
       = reconstruction_term - kl_term
Loss = -ELBO  (minimise)
```

#### Training configuration

```bash
# From m1_exercise/partA_VAE/
python vae_bernoulli.py train \
    --prior gaussian \       # or mog / flow
    --latent-dim 10 \
    --epochs 15 \
    --batch-size 128 \
    --device mps             # or cpu / cuda
```

#### Results achieved (Table 1 of report)

| Prior    | Test ELBO (↑ better) | Variance |
|----------|----------------------|----------|
| Gaussian | 86.28                | 0.034    |
| MoG      | 85.23                | 0.088    |
| Flow     | 84.05                | 0.149    |

Run each prior **10 times** (seeds vary) to estimate variance. Flow achieves the best ELBO.

#### Visualisation (Figure 1)

Plot prior contours and aggregate posterior samples in 2-D latent space:

```bash
python vae_bernoulli.py sample \
    --prior flow \
    --latent-dim 2 \
    --model <saved_model.pt> \
    --plot-file prior_flow.png
```

---

### Part B — DDPM and Latent DDPM (`m1_exercise/partB_DDPM/`)

**Goal:** Implement standard DDPM on image space and a latent DDPM on the β-VAE latent space; compare FID scores and sampling speed.

#### DDPM (image space)

**Forward process** (fixed Markov chain):
```
q(x_t | x_{t-1}) = N(x_t | sqrt(1-beta_t) x_{t-1}, beta_t I)
q(x_t | x_0)     = N(x_t | sqrt(alpha_bar_t) x_0, (1-alpha_bar_t) I)
alpha_bar_t       = prod_{s=1}^t (1 - beta_s)
```

**Negative ELBO (Algorithm 1, Ho et al. 2020)**:
```python
# In ddpm.py → DDPM.nelbo(x)
t = uniform sample from {1, ..., T}
eps = N(0, I)   (same shape as x)
x_t = sqrt(alpha_bar_t) * x + sqrt(1 - alpha_bar_t) * eps
eps_pred = network(x_t, t)
loss = ||eps - eps_pred||^2
```

**Sampling (Algorithm 2, Ho et al. 2020)**:
```python
# In ddpm.py → DDPM.sample(n_samples)
x_T ~ N(0, I)
for t in range(T, 0, -1):
    z = N(0, I) if t > 1 else 0
    x_{t-1} = (1/sqrt(alpha_t)) * (x_t - beta_t/sqrt(1-alpha_bar_t) * eps_theta(x_t, t)) + sigma_t * z
```

**Network:** U-Net (`unet.py`) — 5-level encoder-decoder, skip connections, time-conditioned.

```bash
# Train DDPM on MNIST images
python ddpm.py train --epochs 100 --batch-size 128 --device mps
# Generate samples + compute FID
python ddpm.py sample --model ddpm.pt --n-samples 10000
```

#### Latent DDPM (two-stage)

**Stage 1 — Train β-VAE:**
```
ELBO_beta(x) = E_{q(z|x)}[log p(x|z)] - beta * KL[q(z|x) || N(0,I)]
```
High β forces well-structured latent space. Recommended: β ∈ {1e-6, 1e-4, 1e-2}.

```bash
# In latent_DDPM.py
python latent_DDPM.py train \
    --vae-epochs 50 \
    --beta 1e-4 \
    --latent-dim 32 \
    --device mps
```

**Stage 2 — Train DDPM in latent space (freeze VAE weights):**
```python
# Freeze VAE
for p in vae.parameters():
    p.requires_grad = False

# Collect latent codes
z_dataset = [vae.encoder.mean(x) for x in data_loader]

# Train DDPM on z_dataset (much smaller dimensionality: 32 vs 784)
```

**Sampling from Latent DDPM:**
```python
z_samples = ddpm.sample(n_samples)          # sample in latent space
x_samples = vae.decoder.sample(z_samples)   # decode to image space
```

#### Results achieved (Table 2 of report)

| Model              | FID (↓ better) | s/batch   |
|--------------------|---------------|-----------|
| VAE (Flow prior)   | 6.98          | 0.018     |
| DDPM               | **6.07**      | 134.78    |
| Latent DDPM β=1e-4 | 6.60          | 0.15      |
| Latent DDPM β=1e-2 | 7.07          | 0.16      |

**Key insight:** DDPM achieves best FID but is ~10⁴× slower. Latent DDPM is a practical middle ground.

#### FID computation

```python
# fid.py
from fid import compute_fid
fid_score = compute_fid(
    x_real,                          # (N, 1, 28, 28) in [-1, 1]
    x_gen,                           # (N, 1, 28, 28) in [-1, 1]
    classifier_ckpt="mnist_classifier.pth"
)
```

#### Visualisation

```bash
# Visualise β-VAE prior vs learned Latent DDPM distribution
python visualize.py \
    --vae-model vae_model.pt \
    --ddpm-model latent_ddpm.pt \
    --latent-dim 32 \
    --device mps
```

Produces two plots:
1. β-VAE prior contours + aggregate posterior with digit labels
2. KDE-estimated Latent DDPM density + aggregate posterior

---

## Module 2 — Weekly Programming Exercises

### Week 5 — Representation Invariance (`m2_exercise/w5.py`)

**Core idea:** Euclidean distances in VAE latent space are not identifiable (any smooth bijection of the prior preserves the model). Distances computed in *data space* (after decoding) are identifiable.

**Prerequisite:** Train a 2-D Bernoulli VAE:
```bash
cd m1_exercise/partA_VAE
python vae_bernoulli.py train --latent-dim 2 --epochs 10 --device cpu
# Saves model as model.pt
```

#### Ex 5.5 — Curve length for c(t) = (2t+1, −t²)

**5.5.1** Implement `curve_length_discrete` (Eq. 4.2):
```
L(c) ≈ sum_{i=0}^{N-1} || c(t_{i+1}) - c(t_i) ||
```

**5.5.2** Implement `curve_ex5`, `speed_ex5`, and `curve_length_speed` (Eq. 4.5):
```
c(t)  = (2t+1, -t^2)
c'(t) = (2, -2t)   =>   ||c'(t)|| = 2*sqrt(1 + t^2)
L(c)  = integral_0^1 2*sqrt(1+t^2) dt = sqrt(2) + arcsinh(1) ≈ 2.2956
```

Expected: discrete and speed-integral methods both converge to ≈ 2.2956 as N → ∞.

#### Ex 5.6 — Curve length in VAE latent space

**5.6.1** Implement `latent_polynomial_curve` (quadratic Bézier):
```
c(t) = (1-t)^2 * p0 + 2(1-t)t * p1 + t^2 * p2
```

**5.6.2** Implement `latent_curve_length` (decode, then measure chord lengths in data space):
```
L ≈ sum_{i=0}^{N-1} || decode_mean(c(t_{i+1})) - decode_mean(c(t_i)) ||
```

```bash
cd m2_exercise
python w5.py --model ../m1_exercise/partA_VAE/model.pt --device cpu --N 1000
```

---

### Week 6 — Geodesic Distances (`m2_exercise/w6.py`)

**Core idea:** The geodesic (shortest path on the data manifold) between two latent points is found by minimising the Riemannian energy of a curve.

#### Ex 6.1 — Riemannian metric tensor

```
G(z) = J(z)^T J(z)   where J(z) = d(decode_mean(z))/dz   shape: (D, M)
G(z)  shape: (M, M)
```

Implement using `torch.func.jacrev` or `torch.autograd.functional.jacobian`.

#### Ex 6.2 — Riemannian energy and length

```
E(c) = integral_{t0}^{t1} c'(t)^T G(c(t)) c'(t) dt
L(c) = integral_{t0}^{t1} sqrt( c'(t)^T G(c(t)) c'(t) ) dt
```

Use autodiff to compute c'(t) at each quadrature point.
**Note:** By Cauchy-Schwarz: L² ≤ (t1−t0) · E. A curve is a geodesic iff it has constant speed.

#### Ex 6.3 — Geodesic via energy minimisation

Parameterise the curve by n interior control points initialised on the straight line z0 → z1, then minimise E(c) with Adam:

```python
# Pseudocode
control_pts = lerp(z0, z1, steps=n_control)  # initialise on straight line
control_pts.requires_grad_(True)
optimizer = torch.optim.Adam([control_pts], lr=1e-2)

for _ in range(n_steps):
    optimizer.zero_grad()
    loss = curve_energy(piecewise_curve(control_pts), model)
    loss.backward()
    optimizer.step()
```

#### Ex 6.4 — Visualisation

Compare:
- **Euclidean distance:** `||z0 - z1||`
- **Geodesic length:** `curve_length_riemannian(geodesic_curve, model)`

Plot the aggregate posterior in 2-D, overlay straight line vs geodesic path.

```bash
cd m2_exercise
python w6.py \
    --model ../m1_exercise/partA_VAE/model.pt \
    --device cpu \
    --n-control 5 \
    --n-steps 500
```

---

## Environment Setup

```bash
# From Advanced_ML/ (source of truth for deps)
uv sync

# Or from AML_projects/ using the same Python env:
cd ../Advanced_ML
uv run python <script>
```

Dependencies: `torch >= 2.10.0`, `torchvision`, `tqdm`, `matplotlib`, `numpy`, `scipy`

---

## Key References

| Topic | Reference |
|-------|-----------|
| VAE | Kingma & Welling (2013) arXiv:1312.6114 |
| DDPM | Ho, Jain & Abbeel (2020) arXiv:2006.11239 |
| Normalizing Flows | Papamakarios et al. (2021) review |
| Riemannian geometry for ML | Tomczak (2024) *Deep Generative Modeling*, Springer 2nd ed. — LMLG book |
| FID | Heusel et al. (2017) NeurIPS |
