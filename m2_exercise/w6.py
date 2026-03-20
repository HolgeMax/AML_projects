"""
Week 6 Programming Exercises - Geodesic Distances
DTU 02460 Advanced Machine Learning

Week 6 covers geodesic distances on Riemannian manifolds defined by a generative model.
The exercises build directly on Week 5's curve length machinery.

Key concepts:
  - Riemannian metric tensor: G(z) = J(z)^T J(z)  where J(z) = d(decode(z))/dz
  - Geodesic: shortest curve connecting two points on the manifold
  - Geodesic distance: length of the geodesic in data space

Exercises:
  - Ex 6.1: Compute the Riemannian metric tensor G(z) via automatic differentiation
  - Ex 6.2: Compute energy and length of a straight-line curve in latent space
  - Ex 6.3: Minimize energy over polynomial curves to approximate geodesics
  - Ex 6.4: Visualise geodesic paths and compare Euclidean vs geodesic distances
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Make the partA_VAE importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../m1_exercise/partA_VAE"))


# ---------------------------------------------------------------------------
# Exercise 6.1 - Riemannian metric tensor via autodiff
# ---------------------------------------------------------------------------

def riemannian_metric(model, z: torch.Tensor) -> torch.Tensor:
    """
    Compute the Riemannian metric tensor G(z) = J(z)^T J(z) at latent point z,
    where J(z) = d(decode(z))/dz is the Jacobian of the decoder mean.

    The metric tensor G(z) is (M x M) symmetric positive semi-definite.

    Hint: use torch.func.jacrev (or torch.autograd.functional.jacobian) to
    compute J(z), then compute J^T J.

    Parameters
    ----------
    model : VAE   trained VAE; use model.decoder.mean(z) for the decoder mean
    z     : torch.Tensor  shape (M,)  single latent point

    Returns
    -------
    torch.Tensor  shape (M, M)  Riemannian metric tensor G(z)
    """
    # TODO (Ex 6.1): compute G(z) = J(z)^T J(z) using autodiff
    # Steps:
    #   1. Define a function f(z) = model.decoder.mean(z).flatten()
    #   2. Compute J = jacobian of f w.r.t. z  -> shape (D, M)
    #   3. Return J.T @ J  -> shape (M, M)
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Exercise 6.2 - Energy and length of a curve using the Riemannian metric
# ---------------------------------------------------------------------------

def curve_energy(curve_fn, model, t0: float = 0.0, t1: float = 1.0, N: int = 200) -> float:
    """
    Compute the Riemannian energy of a curve c(t) in latent space:
        E(c) = integral_{t0}^{t1} c'(t)^T G(c(t)) c'(t) dt

    Discretise using a midpoint rule with N sub-intervals.

    Hint: use torch.autograd to compute c'(t) = d(curve_fn(t))/dt at each t_i.

    Parameters
    ----------
    curve_fn : callable  t scalar -> latent point (M,)
    model    : VAE
    t0, t1   : float
    N        : int       number of quadrature intervals

    Returns
    -------
    float  Riemannian energy
    """
    # TODO (Ex 6.2): implement Riemannian energy integral
    # Steps:
    #   1. Sample N midpoints t_i in [t0, t1]
    #   2. For each t_i: compute z_i = curve_fn(t_i), dz_i = d(curve_fn)/dt at t_i
    #   3. Compute G_i = riemannian_metric(model, z_i)
    #   4. Accumulate dz_i^T G_i dz_i * dt
    raise NotImplementedError


def curve_length_riemannian(curve_fn, model, t0: float = 0.0, t1: float = 1.0, N: int = 200) -> float:
    """
    Compute the Riemannian length of a curve c(t) in latent space:
        L(c) = integral_{t0}^{t1} sqrt(c'(t)^T G(c(t)) c'(t)) dt

    Parameters
    ----------
    curve_fn : callable  t scalar -> latent point (M,)
    model    : VAE
    t0, t1   : float
    N        : int

    Returns
    -------
    float  Riemannian length
    """
    # TODO (Ex 6.2b): implement Riemannian length (sqrt of integrand before summing)
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Exercise 6.3 - Geodesic approximation via energy minimisation
# ---------------------------------------------------------------------------

def geodesic_energy_minimisation(
    model,
    z0: torch.Tensor,
    z1: torch.Tensor,
    n_control: int = 5,
    n_steps: int = 500,
    lr: float = 1e-2,
    N: int = 100,
) -> tuple[torch.Tensor, list[float]]:
    """
    Approximate the geodesic between z0 and z1 by minimising the curve energy
    over a polynomial (or piecewise-linear) curve parameterised by n_control
    interior control points.

    The curve is fixed at z0 (t=0) and z1 (t=1); only the n_control interior
    points are optimised.

    Strategy:
      - Initialise interior control points on the straight line z0 -> z1
      - Represent c(t) as a piecewise-linear (or Bezier) curve through control points
      - Minimise E(c) using Adam

    Parameters
    ----------
    model      : VAE
    z0, z1     : torch.Tensor  shape (M,)  endpoints (fixed)
    n_control  : int           number of interior control points
    n_steps    : int           optimisation steps
    lr         : float         learning rate
    N          : int           quadrature points for energy estimate

    Returns
    -------
    control_points : torch.Tensor  shape (n_control, M)  optimised interior points
    loss_history   : list[float]   energy at each optimisation step
    """
    # TODO (Ex 6.3): optimise interior control points to minimise curve_energy
    # Steps:
    #   1. Initialise control_points on the straight line from z0 to z1
    #   2. Build curve_fn(t) as piecewise-linear interpolation through
    #      [z0] + control_points + [z1]
    #   3. Optimise with Adam, recording energy at each step
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Exercise 6.4 - Visualisation and comparison
# ---------------------------------------------------------------------------

def plot_geodesic_vs_euclidean(model, z0, z1, control_points, n_samples=500, device="cpu"):
    """
    Visualise the geodesic path and a straight line in latent space,
    overlaid on the aggregate posterior of the VAE.

    Steps:
      1. Encode a batch of MNIST images to get aggregate posterior samples
      2. Plot samples in 2-D latent space (PCA if M > 2)
      3. Overlay the straight-line path z0 -> z1
      4. Overlay the geodesic path through control_points
      5. Compare Euclidean vs geodesic distances (print both)

    Parameters
    ----------
    model          : VAE
    z0, z1         : torch.Tensor  shape (M,)
    control_points : torch.Tensor  shape (n_control, M)
    n_samples      : int
    device         : str
    """
    # TODO (Ex 6.4): implement visualisation
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Helper: load a trained VAE (same as Week 5)
# ---------------------------------------------------------------------------

def load_vae(checkpoint_path: str, latent_dim: int = 2, device: str = "cpu"):
    """Load a trained Bernoulli VAE from a .pt checkpoint."""
    from vae_bernoulli import GaussianPrior, GaussianEncoder, BernoulliDecoder, VAE

    M = latent_dim
    prior = GaussianPrior(M)

    encoder_net = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 512), nn.ReLU(),
        nn.Linear(512, 512), nn.ReLU(),
        nn.Linear(512, M * 2),
    )
    decoder_net = nn.Sequential(
        nn.Linear(M, 512), nn.ReLU(),
        nn.Linear(512, 512), nn.ReLU(),
        nn.Linear(512, 784),
        nn.Unflatten(-1, (28, 28)),
    )

    model = VAE(prior, BernoulliDecoder(decoder_net), GaussianEncoder(encoder_net))
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device).eval()
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    from torchvision import datasets, transforms

    parser = argparse.ArgumentParser(description="Week 6 programming exercises - Geodesics")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to trained 2-D VAE checkpoint (.pt)")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--n-steps", type=int, default=500,
                        help="Optimisation steps for geodesic minimisation")
    parser.add_argument("--n-control", type=int, default=5,
                        help="Number of interior control points for geodesic")
    args = parser.parse_args()

    model = load_vae(args.model, latent_dim=2, device=args.device)

    # Fixed endpoints in latent space (can be changed)
    z0 = torch.tensor([-1.5, 0.0], device=args.device)
    z1 = torch.tensor([1.5, 0.0], device=args.device)

    # --- Exercise 6.1: metric tensor at midpoint ---
    print("=" * 60)
    print("Exercise 6.1 - Riemannian metric tensor")
    print("=" * 60)
    z_mid = (z0 + z1) / 2
    try:
        G = riemannian_metric(model, z_mid)
        print(f"  G(z_mid) =\n{G}")
    except NotImplementedError:
        print("  [TODO] riemannian_metric not yet implemented")

    # --- Exercise 6.2: energy of straight line ---
    print()
    print("=" * 60)
    print("Exercise 6.2 - Energy of straight-line curve")
    print("=" * 60)

    def straight_line(t):
        return z0 + t * (z1 - z0)

    try:
        E = curve_energy(straight_line, model)
        L = curve_length_riemannian(straight_line, model)
        print(f"  Straight-line Riemannian energy: {E:.4f}")
        print(f"  Straight-line Riemannian length: {L:.4f}")
    except NotImplementedError:
        print("  [TODO] curve_energy / curve_length_riemannian not yet implemented")

    # --- Exercise 6.3: geodesic via energy minimisation ---
    print()
    print("=" * 60)
    print("Exercise 6.3 - Geodesic energy minimisation")
    print("=" * 60)
    try:
        control_pts, loss_hist = geodesic_energy_minimisation(
            model, z0, z1,
            n_control=args.n_control,
            n_steps=args.n_steps,
        )
        print(f"  Initial energy: {loss_hist[0]:.4f}")
        print(f"  Final energy:   {loss_hist[-1]:.4f}")

        # --- Exercise 6.4: visualise ---
        print()
        print("=" * 60)
        print("Exercise 6.4 - Visualisation")
        print("=" * 60)
        try:
            plot_geodesic_vs_euclidean(model, z0, z1, control_pts, device=args.device)
        except NotImplementedError:
            print("  [TODO] plot_geodesic_vs_euclidean not yet implemented")

    except NotImplementedError:
        print("  [TODO] geodesic_energy_minimisation not yet implemented")
