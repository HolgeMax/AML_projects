"""
Week 5 Programming Exercises - Representation Invariance
DTU 02460 Advanced Machine Learning

Exercises 5.5 and 5.6:
  - Ex 5.5: Numerical and analytical curve length for c(t) = (2t+1, -t^2)
  - Ex 5.6: Curve length in latent space of a trained Bernoulli VAE
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Make the partA_VAE importable (adjust path to point at m1_exercise/partA_VAE)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../m1_exercise/partA_VAE"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../m1_exercise/partB_DDPM"))

from vae_bernoulli import VAE
from vae_bernoulli import train

# ---------------------------------------------------------------------------
# Curve definitions
# ---------------------------------------------------------------------------

def curve_ex5(t: torch.Tensor) -> torch.Tensor:
    """
    Curve from Exercise 5.3 / 5.5:  c(t) = (2t+1, -t^2),  t in [0, 1].

    Parameters
    ----------
    t : torch.Tensor  shape (N,)

    Returns
    -------
    torch.Tensor  shape (N, 2)
    """
    return torch.stack([2*t + 1, -t**2], dim=-1)

def speed_ex5(t: torch.Tensor) -> torch.Tensor:
    """
    Analytic speed function ||c'(t)|| for curve_ex5.

    From Ex 5.3: c(t) = (2t+1, -t^2)  =>  c'(t) = (2, -2t)  
    =>  ||c'(t)|| = sqrt(2^2 + (-2t)^2) = sqrt(4 + 4t^2) = 2*sqrt(1+t^2)

    Parameters
    ----------
    t : torch.Tensor  shape (N,)

    Returns
    -------
    torch.Tensor  shape (N,)
    """
    return 2*torch.sqrt(1 + t**2)

# --------------------------------------------------------------------------
# plot function
# ---------------------------------------------------------------------------
def plot(func, t0: float = 0.0, t1: float = 1.0, N: int = 100):
    t = torch.linspace(t0, t1, N)
    y = func(t)
    plt.plot(t, y)
    plt.title(f"{func.__name__}(t)")
    plt.xlabel("t")
    plt.ylabel(f"{func.__name__}(t)")
    plt.grid()
    plt.show()

# ---------------------------------------------------------------------------
# Exercise 5.5 - curve length via Eq. 4.2
# ---------------------------------------------------------------------------

def curve_length_discrete(curve_fn, t0: float = 0.0, t1: float = 1.0, N: int = 1000) -> float:
    """
    Approximate curve length using LMLG Eq. 4.2:
        L(c) approx sum_{i=0}^{N-1} || c(t_{i}) - c(t_{i+1}) ||

    Parameters
    ----------
    curve_fn : callable  t (N,) -> points (N, D)
    t0, t1   : float     parameter interval
    N        : int       number of sub-intervals

    Returns
    -------
    float  approximate arc length
    """
    t_values = torch.linspace(t0, t1, N+1)  # N+1 points define N sub-intervals
    points = curve_fn(t_values)  # shape (N+1, D)
    diffs = points[1:] - points[:-1]  # shape (N, D)
    segment_lengths = torch.norm(diffs, dim=1)  # shape (N,)
    total_length = segment_lengths.sum().item()
    return total_length


# ----------------------------------------------------------------------------
# Exercise 5.6 - curve length in latent space of a trained Bernoulli VAE
# ----------------------------------------------------------------------------
#write a computer program that evaluates the length of any latent second-order
#polynomial curve c using Eq. 4.2 in the LMLG book. It is recommended that you
#write the code to support any callable curve c.
def latent_poly_curve(t, a, b, d):
    """
    Example of a second-order polynomial curve in latent space:
    c(t) = a + b*t + d*t^2 -> shape (N, D) where D is the latent dimension (here we use D=1 for simplicity)

    Parameters
    ----------
    t : torch.Tensor  shape (N,)
    a, b, d : float   polynomial coefficients

    Returns
    -------
    torch.Tensor  shape (N, D) where D is the latent dimension (here we use D=1 for simplicity)
    """
    return a+b *t.unsqueeze(1) + d*t.unsqueeze(1)**2

def decoded_curve(t, a, b, d, vae: VAE):

    z = latent_poly_curve(t, a, b, d)  # shape (N, D)
    logits = vae.decoder.decoder_net(z)  # shape (N, data_dim)
    return torch.sigmoid(logits).flatten(start_dim=1)  # shape (N, data_dim)

# Find first image for each digit
def get_image(digit):
    for img, label in mnist:
        if label == digit:
            return img.unsqueeze(0)  # (1, 28, 28)


if __name__ == "__main__":
    # Exercise 5.5 - curve length via Eq. 4.2
    import argparse
    parser = argparse.ArgumentParser(description="Compute curve length for curve_ex5")
    parser.add_argument("--N", type=int, default=1000, help="Number of sub-intervals for discrete approximation")
    parser.add_argument("--t0", type=float, default=0.0, help="Start of parameter interval")
    parser.add_argument("--t1", type=float, default=1.0, help="End of parameter interval")
    parser.add_argument("--latent_curve", action="store_true", help="Compute curve length for a latent polynomial curve in a trained VAE")
    parser.add_argument("--vae_path", type=str, default="vae_M2.pt", help="Path to trained VAE model")
    parser.add_argument("--digit-start", type=int, default=0, help="MNIST digit to use as curve start (default: 0)")
    parser.add_argument("--digit-end",   type=int, default=1, help="MNIST digit to use as curve end (default: 1)")
    parser.add_argument("--plot", action="store_true", help="Plot the curve and speed functions")
    args = parser.parse_args()

    if args.plot == True:
        plot(curve_ex5, t0=args.t0, t1=args.t1, N=args.N)
        plot(speed_ex5, t0=args.t0, t1=args.t1, N=args.N)

    total = curve_length_discrete(curve_ex5, t0=args.t0, t1=args.t1, N=args.N)
    print("Total curve length (discrete):", total)

    if args.latent_curve == True:
        from torchvision import datasets, transforms
        from vae_bernoulli import GaussianPrior, BernoulliDecoder, GaussianEncoder

        # Reconstruct the same architecture used during training (M=2)
        M = 2
        prior = GaussianPrior(M)
        encoder_net = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(784, 512), torch.nn.ReLU(),
            torch.nn.Linear(512, 512), torch.nn.ReLU(),
            torch.nn.Linear(512, M * 2),
        )
        decoder_net = torch.nn.Sequential(
            torch.nn.Linear(M, 512), torch.nn.ReLU(),
            torch.nn.Linear(512, 512), torch.nn.ReLU(),
            torch.nn.Linear(512, 784),
            torch.nn.Unflatten(-1, (28, 28)),
        )
        vae = VAE(prior, BernoulliDecoder(decoder_net), GaussianEncoder(encoder_net))
        vae.load_state_dict(torch.load(args.vae_path, map_location="cpu"))
        vae.eval()

        # Load MNIST test set
        mnist = datasets.MNIST("data/", train=False, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: (0.5 < x).float().squeeze()),
            ]))

        # Get two images corresponding to the specified digits
        x0 = get_image(args.digit_start)
        x1 = get_image(args.digit_end)

        # estimate coefficients a, b, d for a second-order polynomial curve in latent space that connects the two images
        # c(t) = a + b*t + d*t^2  (quadratic interpolation) d = 0 for linear interpolation
        with torch.no_grad():
            a = vae.encoder(x0).mean.squeeze()   # z_start, shape (2,)
            z_end = vae.encoder(x1).mean.squeeze()
        b = z_end - a
        d = torch.zeros(2)

        print(f"z_start (digit {args.digit_start}): {a.numpy()}")
        print(f"z_end   (digit {args.digit_end}):   {z_end.numpy()}")
        print(f"d (digit {args.digit_end}):   {d.numpy()}")
        # Compute curve length for the decoded curve
        with torch.no_grad():
            total_latent = curve_length_discrete(lambda t: decoded_curve(t, a, b, d, vae), t0=args.t0, t1=args.t1, N=args.N)
        print("Total curve length in decoded space (discrete):", total_latent)

        if args.plot==True:
            t_vis = torch.linspace(args.t0, args.t1, 100)

            # --- Plot 1: curve in latent space ---
            with torch.no_grad():
                z_curve = latent_poly_curve(t_vis, a, b, d).numpy()  # (100, 2)
            fig, ax = plt.subplots()
            ax.plot(z_curve[:, 0], z_curve[:, 1], 'b-', label="latent curve")
            ax.scatter([a[0].item()], [a[1].item()], c='green', zorder=5, label=f"digit {args.digit_start}")
            ax.scatter([z_end[0].item()], [z_end[1].item()], c='red', zorder=5, label=f"digit {args.digit_end}")
            ax.set_title("Latent curve (z-space)")
            ax.set_xlabel("z1"); ax.set_ylabel("z2")
            ax.legend(); ax.grid()
            plt.tight_layout()
            plt.show()

            # --- Plot 2: decoded images along the curve ---
            n_steps = 8
            t_steps = torch.linspace(args.t0, args.t1, n_steps)
            with torch.no_grad():
                decoded = decoded_curve(t_steps, a, b, d, vae)  # (n_steps, 784)
            decoded_imgs = decoded.reshape(n_steps, 28, 28).numpy()

            fig, axes = plt.subplots(1, n_steps, figsize=(2 * n_steps, 2))
            for i, ax in enumerate(axes):
                ax.imshow(decoded_imgs[i], cmap='gray', vmin=0, vmax=1)
                ax.axis('off')
                ax.set_title(f"t={t_steps[i]:.2f}", fontsize=8)
            fig.suptitle(f"Decoded curve: digit {args.digit_start} → digit {args.digit_end}")
            plt.tight_layout()
            plt.show()
