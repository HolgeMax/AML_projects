import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import torch.nn as nn
import torch.distributions as td
import torch.nn.functional as F
from tqdm import tqdm
from unet import Unet
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KernelDensity
from matplotlib.lines import Line2D
import csv
from datetime import datetime
from pathlib import Path
from flow import GaussianBase, MaskedCouplingLayer, Flow
from torch.distributions import MixtureSameFamily
import torch.utils.data
from vae_bernoulli import GaussianPrior, GaussianEncoder, BernoulliDecoder
from ddpm import DDPM, FcNetwork
from latent_DDPM import Beta_VAE, GaussianDecoder
from torchvision import datasets, transforms
from scipy.stats import multivariate_normal



def plot_gaussian_contours(mean, cov, ax, levels=6, color="black"):
    x = np.linspace(mean[0] - 4*np.sqrt(cov[0,0]), mean[0] + 4*np.sqrt(cov[0,0]), 200)
    y = np.linspace(mean[1] - 4*np.sqrt(cov[1,1]), mean[1] + 4*np.sqrt(cov[1,1]), 200)
    X, Y = np.meshgrid(x, y)
    pos = np.dstack((X, Y))
    rv = multivariate_normal(mean, cov)
    Z = rv.pdf(pos)
    ax.contour(X, Y, Z, levels=levels, colors=color, linewidths=1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Lambda(lambda x: x + torch.rand_like(x) / 255.0),
                            transforms.Lambda(lambda x: (x-0.5) * 2.0 ),
                            transforms.Lambda(lambda x: x.flatten())
                        ])
train_data = datasets.MNIST('data/',
                                    train = True,
                                    download = True,
                                    transform = transform)
dataloader = torch.utils.data.DataLoader(train_data, batch_size=256, shuffle=True)

M=10
    # Define the network
num_hidden = 256
network = FcNetwork(M, num_hidden)
    #network = Unet()

    # Set the number of steps in the diffusion process
T = 1000

    # Define model
ddpm = DDPM(network, T=T).to(device)
    
    
prior = GaussianPrior(M)
encoder_net = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, M*2),
    )

decoder_net = nn.Sequential(
        nn.Linear(M, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 784),
        #nn.Unflatten(-1, (28, 28))
    )

    # Define VAE model
decoder = GaussianDecoder(decoder_net)
encoder = GaussianEncoder(encoder_net)
vae = Beta_VAE(prior, decoder, encoder, beta=1e-6 ,prior_type='gaussian').to(device)

ddpm.load_state_dict(torch.load('ddpm1e-6', map_location=torch.device(device)))
vae.load_state_dict(torch.load('vae1e-6', map_location=torch.device(device)))

vae.to(device)
ddpm.to(device)

vae.eval()
ddpm.eval()

# -------------------------
# 1. Collect VAE aggregate posterior
# -------------------------
Z_post = []
Y_post = []

MAX_SAMPLES = 10000

with torch.no_grad():
    for x, y in dataloader:
        x = x.to(device)

        q_z = vae.encoder(x)   # GaussianEncoder
        z = q_z.sample()

        Z_post.append(z.cpu())
        Y_post.append(y)

        if sum(t.shape[0] for t in Z_post) >= MAX_SAMPLES:
            break

Z_post = torch.cat(Z_post, dim=0)[:MAX_SAMPLES]
Y_post = torch.cat(Y_post, dim=0)[:MAX_SAMPLES]

N, D = Z_post.shape
print(f"Collected {N} labeled latent samples (D={D})")

# -------------------------
# 2. Sample β-VAE prior
# -------------------------


# -------------------------
# 3. Sample learned latent DDPM distribution
# -------------------------
with torch.no_grad():
    z_0 = ddpm.sample((N, D))
    Z_ddpm = z_0.cpu()  # full reverse diffusion
    Z_ddpm = z_0.cpu()


# -------------------------
# 4. PCA (fit ONLY on aggregate posterior)
# -------------------------
pca = PCA(n_components=2)
Z_post_2d = pca.fit_transform(Z_post.numpy())
#Z_prior_2d = pca.transform(Z_prior.numpy())
Z_ddpm_2d  = pca.transform(Z_ddpm.numpy())

# -------------------------
# 5. Plot 1: β-VAE prior vs aggregate posterior
# -------------------------
W = pca.components_
cov_prior_pca = W @ W.T            # prior covariance in PCA space
mean_prior_pca = np.zeros(2)

# Build Gaussian grid explicitly
x = np.linspace(-4, 4, 200)
y = np.linspace(-4, 4, 200)
X_prior, Y_prior = np.meshgrid(x, y)
pos = np.dstack((X_prior, Y_prior))

rv = multivariate_normal(mean_prior_pca, cov_prior_pca)
Z_prior_pdf = rv.pdf(pos)

fig, ax = plt.subplots(figsize=(6, 6))

# Aggregate posterior
sc = ax.scatter(
    Z_post_2d[:, 0],
    Z_post_2d[:, 1],
    c=Y_post.numpy(),
    cmap="tab10",
    s=6,
    alpha=0.6
)

# β-VAE prior contours (dashed)
ax.contour(
    X_prior,
    Y_prior,
    Z_prior_pdf,
    levels=6,
    colors="black",
    linestyles="dashed",
    linewidths=1.2
)

# Legend proxy
prior_proxy = Line2D(
    [0], [0],
    color="black",
    linestyle="dashed",
    linewidth=1.2,
    label="β-VAE prior"
)

plt.colorbar(sc, ticks=range(10), label="MNIST digit")
ax.legend(handles=[prior_proxy])
ax.set_xlabel("PCA-1")
ax.set_ylabel("PCA-2")
ax.set_title("β-VAE prior vs aggregate posterior")
plt.tight_layout()
plt.savefig("Beta_VAE_prior")
plt.show()

# -------------------------
# 6. Plot 2: latent DDPM vs aggregate posterior
# -------------------------

kde = KernelDensity(bandwidth=0.3)
kde.fit(Z_ddpm_2d)

x = np.linspace(Z_post_2d[:,0].min()-1, Z_post_2d[:,0].max()+1, 200)
y = np.linspace(Z_post_2d[:,1].min()-1, Z_post_2d[:,1].max()+1, 200)
X, Y = np.meshgrid(x, y)
grid = np.vstack([X.ravel(), Y.ravel()]).T

log_dens = kde.score_samples(grid)
Z = np.exp(log_dens).reshape(X.shape)

fig, ax = plt.subplots(figsize=(6, 6))

# Aggregate posterior
sc = ax.scatter(
    Z_post_2d[:, 0],
    Z_post_2d[:, 1],
    c=Y_post.numpy(),
    cmap="tab10",
    s=6,
    alpha=0.6
)

# DDPM density contours (dashed, red)
cs = ax.contour(
    X, Y, Z,
    levels=6,
    colors="black",
    linestyles="dashed",
    linewidths=1.2
)

# Legend proxy
ddpm_proxy = Line2D(
    [0], [0],
    color="black",
    linestyle="dashed",
    linewidth=1.2,
    label="Latent DDPM density"
)

plt.colorbar(sc, ticks=range(10), label="MNIST digit")
ax.legend(handles=[ddpm_proxy])
ax.set_xlabel("PCA-1")
ax.set_ylabel("PCA-2")
ax.set_title("Latent DDPM prior vs aggregate posterior")
plt.tight_layout()
plt.savefig("Latent_DDPM_prior")
plt.show()