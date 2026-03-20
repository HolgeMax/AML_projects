# based on Code for DTU course 02460 (Advanced Machine Learning Spring) by Jes Frellsen, 2024
# Version 1.2 (2024-02-06)
# Inspiration is taken from:
# - https://github.com/jmtomczak/intro_dgm/blob/main/vaes/vae_example.ipynb
# - https://github.com/kampta/pytorch-distributions/blob/master/gaussian_vae.py




import torch
import torch.nn as nn
import torch.distributions as td
from torch.distributions import MixtureSameFamily
import torch.utils.data
from torch.nn import functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity
from matplotlib.lines import Line2D
import csv
from datetime import datetime
from pathlib import Path
from flow import GaussianBase, MaskedCouplingLayer, Flow


class GaussianPrior(nn.Module):
    def __init__(self, M):
        """
        Define a Gaussian prior distribution with zero mean and unit variance.

                Parameters:
        M: [int] 
           Dimension of the latent space.
        """
        super(GaussianPrior, self).__init__()
        self.M = M
        self.mean = nn.Parameter(torch.zeros(self.M), requires_grad=False)
        self.std = nn.Parameter(torch.ones(self.M), requires_grad=False)

    def forward(self):
        """
        Return the prior distribution.

        Returns:
        prior: [torch.distributions.Distribution]
        """
        return td.Independent(td.Normal(loc=self.mean, scale=self.std), 1)

    def sample(self, sample_shape=torch.Size()):
        """
        Sample z ~ p(z).
        """
        return self.forward().sample(sample_shape)

    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute log p(z) for z of shape (batch_size, latent_dim).
        """
        return self.forward().log_prob(z)

class MoG_Prior(nn.Module):
    def __init__(self, M: int, K: int = 5, multiplier: float = 1.0):
        """
        Mixture-of-Gaussians prior using torch.distributions.MixtureSameFamily.

        Parameters
        ----------
        M : int
            Latent dimension.
        K : int
            Number of mixture components.
        multiplier : float
            Scale for initializing means near 0.
        """
        super(MoG_Prior, self).__init__()
        self.M = M
        self.num_components = K

        # Component params: (K, L)
        self.means = nn.Parameter(torch.randn(K, self.M) * multiplier)
        self.logvars = nn.Parameter(torch.randn(K, self.M))   # log(sigma^2)

        # Mixture logits (kept in the same shape style you used): (K, 1, 1)
        self.w = nn.Parameter(torch.zeros(K, 1, 1))

    def get_params(self):
        return self.means, self.logvars

    def forward(self) -> td.Distribution:
        """
        Return p(z) as a Distribution with event_shape (L,).
        """
        means, logvars = self.get_params()               # (K, L)
        scales = torch.exp(0.5 * logvars)                # std, (K, L)

        mix = td.Categorical(logits=self.w.squeeze())    # (K,)
        comp = td.Independent(td.Normal(means, scales), 1)  # batch_shape (K), event_shape (L,)

        return td.MixtureSameFamily(mix, comp)           # event_shape (L,)

    def sample(self, sample_shape=torch.Size()) -> torch.Tensor:
        """
        Sample z ~ p(z).
        """
        return self.forward().sample(sample_shape)

    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute log p(z). For z shape (B, L) returns (B,).
        """
        return self.forward().log_prob(z)

class FlowPrior(nn.Module):
    def __init__(self, M: int, num_transformations: int = 6, hidden_dim: int = 128):
        """
        RealNVP-style flow prior over latent variables.

        Parameters
        ----------
        M : int
            Latent dimension.
        num_transformations : int
            Number of coupling layers.
        hidden_dim : int
            Hidden layer size in the coupling networks.
        """
        super(FlowPrior, self).__init__()
        self.M = M
        self.num_transformations = num_transformations
        self.hidden_dim = hidden_dim

        base = GaussianBase(M)
        transformations = []

        # Alternating binary mask over latent dimensions.
        mask = torch.tensor([1 if i % 2 == 0 else 0 for i in range(M)], dtype=torch.float32)

        for _ in range(num_transformations):
            mask = 1 - mask
            scale_net = nn.Sequential(
                nn.Linear(M, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, M),
                nn.Tanh(),
            )
            translation_net = nn.Sequential(
                nn.Linear(M, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, M),
            )
            transformations.append(MaskedCouplingLayer(scale_net, translation_net, mask))

        self.flow = Flow(base, transformations)

    def forward(self) -> Flow:
        """
        Return the underlying flow model.
        """
        return self.flow

    def sample(self, sample_shape=torch.Size()) -> torch.Tensor:
        """
        Sample z ~ p(z).
        """
        return self.flow.sample(sample_shape)

    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute log p(z) for z of shape (batch_size, latent_dim).
        """
        return self.flow.log_prob(z)

class GaussianEncoder(nn.Module):
    def __init__(self, encoder_net):
        """
        Define a Gaussian encoder distribution based on a given encoder network.

        Parameters:
        encoder_net: [torch.nn.Module]             
           The encoder network that takes as a tensor of dim `(batch_size,
           feature_dim1, feature_dim2)` and output a tensor of dimension
           `(batch_size, 2M)`, where M is the dimension of the latent space.
        """
        super(GaussianEncoder, self).__init__()
        self.encoder_net = encoder_net

    def forward(self, x):
        """
        Given a batch of data, return a Gaussian distribution over the latent space.

        Parameters:
        x: [torch.Tensor] 
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        """
        mean, std = torch.chunk(self.encoder_net(x), 2, dim=-1)
        return td.Independent(td.Normal(loc=mean, scale=torch.exp(std)), 1)

class BernoulliDecoder(nn.Module):
    def __init__(self, decoder_net):
        """
        Define a Bernoulli decoder distribution based on a given decoder network.

        Parameters: 
        encoder_net: [torch.nn.Module]             
           The decoder network that takes as a tensor of dim `(batch_size, M) as
           input, where M is the dimension of the latent space, and outputs a
           tensor of dimension (batch_size, feature_dim1, feature_dim2).
        """
        super(BernoulliDecoder, self).__init__()
        self.decoder_net = decoder_net
        self.std = nn.Parameter(torch.ones(28, 28)*0.5, requires_grad=True)

    def forward(self, z):
        """
        Given a batch of latent variables, return a Bernoulli distribution over the data space.

        Parameters:
        z: [torch.Tensor] 
           A tensor of dimension `(batch_size, M)`, where M is the dimension of the latent space.
        """
        logits = self.decoder_net(z)
        return td.Independent(td.Bernoulli(logits=logits), 2)

class VAE(nn.Module):
    """
    Define a Variational Autoencoder (VAE) model.
    """
    def __init__(self, prior, decoder, encoder, prior_type='gaussian'):
        """
        Parameters:
        prior: [torch.nn.Module] 
           The prior distribution over the latent space.
        decoder: [torch.nn.Module]
              The decoder distribution over the data space.
        encoder: [torch.nn.Module]
                The encoder distribution over the latent space.
        """
            
        super(VAE, self).__init__()
        self.prior = prior
        self.decoder = decoder
        self.encoder = encoder
        self.prior_type = prior_type

    def elbo(self, x):
        """
        Compute the ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor] 
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2, ...)`
           n_samples: [int]
           Number of samples to use for the Monte Carlo estimate of the ELBO.
        """

        q = self.encoder(x)
        z = q.rsample()

        if self.prior_type == 'gaussian':
            elbo = torch.mean(self.decoder(z).log_prob(x) - td.kl_divergence(q, self.prior()), dim=0)
        elif self.prior_type in ['mog', 'flow']:
            # KL divergence is not analytically tractable for MoG and Flow priors,
            # so we use a Monte Carlo estimate.
            log_p_z = self.prior.log_prob(z)
            log_q_z = q.log_prob(z)
            elbo = torch.mean(self.decoder(z).log_prob(x) + log_p_z - log_q_z, dim=0)
        return elbo

    def sample(self, n_samples=1):
        """
        Sample from the model.
        
        Parameters:
        n_samples: [int]
           Number of samples to generate.
        """
        z = self.prior.sample(torch.Size([n_samples]))
        return self.decoder(z).sample()
    
    def forward(self, x):
        """
        Compute the negative ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor] 
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        """
        return -self.elbo(x)

def train(model, optimizer, data_loader, epochs, device):
    """
    Train a VAE model.

    Parameters:
    model: [VAE]
       The VAE model to train.
    optimizer: [torch.optim.Optimizer]
         The optimizer to use for training.
    data_loader: [torch.utils.data.DataLoader]
            The data loader to use for training.
    epochs: [int]
        Number of epochs to train for.
    device: [torch.device]
        The device to use for training.
    """
    model.train()

    total_steps = len(data_loader)*epochs
    progress_bar = tqdm(range(total_steps), desc="Training")

    for epoch in range(epochs):
        data_iter = iter(data_loader)
        for x in data_iter:
            x = x[0].to(device)
            optimizer.zero_grad()
            loss = model(x)
            loss.backward()
            optimizer.step()

            # Update progress bar
            progress_bar.set_postfix(loss=f"⠀{loss.item():12.4f}", epoch=f"{epoch+1}/{epochs}")
            progress_bar.update()


def evaluate(model, data_loader, device):
    """
    Evaluate a VAE model.

    Parameters:
    model: [VAE]
       The VAE model to evaluate.
    data_loader: [torch.utils.data.DataLoader]
            The data loader to use for evaluation.
    device: [torch.device]
        The device to use for evaluation.
    """
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x in data_loader:
            x = x[0].to(device)
            loss = model(x)
            total_loss += loss.item() * x.size(0)

    return total_loss / len(data_loader.dataset)


def append_test_elbo_log(args, test_elbo, log_path=None):
    """
    Append one CSV row with all run arguments and test-set ELBO to Training_log.csv.
    """
    if log_path is None:
        log_path = Path(__file__).with_name("Training_log.csv")

    params = vars(args)
    param_keys = sorted(params.keys())
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = {"timestamp": timestamp, "test_elbo": f"{test_elbo:.6f}"}
    for key in param_keys:
        row[key] = params[key]

    fieldnames = ["timestamp", "test_elbo"] + param_keys
    write_header = (not log_path.exists()) or log_path.stat().st_size == 0
    with open(log_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    print(f"Appended test -ELBO and run parameters to CSV '{log_path}'")


def visualize_approx_posterior(model, data_loader, device, M, n_samples=10, plot_file=None):
    """
    Visualize the approximate posterior distribution of a VAE model.

    This function is super messy, and should be rewritten. It works tho, so I just use it.
    
    Parameters:
    model: VAE model
    data_loader: DataLoader with test data
    device: torch device
    M: latent dimension
    n_samples: number of random samples to visualize (default: 10)
    plot_file: optional output filename for the saved plot
    """
    prior_display_name = {
        'gaussian': 'Gaussian',
        'mog': 'MoG',
        'flow': 'Flow',
    }.get(getattr(model, 'prior_type', ''), type(model.prior).__name__)
    title = f'Prior and Aggregated Posterior {prior_display_name}'
    output_file = plot_file or f"approx_posterior_2D_{getattr(model, 'prior_type', 'prior')}.png"

    # Collect all data from data_loader
    all_x = []
    all_labels = []
    for x_batch, labels_batch in data_loader:
        all_x.append(x_batch)
        all_labels.append(labels_batch)
    
    # Concatenate all batches
    all_x = torch.cat(all_x, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # Randomly sample n_samples indices
    n_total = all_x.size(0)
    random_indices = torch.randperm(n_total)[:n_samples]
    
    # Get random samples
    x = all_x[random_indices].to(device)
    labels = all_labels[random_indices].cpu().numpy()
    
    # Encode and sample from posterior
    model.eval()
    with torch.no_grad():
        q = model.encoder(x)
        z = q.rsample().cpu().numpy()

    
    if isinstance(model.prior, FlowPrior):
        n_draw = z.shape[0]
        with torch.no_grad():
            z_prior = model.prior.sample(torch.Size([n_draw])).cpu().numpy()

        if M == 2:
            z_post_plot = z
            z_prior_plot = z_prior
            x_label = 'z1'
            y_label = 'z2'
        else:
            pooled_samples = np.concatenate([z, z_prior], axis=0)
            pooled_plot, pca = PCA_sklearn(pooled_samples, n_components=2, return_pca=True)
            z_post_plot = pooled_plot[:n_draw]
            z_prior_plot = pooled_plot[n_draw:]

            explained_var = pca.explained_variance_ratio_
            explained_var_2d = float(np.sum(explained_var[:2]))

            print(f"[FlowPrior PCA] explained variance ratio PC1: {explained_var[0]:.4f}")
            print(f"[FlowPrior PCA] explained variance ratio PC2: {explained_var[1]:.4f}")
            print(f"[FlowPrior PCA] cumulative explained variance (PC1+PC2): {explained_var_2d:.4f}")
            if explained_var_2d < 0.40:
                print("[FlowPrior PCA] Caveat: high information loss in 2D projection; rely mainly on quantitative metrics in full latent space.")
            elif explained_var_2d < 0.70:
                print("[FlowPrior PCA] Caveat: moderate information loss; interpret visual overlap with care.")
            else:
                print("[FlowPrior PCA] Info: 2D projection keeps a large share of pooled variance.")
            print("[FlowPrior PCA] Caveat: overlap in this 2D projection does not imply full-distribution match in latent space.")
            print("[FlowPrior PCA] Caveat: mismatches can remain in lower-variance directions not shown by PC1/PC2.")

            x_label = 'PC1'
            y_label = 'PC2'

        # Build shared 2D grid in projected space.
        combined_plot = np.concatenate([z_post_plot, z_prior_plot], axis=0)
        x_margin = 0.2 * (combined_plot[:, 0].max() - combined_plot[:, 0].min() + 1e-6)
        y_margin = 0.2 * (combined_plot[:, 1].max() - combined_plot[:, 1].min() + 1e-6)
        x_min, x_max = combined_plot[:, 0].min() - x_margin, combined_plot[:, 0].max() + x_margin
        y_min, y_max = combined_plot[:, 1].min() - y_margin, combined_plot[:, 1].max() + y_margin

        grid_n = 220
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, grid_n),
            np.linspace(y_min, y_max, grid_n),
        )
        grid_points = np.stack([xx.ravel(), yy.ravel()], axis=-1)

        # Silverman-style bandwidth for stable KDE in 2D.
        std_xy = np.std(combined_plot, axis=0, ddof=1)
        sigma = float(np.mean(std_xy))
        if not np.isfinite(sigma) or sigma <= 1e-8:
            sigma = 1.0
        bandwidth = ((combined_plot.shape[0] * (2 + 2) / 4.0) ** (-1.0 / (2 + 4))) * sigma
        bandwidth = float(max(bandwidth, 1e-3))
        print(f"[FlowPrior PCA] KDE bandwidth in projected space: {bandwidth:.4f}")

        # Estimate projected marginal pi#p on a shared grid.
        kde_prior = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(z_prior_plot)
        prior_pdf = np.exp(kde_prior.score_samples(grid_points)).reshape(grid_n, grid_n)

        positive_prior = prior_pdf[prior_pdf > 0]
        prior_levels = None
        if positive_prior.size == 0:
            print("[FlowPrior PCA] Warning: KDE produced degenerate values; density plot may be uninformative.")
        else:
            prior_levels = np.quantile(positive_prior, [0.70, 0.85, 0.95])
            prior_levels = np.unique(prior_levels)
            if prior_levels.size < 2:
                max_val = positive_prior.max()
                prior_levels = np.array([0.2 * max_val, 0.5 * max_val, 0.8 * max_val])

        fig, ax = plt.subplots(figsize=(8, 6))
        if prior_levels is not None:
            ax.contour(
                xx,
                yy,
                prior_pdf,
                levels=prior_levels.tolist(),
                colors='k',
                linestyles='--',
                linewidths=1.2,
                alpha=0.95,
                zorder=3,
            )

        # Light sample overlays for support intuition.
        ax.scatter(
            z_prior_plot[:, 0],
            z_prior_plot[:, 1],
            c='black',
            s=6,
            alpha=0.12,
            zorder=1,
        )
        scatter = ax.scatter(
            z_post_plot[:, 0],
            z_post_plot[:, 1],
            c=labels,
            cmap='tab10',
            alpha=0.7,
            zorder=0,
        )
        plt.colorbar(scatter, ticks=range(10), ax=ax)

        if prior_levels is not None:
            ax.legend(
                [Line2D([0], [0], color='k', linestyle='--', linewidth=1.2)],
                ['Prior PDF'],
                loc='best',
                fontsize=8,
                framealpha=0.9,
            )

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.grid()
        plt.savefig(output_file)
        print(f"Saved approximate posterior visualization as '{output_file}'")
        #plt.show()
        return

    if M == 2:
        z_plot = z
        projection_matrix = np.eye(2, dtype=np.float64)
        projection_center = np.zeros(2, dtype=np.float64)
    else:
        z_plot, pca = PCA_sklearn(z, n_components=2, return_pca=True)
        projection_matrix = pca.components_
        projection_center = pca.mean_

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(z_plot[:, 0], z_plot[:, 1], c=labels, cmap='tab10', alpha=0.7)
    plt.colorbar(scatter, ticks=range(10), ax=ax)

    # Gather prior components as (mean, full covariance, weight).
    prior_components = []
    if isinstance(model.prior, GaussianPrior):
        mean = model.prior.mean.detach().cpu().numpy()
        var = (model.prior.std.detach().cpu().numpy()) ** 2
        cov = np.diag(np.maximum(var, 1e-8))
        prior_components.append((mean, cov, 1.0))
    elif isinstance(model.prior, MoG_Prior):
        means = model.prior.means.detach().cpu().numpy()
        vars_ = np.exp(model.prior.logvars.detach().cpu().numpy())
        weights = torch.softmax(model.prior.w.squeeze(), dim=0).detach().cpu().numpy()
        for k in range(means.shape[0]):
            cov = np.diag(np.maximum(vars_[k], 1e-8))
            prior_components.append((means[k], cov, weights[k]))

    # Project prior components to 2D plotting space: mu' = P(mu-c), Sigma' = P Sigma P^T.
    projected_components = []
    for mean, cov, weight in prior_components:
        mean_2d = projection_matrix @ (mean - projection_center)
        cov_2d = projection_matrix @ cov @ projection_matrix.T
        cov_2d = cov_2d + 1e-8 * np.eye(2)
        projected_components.append((mean_2d, cov_2d, weight))

    # Build plotting window from posterior samples and projected prior means.
    if projected_components:
        prior_means = np.array([comp[0] for comp in projected_components])
        x_all = np.concatenate([z_plot[:, 0], prior_means[:, 0]])
        y_all = np.concatenate([z_plot[:, 1], prior_means[:, 1]])
    else:
        x_all = z_plot[:, 0]
        y_all = z_plot[:, 1]

    x_margin = 0.2 * (x_all.max() - x_all.min() + 1e-6)
    y_margin = 0.2 * (y_all.max() - y_all.min() + 1e-6)
    x_min, x_max = x_all.min() - x_margin, x_all.max() + x_margin
    y_min, y_max = y_all.min() - y_margin, y_all.max() + y_margin

    grid_n = 200
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_n),
        np.linspace(y_min, y_max, grid_n),
    )
    grid_points = np.stack([xx, yy], axis=-1)

    def gaussian_pdf_grid(points, mean, cov):
        inv_cov = np.linalg.inv(cov)
        det_cov = np.linalg.det(cov)
        det_cov = max(det_cov, 1e-12)
        centered = points - mean
        quad = np.einsum('...i,ij,...j->...', centered, inv_cov, centered)
        norm = 1.0 / (2.0 * np.pi * np.sqrt(det_cov))
        return norm * np.exp(-0.5 * quad)

    prior_pdf = np.zeros_like(xx, dtype=np.float64)
    for mean, cov, weight in projected_components:
        prior_pdf += weight * gaussian_pdf_grid(grid_points, mean, cov)

    if projected_components:
        positive_vals = prior_pdf[prior_pdf > 0]
        if positive_vals.size > 0:
            levels = np.quantile(positive_vals, [0.70, 0.85, 0.95])
            levels = np.unique(levels)
            if levels.size < 2:
                max_val = positive_vals.max()
                levels = np.array([0.2 * max_val, 0.5 * max_val, 0.8 * max_val])

            ax.contour(
                xx,
                yy,
                prior_pdf,
                levels=levels.tolist(),
                colors='k',
                linestyles='--',
                linewidths=1.2,
                alpha=0.95,
            )
            ax.legend(
                [Line2D([0], [0], color='k', linestyle='--', linewidth=1.2)],
                ['Prior PDF'],
                loc='best',
                fontsize=8,
                framealpha=0.9,
            )

    ax.set_title(title)
    ax.set_xlabel('z1')
    ax.set_ylabel('z2')
    ax.grid()
    plt.savefig(output_file)
    print(f"Saved approximate posterior visualization as '{output_file}'")
    #plt.show()

        
def PCA_sklearn(z, n_components=2, return_pca=False):
    """
    Perform PCA using scikit-learn.

    Parameters:
    z: numpy.ndarray of shape (n_samples, latent_dim)
    n_components: number of principal components

    Returns:
    z_pca: numpy.ndarray of shape (n_samples, n_components)
    pca: sklearn PCA object (optional if return_pca=True)
    """
    pca = PCA(n_components=n_components)
    z_pca = pca.fit_transform(z)
    if return_pca:
        return z_pca, pca
    return z_pca



if __name__ == "__main__":
    from torchvision import datasets, transforms
    from torchvision.utils import save_image, make_grid
    import glob

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', nargs='?', type=str, default='full', choices=['train', 'sample', 'evaluate', 'plot', 'full'], help='what to do when running the script (default: %(default)s)')
    parser.add_argument('--model', type=str, default='model.pt', help='file to save model to or load model from (default: %(default)s)')
    parser.add_argument('--samples', type=str, default='samples.png', help='file to save samples in (default: %(default)s)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'], help='torch device (default: %(default)s)')
    parser.add_argument('--cuda', action='store_true', help='shortcut for --device cuda')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N', help='batch size for training (default: %(default)s)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train (default: %(default)s)')
    parser.add_argument('--latent-dim', type=int, default=32, metavar='N', help='dimension of latent variable (default: %(default)s)')
    parser.add_argument('--prior', type=str, default='gaussian', choices=['gaussian', 'mog', 'flow'], help='type of prior distribution (default: %(default)s)')
    parser.add_argument('--plot-file', type=str, default=None, help='file to save posterior/prior plot in (default: auto by prior type)')
    parser.add_argument('--flow-num-transformations', type=int, default=10, metavar='N', help='number of coupling layers in flow prior (default: %(default)s)')
    parser.add_argument('--flow-hidden-dim', type=int, default=128, metavar='N', help='hidden dimension in flow prior coupling nets (default: %(default)s)')
    parser.add_argument('--train-subset-size', type=int, default=0, metavar='N', help='if > 0, train on a random subset of this size for fast prototyping (default: %(default)s)')

    args = parser.parse_args()
    print('# Options')
    for key, value in sorted(vars(args).items()):
        print(key, '=', value)

    requested_device = 'cuda' if args.cuda else args.device
    if requested_device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA requested but unavailable. This usually means the job is running on a non-GPU node. "
            "Submit the script to a GPU queue (LSF) and load the CUDA module first, or use --device cpu."
        )
    if requested_device == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable on this machine. Use --device cpu or --device cuda.")
    device = torch.device(requested_device)
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print("CUDA device:", torch.cuda.get_device_name(0))

    # Load MNIST as binarized at 'thresshold' and create data loaders
    thresshold = 0.5
    train_dataset = datasets.MNIST(
        'data/',
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: (thresshold < x).float().squeeze())
        ])
    )
    if args.train_subset_size > 0:
        subset_size = min(args.train_subset_size, len(train_dataset))
        generator = torch.Generator().manual_seed(0)
        subset_indices = torch.randperm(len(train_dataset), generator=generator)[:subset_size]
        train_dataset = torch.utils.data.Subset(train_dataset, subset_indices.tolist())
        print(f"Using train subset: {subset_size}/{60000} samples")

    mnist_train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )
    mnist_test_loader = torch.utils.data.DataLoader(datasets.MNIST('data/', train=False, download=True,
                                                                transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: (thresshold < x).float().squeeze())])),
                                                    batch_size=args.batch_size, shuffle=True)

    # Define prior distribution
    M = args.latent_dim
    prior_type = args.prior
    if prior_type == 'gaussian':
        prior = GaussianPrior(M)
    elif prior_type == 'mog':
        prior = MoG_Prior(M, K=10)
    elif prior_type == 'flow':
        prior = FlowPrior(
            M=M,
            num_transformations=args.flow_num_transformations,
            hidden_dim=args.flow_hidden_dim,
        )

    # Define encoder and decoder networks
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
        nn.Unflatten(-1, (28, 28))
    )

    # Define VAE model
    decoder = BernoulliDecoder(decoder_net)
    encoder = GaussianEncoder(encoder_net)
    model = VAE(prior, decoder, encoder,prior_type).to(device)

    # Choose mode to run
    if args.mode in ['train', 'full']:
        # Define optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Train model
        train(model, optimizer, mnist_train_loader, args.epochs, device)

        # Save model
        torch.save(model.state_dict(), args.model)

    if args.mode == 'sample':
        model.load_state_dict(torch.load(args.model, map_location=device))

        # Generate samples
        model.eval()
        with torch.no_grad():
            samples = (model.sample(64)).cpu() 
            save_image(samples.view(64, 1, 28, 28), args.samples)


    elif args.mode == 'evaluate':
        model.load_state_dict(torch.load(args.model, map_location=device))
        
        test_loss = evaluate(model, mnist_test_loader, device)
        print(f"Test -ELBO (loss): {test_loss:.4f}")
        append_test_elbo_log(args, test_loss)

    elif args.mode == 'plot':
        model.load_state_dict(torch.load(args.model, map_location=device))
        print("Visualizing approximate posterior...")
        visualize_approx_posterior(model, mnist_test_loader, device, M, n_samples=2000, plot_file=args.plot_file)

    elif args.mode == 'full':
        test_loss = evaluate(model, mnist_test_loader, device)
        print(f"Test -ELBO (loss): {test_loss:.4f}")
        append_test_elbo_log(args, test_loss)

        print("Visualizing approximate posterior...")
        visualize_approx_posterior(model, mnist_test_loader, device, M, n_samples=2000, plot_file=args.plot_file)
