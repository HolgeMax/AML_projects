# Code for DTU course 02460 (Advanced Machine Learning Spring) by Jes Frellsen, 2024
# Version 1.0 (2024-01-27)
# Inspiration is taken from:
# - https://github.com/jmtomczak/intro_dgm/blob/main/vaes/vae_example.ipynb
# - https://github.com/kampta/pytorch-dist_geodesicributions/blob/master/gaussian_vae.py
#
# Significant extension by Søren Hauberg, 2024

import torch
import torch.nn as nn
import torch.distributions as td
import torch.utils.data
from tqdm import tqdm
from copy import deepcopy
import os
import math
import matplotlib.pyplot as plt
import copy
import numpy as np
import random



class GaussianPrior(nn.Module):
    def __init__(self, M):
        """
        Define a Gaussian prior dist_geodesicribution with zero mean and unit variance.

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
        Return the prior dist_geodesicribution.

        Returns:
        prior: [torch.dist_geodesicributions.dist_geodesicribution]
        """
        return td.Independent(td.Normal(loc=self.mean, scale=self.std), 1)


class GaussianEncoder(nn.Module):
    def __init__(self, encoder_net):
        """
        Define a Gaussian encoder dist_geodesicribution based on a given encoder network.

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
        Given a batch of data, return a Gaussian dist_geodesicribution over the latent space.

        Parameters:
        x: [torch.Tensor]
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        """
        mean, std = torch.chunk(self.encoder_net(x), 2, dim=-1)
        return td.Independent(td.Normal(loc=mean, scale=torch.exp(std)), 1)


class GaussianDecoder(nn.Module):
    def __init__(self, decoder_net):
        """
        Define a Bernoulli decoder dist_geodesicribution based on a given decoder network.

        Parameters:
        encoder_net: [torch.nn.Module]
           The decoder network that takes as a tensor of dim `(batch_size, M) as
           input, where M is the dimension of the latent space, and outputs a
           tensor of dimension (batch_size, feature_dim1, feature_dim2).
        """
        super(GaussianDecoder, self).__init__()
        self.decoder_net = decoder_net
        # self.std = nn.Parameter(torch.ones(28, 28) * 0.5, requires_grad=True) # In case you want to learn the std of the gaussian.

    def forward(self, z):
        """
        Given a batch of latent variables, return a Bernoulli dist_geodesicribution over the data space.

        Parameters:
        z: [torch.Tensor]
           A tensor of dimension `(batch_size, M)`, where M is the dimension of the latent space.
        """
        means = self.decoder_net(z)
        return td.Independent(td.Normal(loc=means, scale=1e-1), 3)
    
class VAE(nn.Module):
    """
    Define an Ensemble Variational Autoencoder (VAE) model.
    """

    def __init__(self, prior, decoder, encoder, num_ensemble):
        """
        Parameters:
        prior: [torch.nn.Module]
           The prior dist_geodesicribution over the latent space.
        decoder: [torch.nn.Module]
              The decoder dist_geodesicribution over the data space.
        encoder: [torch.nn.Module]
                The encoder dist_geodesicribution over the latent space.
        """

        super(VAE, self).__init__()
        self.prior = prior
        self.decoders = nn.ModuleList([
            copy.deepcopy(decoder) for _ in range(num_ensemble)
        ])
        self.encoder = encoder

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

        log_prob_x = 0
        for decoder in self.decoders:
            log_prob_x += decoder(z).log_prob(x)

        elbo = torch.mean(
            log_prob_x - q.log_prob(z) + self.prior().log_prob(z)
        )
        return elbo

    def sample(self, n_samples=1):
        """
        Sample from the model.

        Parameters:
        n_samples: [int]
           Number of samples to generate.
        """
        z = self.prior().sample(torch.Size([n_samples]))
        return self.decoders(z).sample()

    def forward(self, x):
        """
        Compute the negative ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor]
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        """
        return -self.elbo(x)
    
    def energy(self, z):
        if len(self.decoders) == 1:    
            dist_geodesic = self.decoders[0](z)

            mu = dist_geodesic.mean
            mu = mu.view(mu.shape[0], -1)

            diffs = mu[1:] - mu[:-1] #Compute mu_i - mu_{i-1}
            squared_norms = torch.sum(diffs**2, dim=1) #Compute ||mu_i - mu_{i-1}||^2

            N = z.shape[0] - 1
            energy = N/(2*0.1**2)*torch.sum(squared_norms)
        else:
            M = len(self.decoders)

            idx_i = torch.randint(0, M-1, (1,)).item()
            idx_j = torch.randint(0, M-1, (1,)).item()

            dist_geodesic_i = self.decoders[idx_i](z)
            dist_geodesic_j = self.decoders[idx_j](z)

            mu_i = dist_geodesic_i.mean
            mu_i = mu_i.view(mu_i.shape[0], -1)
            mu_j = dist_geodesic_j.mean
            mu_j = mu_j.view(mu_j.shape[0], -1)

            diffs = mu_i[1:] - mu_j[:-1]
            squared_norms = torch.sum(diffs**2, dim=1)

            N = z.shape[0] - 1
            energy = N/(2*0.1**2)*torch.sum(squared_norms)

        return energy

    def geodesics(self, z_start, z_end, param="pwl", n_points=20, n_steps=500, lr=1e-2, sigma=0.1):
        device = z_start.device
        latent_dim = z_start[0]

        if (param == "pwl"):
            t = torch.linspace(0, 1, n_points, device = device).unsqueeze(1)
            z_path = (1 - t) * z_start + t * z_end

            z_learnable = nn.Parameter(z_path[1:-1].clone())
            optimizer = torch.optim.Adam([z_learnable], lr)
        elif (param == "NN"):
            net = nn.Sequential(
                nn.Linear(4, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, n_points*2)
            ).to(device)

            optimizer = torch.optim.Adam(net.parameters(), lr)

            z = torch.cat([z_start, z_end]).to(device)
    
        for step in range(n_steps):
            optimizer.zero_grad()

            if (param == "NN"):
                z_learnable = net(z).view(n_points, 2)

            z_full = torch.cat([
                z_start.view(1, -1),
                z_learnable,
                z_end.view(1, -1)
            ], dim=0)

            energy = self.energy(z_full)

            energy.backward()
            optimizer.step()
        
        z_final = torch.cat([
            z_start.view(1, -1),
            z_learnable.detach(),
            z_end.view(1, -1)
        ], dim=0)

        return z_final

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

    num_steps = len(data_loader) * epochs
    epoch = 0

    def noise(x, std=0.05):
        eps = std * torch.randn_like(x)
        return torch.clamp(x + eps, min=0.0, max=1.0)

    with tqdm(range(num_steps)) as pbar:
        for step in pbar:
            try:
                x = next(iter(data_loader))[0]
                x = noise(x.to(device))
                model = model
                optimizer.zero_grad()
                # from IPython import embed; embed()
                loss = model(x)
                loss.backward()
                optimizer.step()

                # Report
                if step % 5 == 0:
                    loss = loss.detach().cpu()
                    pbar.set_description(
                        f"total epochs ={epoch}, step={step}, loss={loss:.1f}"
                    )

                if (step + 1) % len(data_loader) == 0:
                    epoch += 1
            except KeyboardInterrupt:
                print(
                    f"Stopping training at total epoch {epoch} and current loss: {loss:.1f}"
                )
                break


if __name__ == "__main__":
    from torchvision import datasets, transforms
    from torchvision.utils import save_image

    # Parse arguments
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        type=str,
        default="train",
        choices=["train", "sample", "eval", "geodesics", "cov"],
        help="what to do when running the script (default: %(default)s)",
    )
    parser.add_argument(
        "--experiment-folder",
        type=str,
        default="experiment",
        help="folder to save and load experiment results in (default: %(default)s)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="model.pt",
        help="file to save model in (default: %(default)s)",
    )
    parser.add_argument(
        "--samples",
        type=str,
        default="samples.png",
        help="file to save samples in (default: %(default)s)",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="torch device (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        metavar="N",
        help="batch size for training (default: %(default)s)",
    )
    parser.add_argument(
        "--epochs-per-decoder",
        type=int,
        default=50,
        metavar="N",
        help="number of training epochs per each decoder (default: %(default)s)",
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=2,
        metavar="N",
        help="dimension of latent variable (default: %(default)s)",
    )
    parser.add_argument(
        "--num-decoders",
        type=int,
        default=3,
        metavar="N",
        help="number of decoders in the ensemble (default: %(default)s)",
    )
    parser.add_argument(
        "--num-reruns",
        type=int,
        default=10,
        metavar="N",
        help="number of reruns (default: %(default)s)",
    )
    parser.add_argument(
        "--num-curves",
        type=int,
        default=10,
        metavar="N",
        help="number of geodesics to plot (default: %(default)s)",
    )
    parser.add_argument(
        "--num-t",  # number of points along the curve
        type=int,
        default=20,
        metavar="N",
        help="number of points along the curve (default: %(default)s)",
    )
    parser.add_argument(
        "--curve-param",  # type of cureve parametrication
        type=str,
        default="pwl",
        choices=["pwl", "NN"],
        help="Curve parametrication(default: %(default)s)",
    )

    args = parser.parse_args()
    print("# Options")
    for key, value in sorted(vars(args).items()):
        print(key, "=", value)

    device = args.device

    # Load a subset of MNIST and create data loaders
    def subsample(data, targets, num_data, num_classes):
        idx = targets < num_classes
        new_data = data[idx][:num_data].unsqueeze(1).to(torch.float32) / 255
        new_targets = targets[idx][:num_data]

        return torch.utils.data.TensorDataset(new_data, new_targets)

    num_train_data = 2048
    num_classes = 3
    train_tensors = datasets.MNIST(
        "data/",
        train=True,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )
    test_tensors = datasets.MNIST(
        "data/",
        train=False,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )
    train_data = subsample(
        train_tensors.data, train_tensors.targets, num_train_data, num_classes
    )
    test_data = subsample(
        test_tensors.data, test_tensors.targets, num_train_data, num_classes
    )

    mnist_train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True
    )
    mnist_test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False
    )

    # Define prior dist_geodesicribution
    M = args.latent_dim

    def new_encoder():
        encoder_net = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.Softmax(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.Softmax(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.Flatten(),
            nn.Linear(512, 2 * M),
        )
        return encoder_net

    def new_decoder():
        decoder_net = nn.Sequential(
            nn.Linear(M, 512),
            nn.Unflatten(-1, (32, 4, 4)),
            nn.Softmax(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1, output_padding=0),
            nn.Softmax(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.Softmax(),
            nn.BatchNorm2d(16),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
        )
        return decoder_net

    pair_rng = np.random.default_rng(seed=42) 
    num_test_images = len(test_data)
    FIXED_PAIRS = pair_rng.integers(0, num_test_images, size=(args.num_curves, 2))

    # Choose mode to run
    if args.mode == "train":

        experiments_folder = args.experiment_folder
        os.makedirs(f"{experiments_folder}", exist_ok=True)

        for m in range(args.num_reruns):
            model = VAE(
                GaussianPrior(M),
                GaussianDecoder(new_decoder()),
                GaussianEncoder(new_encoder()),
                num_ensemble=args.num_decoders,
            ).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            train(
                model,
                optimizer,
                mnist_train_loader,
                args.epochs_per_decoder,
                args.device,
            )
            os.makedirs(f"{experiments_folder}", exist_ok=True)

            torch.save(
                model.state_dict(),
                f"{experiments_folder}/run{m}_{args.model}",
            )

    elif args.mode == "sample":
        model = VAE(
            GaussianPrior(M),
            GaussianDecoder(new_decoder()),
            GaussianEncoder(new_encoder()),
            num_ensemble=args.num_decoders,
        ).to(device)
        model.load_state_dict(torch.load(args.experiment_folder + args.model))
        model.eval()

        with torch.no_grad():
            samples = (model.sample(64)).cpu()
            save_image(samples.view(64, 1, 28, 28), args.samples)

            data = next(iter(mnist_test_loader))[0].to(device)
            recon = model.decoder(model.encoder(data).mean).mean
            save_image(
                torch.cat([data.cpu(), recon.cpu()], dim=0), "reconstruction_means.png"
            )

    elif args.mode == "eval":
        # Load trained model
        model = VAE(
            GaussianPrior(M),
            GaussianDecoder(new_decoder()),
            GaussianEncoder(new_encoder()),
            num_ensemble=args.num_decoders,
        ).to(device)
        model.load_state_dict(torch.load(args.experiment_folder + args.model))
        model.eval()

        elbos = []
        with torch.no_grad():
            for x, y in mnist_test_loader:
                x = x.to(device)
                elbo = model.elbo(x)
                elbos.append(elbo)
        mean_elbo = torch.tensor(elbos).mean()
        print("Print mean test elbo:", mean_elbo)

    elif args.mode == "geodesics":
        model = VAE(
            GaussianPrior(M),
            GaussianDecoder(new_decoder()),
            GaussianEncoder(new_encoder()),
            num_ensemble=args.num_decoders,
        ).to(device)
        model.load_state_dict(torch.load(args.experiment_folder + "/"+ args.model))
        model.eval()

        samples_z = []
        samples_y = []

        with torch.no_grad():
            for x, y in mnist_test_loader:
                x = x.to(device)
                z = model.encoder(x).sample()
                samples_z.append(z)
                samples_y.append(y)

            samples_z = torch.cat(samples_z, dim=0).numpy()
            samples_y = torch.cat(samples_y, dim=0).numpy()

        fig = plt.figure()
        ax = fig.add_subplot(111)
        labels = [f"Digit {int(c)}" for c in np.unique(samples_y)]
        for i, label in enumerate(labels):
            idx = samples_y == i
            ax.scatter(
                samples_z[idx, 0],
                samples_z[idx, 1],
                label=label
            )

        ax.legend()
        ax.set_xlabel('z1')
        ax.set_ylabel('z2')
        n_geodesics = args.num_curves

        for i in range(n_geodesics):
            idx_start, idx_end = FIXED_PAIRS[i]

            z_start = torch.tensor(samples_z[idx_start])
            z_end = torch.tensor(samples_z[idx_end])

            geodesic = model.geodesics(z_start, z_end, param = args.curve_param, n_points = args.num_t)

            ax.plot(geodesic[:,0], geodesic[:,1], linewidth = 2, color = 'black')

        if (len(model.decoders) == 1):
            plt.title('Pull-back geodesics with standard VAE')
        else:
            plt.title(f'Pull-back geodesics with ensemble of {len(model.decoders)} decoders')
        plt.legend()
        plt.savefig(args.samples)
    
    elif args.mode == "cov":
        def load_models(num_decoders, num_reruns, device="cpu"):
            model_list = []
            for i in range(num_reruns):
                model = VAE(
                    GaussianPrior(M),
                    GaussianDecoder(new_decoder()),
                    GaussianEncoder(new_encoder()),
                    num_ensemble=num_decoders,
                ).to(device)

                path = f"models/ensemble_decoders{num_decoders}/run{i}_{args.model}"

                model.load_state_dict(torch.load(path))
                model.eval()

                model_list.append(model)
            return model_list
    
        def CoV(Y_i, Y_j, models):
            geodesicCoV = []
            euclideanCoV = []
            for i in range(len(Y_i)):
                dist_geodesic = []
                dist_euclidean = []
                for model in models:
                    with torch.no_grad():
                        X_i, X_j = model.encoder(Y_i[i]).sample(), model.encoder(Y_j[i]).sample()

                    #Compute geodesic length in parameter space
                    curve = model.geodesics(X_i, X_j)
                    
                    all_mu = torch.stack([dec(curve).mean.view(curve.shape[0], -1) for dec in model.decoders])
                    geodesic = torch.mean(all_mu, dim=0)
                    delta = (geodesic[1:] - geodesic[:-1])
                    d = torch.sum(delta**2, dim=1).sqrt().sum()
                    dist_geodesic.append(d)

                    #Compute euclidean distance in latent space
                    dist_euclidean.append(torch.sum((X_i-X_j)**2).sqrt())
                #Compute CoV for point pair    
                mean = sum(dist_geodesic) / len(dist_geodesic)
                std = (sum((i-mean)**2 for i in dist_geodesic) / len(dist_geodesic))**0.5
                geodesicCoV.append(std/mean)

                mean = sum(dist_euclidean) / len(dist_euclidean)
                std = (sum((i-mean)**2 for i in dist_euclidean) / len(dist_euclidean))**0.5
                euclideanCoV.append(std/mean)


            ## Compute average CoV  
            avg_geodesic_CoV = sum(geodesicCoV) / len(geodesicCoV)
            avg_euclidean_CoV = sum(euclideanCoV) / len(euclideanCoV)
            return avg_geodesic_CoV, avg_euclidean_CoV

        ## Get list of point pairs
        Y_i = []
        Y_j = []
        for i in range(10):
            idx_start, idx_end = FIXED_PAIRS[i]
            y_i, _ = mnist_test_loader.dataset[idx_start]
            y_j, _ = mnist_test_loader.dataset[idx_end]

            y_i = y_i.unsqueeze(0).to(device)
            y_j = y_j.unsqueeze(0).to(device)
            
            Y_i.append(y_i)
            Y_j.append(y_j)
        

        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i in range(1,4):
            models = load_models(i, 10)
            geo_CoV, euc_CoV = CoV(Y_i, Y_j, models)

            ax.plot(i, geo_CoV.detach().numpy(), 'o', label="Geodesic")
            ax.plot(i, euc_CoV.detach().numpy(), 'o', label="Euclidean")
        
        ax.set_xlabel('Number of decoders')
        ax.set_ylabel('CoV')
        plt.legend()
        plt.savefig('CoV_plot.png')







