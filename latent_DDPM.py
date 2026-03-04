import torch
import torch.nn as nn
import torch.distributions as td
import torch.nn.functional as F
from tqdm import tqdm
from unet import Unet
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
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
from fid import compute_fid

class GaussianDecoder(nn.Module):
    def __init__(self, net, std=0.1):
        super().__init__()
        self.net = net
        self.std = std

    def forward(self, z):
        mu = self.net(z)
        return td.Independent(td.Normal(mu, self.std), 1)

class Beta_VAE(nn.Module):
    """
    Define a Variational Autoencoder (VAE) model.
    """
    def __init__(self, prior, decoder, encoder, beta, prior_type='gaussian'):
        """
        Parameters:
        prior: [torch.nn.Module] 
           The prior distribution over the latent space.
        decoder: [torch.nn.Module]
              The decoder distribution over the data space. 
        encoder: [torch.nn.Module]
                The encoder distribution over the latent space.
        """
            
        super(Beta_VAE, self).__init__()
        self.prior = prior
        self.decoder = decoder
        self.encoder = encoder
        self.prior_type = prior_type
        self.beta = beta

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
            elbo = torch.mean(self.decoder(z).log_prob(x) - self.beta * td.kl_divergence(q, self.prior()), dim=0)
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
    

def train(vae, ddpm, optimizer_vae, optimizer_ddpm, data_loader, 
          vae_epochs, ddpm_epochs, device):
    
    # ===== Stage 1: Train VAE =====
    vae.train()
    for epoch in range(vae_epochs):
        for batch in tqdm(data_loader, desc=f"VAE Epoch {epoch+1}/{vae_epochs}"):
            x = batch[0].to(device)  # flattened input
            optimizer_vae.zero_grad()
            loss = vae(x)
            loss.backward()
            optimizer_vae.step()
    
    # Freeze VAE for DDPM
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False
    
    # ===== Stage 2: Train latent DDPM =====
    ddpm.train()
    for epoch in range(ddpm_epochs):
        for batch in tqdm(data_loader, desc=f"DDPM Epoch {epoch+1}/{ddpm_epochs}"):
            x = batch[0].to(device)
            
            # Sample latent z from VAE encoder
            with torch.no_grad():
                q = vae.encoder(x)
                z0 = q.rsample()   # [batch, latent_dim]
            
            optimizer_ddpm.zero_grad()
            loss = ddpm.loss(z0)
            loss.backward()
            optimizer_ddpm.step()
    


if __name__ == "__main__":
    import torch.utils.data
    from torchvision import datasets, transforms
    from torchvision.utils import save_image
    import time
    #import ToyData

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, default='train', choices=['train', 'sample', 'test'], help='what to do when running the script (default: %(default)s)')
    parser.add_argument('--data', type=str, default='tg', choices=['tg', 'cb', 'mnist'], help='dataset to use {tg: two Gaussians, cb: chequerboard} (default: %(default)s)')
    parser.add_argument('--vae', type=str, default='vae.pt', help='file to save model to or load model from (default: %(default)s)')
    parser.add_argument('--ddpm', type=str, default='ddpm.pt', help='file to save model to or load model from (default: %(default)s)')
    parser.add_argument('--samples', type=str, default='samples.png', help='file to save samples in (default: %(default)s)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'], help='torch device (default: %(default)s)')
    parser.add_argument('--batch-size', type=int, default=10000, metavar='N', help='batch size for training (default: %(default)s)')
    parser.add_argument('--vae_epochs', type=int, default=15, metavar='N', help='number of epochs to train (default: %(default)s)')
    parser.add_argument('--ddpm_epochs', type=int, default=100, metavar='N', help='number of epochs to train (default: %(default)s)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='V', help='learning rate for training (default: %(default)s)')
    parser.add_argument('--beta', type=float, default=1e-6, metavar='V', help='Beta for training (default: %(default)s)')

    args = parser.parse_args()
    print('# Options')
    for key, value in sorted(vars(args).items()):
        print(key, '=', value)

    if args.data == 'mnist':
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
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    else:
        # Generate the data
        n_data = 10000000
        toy = {'tg': ToyData.TwoGaussians, 'cb': ToyData.Chequerboard}[args.data]()
        transform = lambda x: (x-0.5)*2.0
        train_loader = torch.utils.data.DataLoader(transform(toy().sample((n_data,))), batch_size=args.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(transform(toy().sample((n_data,))), batch_size=args.batch_size, shuffle=True)

    # Get the dimension of the dataset
    D = next(iter(train_loader))[0].shape[1]        
    M=10
    # Define the network
    num_hidden = 256
    network = FcNetwork(M, num_hidden)
    #network = Unet()

    # Set the number of steps in the diffusion process
    T = 1000

    # Define model
    ddpm = DDPM(network, T=T).to(args.device)
    
    
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
    vae = Beta_VAE(prior, decoder, encoder, beta=args.beta ,prior_type='gaussian').to(args.device)
    

    # Choose mode to run
    if args.mode == 'train':
        # Define optimizer
        optimizer_vae = torch.optim.Adam(vae.parameters(), lr=args.lr)
        optimizer_ddpm = torch.optim.Adam(ddpm.parameters(), lr=args.lr)

        # Train model
        train(vae, ddpm, optimizer_vae, optimizer_ddpm, train_loader, vae_epochs=args.vae_epochs, ddpm_epochs=args.ddpm_epochs, device=args.device)

        # Save model
        torch.save(vae.state_dict(), args.vae)
        torch.save(ddpm.state_dict(), args.ddpm)

    elif args.mode == 'sample':
        import matplotlib.pyplot as plt
        import numpy as np

        # Load the model
        ddpm.load_state_dict(torch.load(args.ddpm, map_location=torch.device(args.device)))
        vae.load_state_dict(torch.load(args.vae, map_location=torch.device(args.device)))

        ddpm.to(args.device)
        vae.to(args.device)

        # Generate samples
        ddpm.eval()
        vae.eval()

        with torch.no_grad():
            z_samples = (ddpm.sample((args.batch_size,10))).to(args.device) 
            samples = vae.decoder(z_samples)
        
        start = time.time()

        with torch.no_grad():
            z_samples = (ddpm.sample((args.batch_size,10))).to(args.device) 
            samples = vae.decoder(z_samples)
        
        end = time.time()
            
        samples = samples.mean
        samples = samples.view(args.batch_size, 1, 28, 28)
        # Compute FID
        x_real = next(iter(train_loader))[0].view(args.batch_size, 1, 28, 28)
        x_gen = samples
        fid = compute_fid(x_real, x_gen)
        print('FID:', fid)
        print(f"Sampling time: {end - start:.4f} seconds")

        # Transform the samples back to the original space
        samples = samples /2 + 0.5

        

        # Plot MNIST samples
        fig, ax = plt.subplots(1,4, figsize=(7,5))

        for i in range(4):
            img = samples[i].view(28, 28)   # reshape
            ax[i].imshow(img, cmap='gray')
            ax[i].axis('off')

        plt.tight_layout()
        
        plt.savefig(args.samples)
        plt.close()