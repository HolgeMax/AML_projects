# Code for DTU course 02460 (Advanced Machine Learning Spring) by Jes Frellsen, 2024
# Version 1.0 (2024-02-11)

import torch
import torch.nn as nn
import torch.distributions as td
import torch.nn.functional as F
from tqdm import tqdm
from unet import Unet
from fid import compute_fid


class DDPM(nn.Module):
    def __init__(self, network, beta_1=1e-4, beta_T=2e-2, T=100):
        """
        Initialize a DDPM model.

        Parameters:
        network: [nn.Module]
            The network to use for the diffusion process.
        beta_1: [float]
            The noise at the first step of the diffusion process.
        beta_T: [float]
            The noise at the last step of the diffusion process.
        T: [int]
            The number of steps in the diffusion process.
        """
        super(DDPM, self).__init__()
        self.network = network
        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.T = T

        self.beta = nn.Parameter(torch.linspace(beta_1, beta_T, T), requires_grad=False)
        self.alpha = nn.Parameter(1 - self.beta, requires_grad=False)
        self.alpha_cumprod = nn.Parameter(self.alpha.cumprod(dim=0), requires_grad=False)
    
    def negative_elbo(self, x):
        """
        Evaluate the DDPM negative ELBO on a batch of data.

        Parameters:
        x: [torch.Tensor]
            A batch of data (x) of dimension `(batch_size, *)`.
        Returns:
        [torch.Tensor]
            The negative ELBO of the batch of dimension `(batch_size,)`.
        """

        ### Implement Algorithm 1 here ###
        
        t = torch.randint(low=1, high=self.T, size=(x.shape[0],), device=x.device)
        t = t[:, None]
        epsilon = torch.randn_like(x)
        alpha_bar_t = self.alpha_cumprod[t]
        alpha_bar_t = alpha_bar_t.view(x.shape[0], 1)
        
        x_t = torch.sqrt(alpha_bar_t)*x + torch.sqrt(1- alpha_bar_t)*epsilon
        neg_elbo = ((epsilon - self.network(x_t, t))**2)     

        return neg_elbo

    def sample(self, shape):
        """
        Sample from the model.

        Parameters:
        shape: [tuple]
            The shape of the samples to generate.
        Returns:
        [torch.Tensor]
            The generated samples.
        """
        # Sample x_t for t=T (i.e., Gaussian noise)
        x_t = torch.randn(shape).to(self.alpha.device)

        # Sample x_t given x_{t+1} until x_0 is sampled
        for t in range(self.T-1, -1, -1):
            ### Implement the remaining of Algorithm 2 here ###
            alpha_t = self.alpha[t]
            sqrt_alpha_t = torch.sqrt(self.alpha[t])
            sqrt_beta_t = torch.sqrt(self.beta[t])
            sqrt_one_minus_alpha_bar_t = torch.sqrt(1-self.alpha_cumprod[t])

            if t > 1:
                z = torch.randn_like(x_t)
            else: z = torch.zeros_like(x_t)

            x_t = 1/sqrt_alpha_t * (x_t - (1-alpha_t)/(sqrt_one_minus_alpha_bar_t)*self.network(x_t, torch.full((shape[0],1), t, device = self.alpha.device))) + sqrt_beta_t*z

        return x_t

    def loss(self, x):
        """
        Evaluate the DDPM loss on a batch of data.

        Parameters:
        x: [torch.Tensor]
            A batch of data (x) of dimension `(batch_size, *)`.
        Returns:
        [torch.Tensor]
            The loss for the batch.
        """
        return self.negative_elbo(x).mean()


def train(vae, ddpm, optimizer, data_loader, epochs, device):
    """
    Train a Flow model.

    Parameters:
    model: [Flow]
       The model to train.
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
            if isinstance(x, (list, tuple)):
                x = x[0]
            x = x.to(device)
            optimizer.zero_grad()
            loss = model.loss(x)
            loss.backward()
            optimizer.step()

            # Update progress bar
            progress_bar.set_postfix(loss=f"⠀{loss.item():12.4f}", epoch=f"{epoch+1}/{epochs}")
            progress_bar.update()


class FcNetwork(nn.Module):
    def __init__(self, input_dim, num_hidden):
        """
        Initialize a fully connected network for the DDPM, where the forward function also take time as an argument.
        
        parameters:
        input_dim: [int]
            The dimension of the input data.
        num_hidden: [int]
            The number of hidden units in the network.
        """
        super(FcNetwork, self).__init__()
        self.network = nn.Sequential(nn.Linear(input_dim+1, num_hidden), nn.ReLU(), 
                                     nn.Linear(num_hidden, num_hidden), nn.ReLU(), 
                                     nn.Linear(num_hidden, input_dim))

    def forward(self, x, t):
        """"
        Forward function for the network.
        
        parameters:
        x: [torch.Tensor]
            The input data of dimension `(batch_size, input_dim)`
        t: [torch.Tensor]
            The time steps to use for the forward pass of dimension `(batch_size, 1)`
        """
        x_t_cat = torch.cat([x, t], dim=1)
        return self.network(x_t_cat)


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
    parser.add_argument('--model', type=str, default='model.pt', help='file to save model to or load model from (default: %(default)s)')
    parser.add_argument('--samples', type=str, default='samples.png', help='file to save samples in (default: %(default)s)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'], help='torch device (default: %(default)s)')
    parser.add_argument('--batch-size', type=int, default=10000, metavar='N', help='batch size for training (default: %(default)s)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N', help='number of epochs to train (default: %(default)s)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='V', help='learning rate for training (default: %(default)s)')

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

    # Define the network
    num_hidden = 64
    #network = FcNetwork(D, num_hidden)
    network = Unet()

    # Set the number of steps in the diffusion process
    T = 1000

    # Define model
    model = DDPM(network, T=T).to(args.device)

    # Choose mode to run
    if args.mode == 'train':
        # Define optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        # Train model
        train(model, optimizer, train_loader, args.epochs, args.device)

        # Save model
        torch.save(model.state_dict(), args.model)

    elif args.mode == 'sample':
        import matplotlib.pyplot as plt
        import numpy as np

        # Load the model
        model.load_state_dict(torch.load(args.model, map_location=torch.device(args.device)))

        # Generate samples
        model.eval()

        # Warm-up (important!)
        # with torch.no_grad():
        #     _ = model.sample((args.batch_size, D))

        start = time.time()

        with torch.no_grad():
            samples = (model.sample((args.batch_size,D))).cpu() 
        end = time.time()

        x_real = next(iter(train_loader))[0].view(args.batch_size, 1, 28, 28)
        x_gen = samples.view(args.batch_size, 1, 28, 28)
        fid = compute_fid(x_real, x_gen)
        print('FID:', fid)
        print(f"Sampling time: {end - start:.4f} seconds")

        # Transform the samples back to the original space
        samples = samples /2 + 0.5

        # Plot MNIST samples
        num_samples = 6
        fig, ax = plt.subplots(1, num_samples, figsize=(num_samples*1.5, 1.5))  # adjust width per image

        for i in range(num_samples):
            img = samples[i].view(28, 28)   # reshape
            ax[i].imshow(img, cmap='gray')
            ax[i].axis('off')

        # Remove extra padding around subplots
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.05, hspace=0.05)

        # Save figure tightly
        plt.savefig(args.samples, bbox_inches='tight', pad_inches=0)
        plt.close()
        
        

        # Plot the density of the toy data and the model samples
        # coordinates = [[[x,y] for x in np.linspace(*toy.xlim, 1000)] for y in np.linspace(*toy.ylim, 1000)]
        # prob = torch.exp(toy().log_prob(torch.tensor(coordinates)))

        # fig, ax = plt.subplots(1, 1, figsize=(7, 5))
        # im = ax.imshow(prob, extent=[toy.xlim[0], toy.xlim[1], toy.ylim[0], toy.ylim[1]], origin='lower', cmap='YlOrRd')
        # ax.scatter(samples[:, 0], samples[:, 1], s=1, c='black', alpha=0.5)
        # ax.set_xlim(toy.xlim)
        # ax.set_ylim(toy.ylim)
        # ax.set_aspect('equal')
        # fig.colorbar(im)
        # plt.savefig(args.samples)
        # plt.close()
