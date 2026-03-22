import torch

def curve_energy(curve_points, decoder, device):


    curve_points = curve_points.to(device)

    f_curve = decoder(curve_points).mean
    f_curve = f_curve.view(len(curve_points), -1) # flatten

    # find velocity
    v = f_curve[1:]- f_curve[:-1]

    # Energy
    energy = (v**2).sum()
    return energy


def compute_geodesic(decoder,
                     z_start,
                     z_end,
                     n_points=20,
                     n_steps=200,
                     device="cpu"):
    z_start = z_start.to(device).detach()
    z_end = z_end.to(device).detach()

    t = torch.linspace(0, 1, n_points, device=device)
    z_init = torch.stack([
    (1-ti)*z_start + ti*z_end for ti in t
    ])

    interior = z_init[1:-1].clone().detach().requires_grad_(True)

    optimizer = torch.optim.LBFGS(
        [interior],
        lr=1.0,
        max_iter=20,
        line_search_fn='strong_wolfe'
    )

    def closure():
        optimizer.zero_grad()

        curve = torch.cat([
            z_start.unsqueeze(0),
            interior,
            z_end.unsqueeze(0)
        ],dim=0)

        loss = curve_energy(curve, decoder, device)
        loss.backward()
        return loss

    for _ in range(n_steps):
        optimizer.step(closure)

        # return optimized curve
        with torch.no_grad():
            curve = torch.cat([
                z_start.unsqueeze(0),
                interior,
                z_end.unsqueeze(0)
            ],dim=0)

        return curve.detach()

def decode_geodesics(geo, decoder, n_samples=5, device="cpu"):
    """
    Decode evenly spaced points along a geodesic curve.

    Selects n_samples interior points plus the two endpoints (n_samples+2 total),
    decodes each through the decoder, and returns the decoded images.

    Parameters:
    geo:       (n_points, M) tensor of latent points along the geodesic
    decoder:   the VAE decoder module
    n_samples: number of interior points to sample between endpoints (default: 5)
    device:    torch device

    Returns:
    decoded: tensor of shape (n_samples+2, ...) with decoded images
    """
    n = len(geo)
    interior_idx = torch.linspace(1, n - 2, n_samples).long()
    selected = torch.cat([
        geo[0:1],
        geo[interior_idx],
        geo[n - 1:n]
    ], dim=0).to(device)

    with torch.no_grad():
        decoded = decoder(selected).mean
    return decoded


# part B
def geodesic_length(curve_points, decoder, device):

    with torch.no_grad():
        f_curve = decoder(curve_points.to(device)).mean
        f_curve = f_curve.view(len(curve_points), -1)
        v = f_curve[1:]-f_curve[:-1]
        lenght = v.norm(dim=1).sum()
    return lenght.item()

