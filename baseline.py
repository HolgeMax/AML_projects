import torch
from torch.utils.data import random_split
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from torch_geometric.utils import to_networkx
from torch_geometric.utils import to_dense_adj, to_dense_batch

device = 'cpu'

# Load the MUTAG dataset
# Load data
dataset = TUDataset(root='./data/', name='MUTAG').to(device)
node_feature_dim = 7

# Split into training and validation
rng = torch.Generator().manual_seed(0)
train_dataset, validation_dataset, test_dataset = random_split(dataset, (100, 44, 44), generator=rng)

# Create dataloader for training and validation
train_loader = DataLoader(train_dataset, batch_size=100)
validation_loader = DataLoader(validation_dataset, batch_size=44)
test_loader = DataLoader(test_dataset, batch_size=44)

data = dataset[0]

class ErdosRenyi(torch.nn.Module):
    def __init__(self, dataset):
        super(ErdosRenyi, self).__init__()
        self.node_count = [data.num_nodes for data in dataset]
        self.num_node_features = dataset.num_node_features
        self.num_edge_features = dataset.num_edge_features
        self.N = int(np.random.choice(self.node_count))
        self.r = self.get_num_edges(self.N, dataset) / (self.N*(self.N-1)/2)

    def get_num_edges(self, N, dataset):
        edges_list = []
        for data in dataset:
            if data.num_nodes == N:
                edges_list.append(data.num_edges // 2)

        if len(edges_list) == 0:
            return 1  # fallback to avoid NaN

        return float(np.mean(edges_list))
    
    def forward(self):
        ## Assign edges at random with prob r
        A = torch.rand(self.N, self.N) < self.r
        A = torch.triu(A, diagonal=1)             ## Upper triangular to ensure undirected

        ## Add edges to edge_index attribute and ensure undirected graph
        edge_index = A.nonzero().t().contiguous()
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

        ## Add random node labels
        node_labels = torch.randint(0, self.num_node_features, (self.N,))
        x = torch.nn.functional.one_hot(node_labels, num_classes=self.num_node_features).float()

        ## Add random edge labels
        num_edges = edge_index.shape[1]
        edge_labels = torch.randint(0, self.num_edge_features, (num_edges,))
        edge_attr = torch.nn.functional.one_hot(edge_labels, num_classes=self.num_edge_features).float()

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
class GraphDiffusion(torch.nn.Module):
    def __init__(self, network, beta_1=1e-4, beta_T=2e-2, T=100):
        super(GraphDiffusion, self).__init__()
        self.network = network
        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.T = T

        self.beta = torch.nn.Parameter(torch.linspace(beta_1, beta_T, T), requires_grad=False)
        self.alpha = torch.nn.Parameter(1 - self.beta, requires_grad=False)
        self.alpha_cumprod = torch.nn.Parameter(self.alpha.cumprod(dim=0), requires_grad=False)

    def negative_elbo(self, x, edge_index, batch):

        # Convert to dense
        x_dense, mask = to_dense_batch(x, batch)   # [B, N, F]
        A_dense = to_dense_adj(edge_index, batch)  # [B, N, N]

        B, N, F = x_dense.shape
        device = x.device

        # Sample timestep per graph
        t = torch.randint(0, self.T, (B,), device=device)

        alpha_bar_t = self.alpha_cumprod[t].view(B, 1, 1)

        # Sample noise
        eps_x = torch.randn_like(x_dense)
        eps_A = torch.randn_like(A_dense)

        # Forward diffusion
        x_t = torch.sqrt(alpha_bar_t) * x_dense + torch.sqrt(1 - alpha_bar_t) * eps_x
        A_t = torch.sqrt(alpha_bar_t) * A_dense + torch.sqrt(1 - alpha_bar_t) * eps_A

        # Predict noise
        eps_x_pred, eps_A_pred = self.network(x_t, A_t, t, mask)

        # Masks
        node_mask = mask.unsqueeze(-1)              # [B, N, 1]
        adj_mask = node_mask * node_mask.transpose(1, 2)  # [B, N, N]

        # Loss
        loss_x = ((eps_x - eps_x_pred)**2 * node_mask).sum() / node_mask.sum()
        loss_A = ((eps_A - eps_A_pred)**2 * adj_mask).sum() / adj_mask.sum()

        return loss_x + loss_A
    
    def sample(self, num_nodes):
        B = 1
        N = num_nodes
        F = 7
        device = self.beta.device

        x_t = torch.randn(B, N, F, device=device)
        mask = torch.ones(B, N, dtype=torch.bool, device=device)
        A_t = torch.randn(B, N, N, device=device)

        for t in range(self.T-1, -1, -1):
            ### Implement the remaining of Algorithm 2 here ###
            t_batch = t_batch = torch.full((B,), t, device=device, dtype=torch.long)

            alpha_t = self.alpha[t]
            sqrt_alpha_t = torch.sqrt(self.alpha[t])
            sqrt_beta_t = torch.sqrt(self.beta[t])
            sqrt_one_minus_alpha_bar_t = torch.sqrt(1-self.alpha_cumprod[t])

            ## PRedict noise 
            eps_x_pred, eps_A_pred = self.network(x_t, A_t, t_batch, mask)

            ##DDPM step

            if t > 1:
                z_t = torch.randn_like(x_t)
                Z_t = torch.randn_like(A_t)
            else:
                z_t = torch.zeros_like(x_t)
                Z_t = torch.zeros_like(A_t)
            
            x_t = 1/sqrt_alpha_t * (x_t - (1-alpha_t)/sqrt_one_minus_alpha_bar_t*eps_x_pred) + sqrt_beta_t*z_t
            A_t = 1/sqrt_alpha_t * (A_t - (1-alpha_t)/sqrt_one_minus_alpha_bar_t*eps_A_pred) + sqrt_beta_t*Z_t

        # Node features → one-hot
        x_idx = torch.argmax(x_t, dim=-1)  # [B, N]
        x = torch.nn.functional.one_hot(x_idx, num_classes=F).float()[0]

        # Adjacency → binary
        A = torch.sigmoid(A_t)[0]
        A = (A > 0.5).float()

        # Symmetrize + remove self-loops
        A = (A + A.t()) / 2
        A = (A > 0.5).float()
        A.fill_diagonal_(0)

        # Convert to edge_index
        edge_index = A.nonzero(as_tuple=False).t().contiguous()

        return Data(x=x, edge_index=edge_index)

    def loss(self, x, edge_index, batch):
        return self.negative_elbo(x, edge_index, batch).mean()

class GraphConvDenoiser(torch.nn.Module):
    def __init__(self, node_feature_dim, filter_length, hidden_dim, T):
        super().__init__()

        self.node_feature_dim = node_feature_dim
        self.filter_length = filter_length
        self.hidden_dim = hidden_dim

        self.h = torch.nn.Parameter(1e-5*torch.randn(filter_length))
        self.h.data[0] = 1.

        # Node encoder
        self.input_proj = torch.nn.Linear(node_feature_dim, hidden_dim)

        # Time embedding
        self.time_emb = torch.nn.Embedding(T, hidden_dim)

        # Node output (predict noise)
        self.node_out = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, 2*hidden_dim),
            torch.nn.ReLU(),
            # torch.nn.Linear(2*hidden_dim, 2*hidden_dim),
            # torch.nn.ReLU(),
            # torch.nn.Linear(4*hidden_dim, 4*hidden_dim),
            # torch.nn.ReLU(),
            # torch.nn.Linear(4*hidden_dim, 2*hidden_dim),
            # torch.nn.ReLU(),
            torch.nn.Linear(2*hidden_dim, node_feature_dim)
        )

        # Edge output
        self.edge_mlp = torch.nn.Sequential(
            torch.nn.Linear(2 * hidden_dim, 4*hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(4*hidden_dim, 4*hidden_dim),
            torch.nn.ReLU(),
            # torch.nn.Linear(4*hidden_dim, 4*hidden_dim),
            # torch.nn.ReLU(),
            torch.nn.Linear(4*hidden_dim, 2*hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(2*hidden_dim, 1)
        )

    def forward(self, x, A, t, mask):
        B, N, F = x.shape
        X = self.input_proj(x)
        t_emb = self.time_emb(t)[:, None, :]
        X = X + t_emb

        A = (A + A.transpose(1, 2)) / 2
        A = A + 1e-4 * torch.eye(A.size(-1), device=A.device).unsqueeze(0)

        # Implementation in vertex domain
        node_state = torch.zeros_like(X)
        for k in range(self.filter_length):
            node_state += torch.relu(self.h[k] * torch.linalg.matrix_power(A, k) @ X)

        node_state = node_state * mask.unsqueeze(-1)
        eps_x = self.node_out(node_state)

        # --- Predict edge noise ---
        h_i = node_state.unsqueeze(2).expand(-1, -1, N, -1)
        h_j = node_state.unsqueeze(1).expand(-1, N, -1, -1)

        edge_input = torch.cat([h_i, h_j], dim=-1)
        eps_A = self.edge_mlp(edge_input).squeeze(-1)

        # Mask invalid edges
        adj_mask = mask.unsqueeze(1) * mask.unsqueeze(2)
        eps_A = eps_A * adj_mask

        # Symmetrize
        eps_A = (eps_A + eps_A.transpose(1, 2)) / 2

        return eps_x, eps_A

def train_diffusion(
    model,
    dataset,
    epochs=100,
    batch_size=32,
    lr=1e-3,
    save_path="graph_diffusion.pt",
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    model = model.to(device)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, min_lr=1e-8)

    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0

        for batch in loader:
            batch = batch.to(device)

            optimizer.zero_grad()

            loss = model.loss(batch.x, batch.edge_index, batch.batch)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        scheduler.step(avg_loss)
        print(f"Epoch {epoch:03d} | Loss: {avg_loss:.4f}")

        # --- Save best model ---
        if avg_loss < best_loss:
            best_loss = avg_loss

            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "loss": avg_loss,
            }, save_path)

            print(f"Saved best model (loss={avg_loss:.4f})")

    # --- Save final model ---
    torch.save(model.state_dict(), save_path.replace(".pt", "_last.pt"))

    print("Training complete")
    
def get_wl_hash(data):
    """Converts a PyG Data object to a WL hash string."""
    # 1. Convert to NetworkX
    G = to_networkx(data, node_attrs=['x'], to_undirected=True)
    
    return nx.weisfeiler_lehman_graph_hash(G, node_attr='x', iterations=3)


def evaluate_generator(gen_samples, train_dataset, node_count=None, n_samples=1000, diffusion = False):
    # Get hashes for Training Data
    train_hashes = set(get_wl_hash(d) for d in train_dataset)
    
    gen_hashes = [get_wl_hash(s) for s in gen_samples]
    
    # Calculate Metrics
    unique_gen_hashes = set(gen_hashes)
    
    uniqueness = len(unique_gen_hashes) / n_samples
    
    # Novelty: percent of generated graphs that don't exist in training
    novel_count = sum(1 for h in gen_hashes if h not in train_hashes)
    novelty = novel_count / n_samples
    
    # Unique & Novel: percent of generated graphs that are unique AND not in training
    unique_novel_count = sum(1 for h in unique_gen_hashes if h not in train_hashes)
    unique_novelty = unique_novel_count / n_samples

    print(f"Uniqueness:     {uniqueness:.2%}")
    print(f"Novelty:        {novelty:.2%}")
    print(f"Unique & Novel: {unique_novelty:.2%}")

def extract_graph_stats(dataset):
    degrees = []
    clustering = []
    eigen = []

    for data in dataset:
        G = to_networkx(data, to_undirected=True)

        if G.number_of_nodes() == 0:
            continue

        deg = np.mean([d for _, d in G.degree()])
        clust = np.mean(list(nx.clustering(G).values()))

        try:
            eig = np.mean(list(nx.eigenvector_centrality(G, max_iter=500).values()))
        except:
            continue

        degrees.append(deg)
        clustering.append(clust)
        eigen.append(eig)

    return degrees, clustering, eigen

def generate_samples(generator, n_samples, diffusion=False, node_count=None):
    samples = []

    for _ in range(n_samples):
        if diffusion:
            N = int(np.random.choice(node_count))
            samples.append(generator.sample(N))
        else:
            samples.append(generator())

    return samples

def get_bins(*arrays, bins=30):
    all_data = np.concatenate(arrays)
    return np.histogram_bin_edges(all_data, bins=bins)

def extract_node_stats(dataset):
    degrees, clustering, eigen = [], [], []
    for data in dataset:
        G = to_networkx(data, to_undirected=True)
        if G.number_of_nodes() == 0: continue
        
        # Node-level metrics
        degrees.extend([d for _, d in G.degree()])
        clustering.extend(list(nx.clustering(G).values()))
        try:
            # Eigenvector centrality can fail on very sparse/disconnected graphs
            ec = nx.eigenvector_centrality(G, max_iter=1000)
            eigen.extend(list(ec.values()))
        except:
            continue
    return degrees, clustering, eigen

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, default='train', choices=['train', 'test'], help='what to do when running the script (default: %(default)s)')
    args = parser.parse_args()

    network = GraphConvDenoiser(
        node_feature_dim=7,   
        filter_length=5,      
        hidden_dim=128,        
        T=100                 
    )

    graph_ddpm = GraphDiffusion(network)

    if args.mode == 'train':
        train_diffusion(graph_ddpm, dataset)

    
    if args.mode == 'test':
        node_count = [data.num_nodes for data in dataset]
        erdos_renyi_generator = ErdosRenyi(dataset)
        

        checkpoint = torch.load("graph_diffusion.pt")
        graph_ddpm.load_state_dict(checkpoint["model_state_dict"])

        graph_ddpm.eval()

        train_deg, train_clust, train_eig = extract_graph_stats(train_dataset)

        baseline_samples = generate_samples(erdos_renyi_generator, 1000)
        evaluate_generator(baseline_samples, train_dataset)
        base_deg, base_clust, base_eig = extract_graph_stats(baseline_samples)

        diff_samples = generate_samples(graph_ddpm, 1000, diffusion=True, node_count=node_count)
        evaluate_generator(diff_samples, train_dataset)
        diff_deg, diff_clust, diff_eig = extract_graph_stats(diff_samples)

        train_stats = extract_node_stats(train_dataset)
        base_stats = extract_node_stats(baseline_samples)
        diff_stats = extract_node_stats(diff_samples)

        # 2. Setup Plot
        fig, axes = plt.subplots(3, 3, figsize=(15, 12), sharey='row')
        metrics_names = ['Node Degree', 'Clustering Coefficient', 'Eigenvector Centrality']
        column_names = ['Empirical', 'Baseline', 'Diffusion']

        # 3. Loop through metrics (Rows)
        for i in range(3):
            # Combine data from all 3 sources to find global min/max for consistent bins
            combined_data = train_stats[i] + base_stats[i] + diff_stats[i]
            
            # Special handling for Degree (integer bins) vs Coefficients (float bins)
            if i == 0: # Degree
                bins = np.arange(min(combined_data), max(combined_data) + 2) - 0.5
            else: # Clustering and Centrality
                bins = np.histogram_bin_edges(combined_data, bins=30)

            # Plot each source (Columns)
            row_data = [train_stats[i], base_stats[i], diff_stats[i]]
            
            
            for j in range(3):
                ax = axes[i, j]
                ax.hist(row_data[j], bins=bins, density=True, rwidth = 0.9)
                
                # Labeling
                if i == 0: ax.set_title(column_names[j], fontsize=14)
                if j == 0: ax.set_ylabel(metrics_names[i], fontsize=12)
                
                

        plt.tight_layout()
        plt.savefig('MUTAG_Comparison_Grid.png')






