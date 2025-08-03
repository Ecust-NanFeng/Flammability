import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_add_pool, global_max_pool
import warnings

warnings.filterwarnings('ignore')
class MolecularGNN(nn.Module):
    """
    Molecular-level Graph Neural Network encoder.

    Encodes individual molecules into fixed-size embeddings using GCN layers.
    """

    def __init__(self, num_atom_features=25, hidden_dim=128, dropout_rate=0.2):
        """
        Initialize molecular GNN.

        Args:
            num_atom_features (int): Number of input atom features
            hidden_dim (int): Hidden dimension size
            dropout_rate (float): Dropout probability
        """
        super(MolecularGNN, self).__init__()

        # Graph convolution layers
        self.conv1 = GCNConv(num_atom_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        # Batch normalization layers
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, edge_index, batch):
        """
        Forward pass of molecular GNN.

        Args:
            x: Node features
            edge_index: Edge connectivity
            batch: Batch assignment vector

        Returns:
            torch.Tensor: Molecular embeddings
        """
        # Check batch size for training stability
        batch_size = batch.max().item() + 1 if batch.numel() > 0 else 1
        if batch_size == 1 and self.training:
            return None  # Skip batch with size 1 during training

        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)

        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x_add = global_add_pool(x, batch)

        x = torch.cat([x_mean, x_max, x_add], dim=1)

        return x


class MixtureGNN(nn.Module):
    """
    Mixture-level Graph Neural Network model.

    Combines molecular embeddings based on mixture ratios to predict properties.
    """

    def __init__(self, num_atom_features=25, hidden_dim=128, dropout_rate=0.2):
        """
        Initialize mixture GNN.

        Args:
            num_atom_features (int): Number of input atom features
            hidden_dim (int): Hidden dimension size
            dropout_rate (float): Dropout probability
        """
        super(MixtureGNN, self).__init__()

        # Molecular encoder
        self.molecular_gnn = MolecularGNN(num_atom_features, hidden_dim, dropout_rate)

        # Mixture feature transformation
        self.mixture_transform = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        # Regression head
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, batch_data):
        """
        Forward pass of mixture GNN.

        Args:
            batch_data (dict): Batched mixture data

        Returns:
            torch.Tensor: Property predictions
        """
        graphs = batch_data['graphs']
        ratios_list = batch_data['ratios']
        batch_indices = batch_data['batch_indices']

        if graphs is None:
            return torch.zeros((len(ratios_list), 1))

        # Get embeddings for all molecules
        mol_embeddings = self.molecular_gnn(
            graphs.x,
            graphs.edge_index,
            graphs.batch
        )

        # Aggregate components for each mixture
        mixture_embeddings = []
        current_idx = 0

        for i in range(len(ratios_list)):
            # Find molecules belonging to current mixture
            num_components = len(ratios_list[i])
            component_embeddings = mol_embeddings[current_idx:current_idx + num_components]
            component_ratios = ratios_list[i].unsqueeze(1)

            # Weighted combination based on mixture ratios
            weighted_embedding = torch.sum(component_embeddings * component_ratios, dim=0)
            mixture_embeddings.append(weighted_embedding)

            current_idx += num_components

        mixture_embeddings = torch.stack(mixture_embeddings)

        # Mixture-level transformation
        mixture_features = self.mixture_transform(mixture_embeddings)

        # Property prediction
        predictions = self.regressor(mixture_features)

        return predictions