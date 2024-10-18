import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class SignGraphReasoner(nn.Module):
    def __init__(self, node_features, hidden_dim, output_dim):
        """
        Initializes the Graph Neural Network for contextual reasoning.

        Args:
            node_features (int): Number of features per node.
            hidden_dim (int): Number of hidden units.
            output_dim (int): Number of output features per node.
        """
        super(SignGraphReasoner, self).__init__()
        self.conv1 = GCNConv(node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
    
    def forward(self, data):
        """
        Forward pass through the GNN.

        Args:
            data (torch_geometric.data.Data): Graph data.

        Returns:
            torch.Tensor: Output node features.
        """
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x
