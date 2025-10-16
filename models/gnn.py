import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)
    
    def forward(self, x, adj):
        """
        x: node features (batch_size, num_nodes, in_features)
        adj: adjacency matrix (batch_size, num_nodes, num_nodes)
        """
        support = torch.matmul(x, self.weight)
        output = torch.matmul(adj, support)
        return output + self.bias

class GNN(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, output_dim=512, num_layers=3, dropout=0.2):
        super().__init__()
        
        self.layers = nn.ModuleList()
        self.layers.append(GraphConvolution(input_dim, hidden_dim))
        
        for _ in range(num_layers - 2):
            self.layers.append(GraphConvolution(hidden_dim, hidden_dim))
        
        self.layers.append(GraphConvolution(hidden_dim, output_dim))
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, node_features, adjacency_matrix):
        """
        node_features: (batch_size, num_nodes, input_dim)
        adjacency_matrix: (batch_size, num_nodes, num_nodes)
        """
        x = node_features
        
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x, adjacency_matrix)
            x = F.relu(x)
            x = self.dropout(x)
        
        # Last layer
        x = self.layers[-1](x, adjacency_matrix)
        
        # Global pooling
        x = torch.mean(x, dim=1)  # (batch_size, output_dim)
        return x