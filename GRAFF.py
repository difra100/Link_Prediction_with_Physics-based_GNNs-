import torch
import torch.nn.functional as F
import torch.nn.utils.parametrize as parametrize
import torch.nn as nn

import torch_geometric
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, homophily

class Symmetric(torch.nn.Module):
    def forward(self, w):
        # This class implements the method to define the symmetry in the squared matrices.
        return w.triu(0) + w.triu(1).transpose(-1, -2)
    
class PairwiseParametrization(torch.nn.Module):
    def forward(self, W):
        # Construct a symmetric matrix with zero diagonal
        # The weights are initialized to be non-squared, with 2 additional columns. We cut from two of these
        # two vectors q and r, and then we compute w_diag as described in the paper.
        # This procedure is done in order to easily distribute the mass in its spectrum through the values of q and r
        W0 = W[:, :-2].triu(1)

        W0 = W0 + W0.T

        # Retrieve the `q` and `r` vectors from the last two columns
        q = W[:, -2]
        r = W[:, -1]
        # Construct the main diagonal
        w_diag = torch.diag(q * torch.sum(torch.abs(W0), 1) + r)

        return W0 + w_diag

class External_W(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.w = torch.nn.Parameter(torch.empty((1, input_dim)))
        self.reset_parameters()
    
    def reset_parameters(self):
        torch.nn.init.normal_(self.w)

    def forward(self, x):
        # x * self.w behave like a diagonal matrix op., we multiply each row of x by the element-wise w
        return x * self.w


class Source_b(nn.Module):
    def __init__(self):
        super().__init__()
        self.beta = torch.nn.Parameter(torch.empty(1))

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.normal_(self.beta)
    


    def forward(self, x):
        return x * self.beta


class PairwiseInteraction_w(nn.Module):
    def __init__(self, input_dim, symmetry_type='1'):
        super().__init__()
        self.W = torch.nn.Linear(input_dim + 2, input_dim)

        if symmetry_type == '1':
            symmetry = PairwiseParametrization()
        elif symmetry_type == '2':
            symmetry = Symmetric()

        parametrize.register_parametrization(
            self.W, 'weight', symmetry, unsafe=True)
        self.reset_parameters()
        
    def reset_parameters(self):
        self.W.reset_parameters()

    def forward(self, x):
        return self.W(x)


class GRAFFConv(MessagePassing):
    def __init__(self, input_dim, symmetry_type='1', self_loops=True):
        super().__init__(aggr='add')
        self.in_dim = input_dim
        self.self_loops = self_loops
        self.external_w = External_W(self.in_dim)
        self.beta = Source_b()
        self.pairwise_W = PairwiseInteraction_w(
            self.in_dim, symmetry_type=symmetry_type)

    def forward(self, x, edge_index, x0):

        # We set the source term, which corrensponds with the initial conditions of our system.

        if self.self_loops:
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.shape[0])

        out_p = self.pairwise_W(x)

        out = self.propagate(edge_index, x=out_p)

        out = out - (self.external_w(x) + self.beta(x0))

        return out

    def message(self, x_i, edge_index, x):
        # Does we need the degree of the row or from the columns?
        # x_i are the columns indices, whereas x_j are the row indices
        row, col = edge_index

        # Degree is specified by the row (outgoing edges)
        deg_matrix = degree(row, num_nodes=x.shape[0], dtype=x.dtype)
        deg_inv = deg_matrix.pow(-0.5)
        
        deg_inv[deg_inv == float('inf')] = 0

        denom_degree = deg_inv[row]*deg_inv[col]

        # Each row of denom_degree multiplies (element-wise) the rows of x_j
        return denom_degree.unsqueeze(-1) * x_i


class PhysicsGNN_NC(nn.Module):
    def __init__(self, dataset, hidden_dim, num_layers, step = 0.1, symmetry_type='1', self_loops=False):
        super().__init__()

        self.enc = torch.nn.Linear(dataset.num_features, hidden_dim)
        self.dec = torch.nn.Linear(hidden_dim, dataset.num_classes)

        self.layers = [GRAFFConv(hidden_dim, symmetry_type=symmetry_type,
                            self_loops=self_loops) for i in range(num_layers)]
        self.step = step
        self.reset_parameters()

    def reset_parameters(self):
        self.enc.reset_parameters()
        self.dec.reset_parameters()
        for layer in self.layers:
            layer.reset_parameters()


        
    def forward(self, data):

        x, edge_index = data.x, data.edge_index

        
        x = enc_out = self.enc(x)

        x0 = enc_out.clone()
        for layer in self.layers:
                
            x = x + self.step*F.relu(layer(x, edge_index, x0))

        output = self.dec(x)

        return output
        
            
    
        
        