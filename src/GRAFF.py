import torch
import torch.nn.functional as F
import torch.nn.utils.parametrize as parametrize
import torch.nn as nn

import torch_geometric
from torch_geometric.nn import MessagePassing, GCNConv
from torch_geometric.utils import add_self_loops, degree, homophily
from torch_geometric.utils import negative_sampling

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
    def __init__(self, input_dim, device = 'cpu'):
        super().__init__()
        self.w = torch.nn.Parameter(torch.empty((1, input_dim)))
        self.reset_parameters()
        self.to(device)
    
    def reset_parameters(self):
        torch.nn.init.normal_(self.w)

    def forward(self, x):
        # x * self.w behave like a diagonal matrix op., we multiply each row of x by the element-wise w
        return x * self.w


class Source_b(nn.Module):
    def __init__(self, device = 'cpu'):
        super().__init__()
        self.beta = torch.nn.Parameter(torch.empty(1))
     
        self.reset_parameters()
        self.to(device)

    def reset_parameters(self):
        torch.nn.init.normal_(self.beta)
    


    def forward(self, x):
        return x * self.beta


class PairwiseInteraction_w(nn.Module):
    def __init__(self, input_dim, symmetry_type='1', device = 'cpu'):
        super().__init__()
        self.W = torch.nn.Linear(input_dim + 2, input_dim, bias = False)

        if symmetry_type == '1':
            symmetry = PairwiseParametrization()
        elif symmetry_type == '2':
            symmetry = Symmetric()

        parametrize.register_parametrization(
            self.W, 'weight', symmetry, unsafe=True)
        self.reset_parameters()
        self.to(device)
        
    def reset_parameters(self):
        self.W.reset_parameters()

    def forward(self, x):
        return self.W(x)


class GRAFFConv(MessagePassing):
    def __init__(self, external_w, source_b, pairwise_w, self_loops=True):
        super().__init__(aggr='add')

        self.self_loops = self_loops
        self.external_w = external_w #External_W(self.in_dim, device=device)
        self.beta = source_b #Source_b(device = device)
        self.pairwise_W = pairwise_w #PairwiseInteraction_w(self.in_dim, symmetry_type=symmetry_type, device=device)
   

    def forward(self, x, edge_index, x0):

        # We set the source term, which corrensponds with the initial conditions of our system.

        if self.self_loops:
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.shape[0])

        out_p = self.pairwise_W(x)

        out = self.propagate(edge_index, x=out_p)

        out = out - self.external_w(x) + self.beta(x0)

        return out

    def message(self, x_j, edge_index, x):
        # Does we need the degree of the row or from the columns?
        # x_i are the columns indices, whereas x_j are the row indices
        row, col = edge_index

        # Degree is specified by the row (outgoing edges)
        deg_matrix = degree(col, num_nodes=x.shape[0], dtype=x.dtype)
        deg_inv = deg_matrix.pow(-0.5)
        
        deg_inv[deg_inv == float('inf')] = 0

        denom_degree = deg_inv[row]*deg_inv[col]

        # Each row of denom_degree multiplies (element-wise) the rows of x_j
        return denom_degree.unsqueeze(-1) * x_j



class PhysicsGNN_NC(nn.Module):
    def __init__(self, dataset, hidden_dim, num_layers, step = 0.1, symmetry_type='1', self_loops=False, device = 'cpu'):
        super().__init__()

        self.enc = torch.nn.Linear(dataset.num_features, hidden_dim, bias = False)
        self.dec = torch.nn.Linear(hidden_dim, dataset.num_classes, bias = False)

        self.external_w = External_W(hidden_dim, device=device)
        self.source_b = Source_b(device = device)
        self.pairwise_w = PairwiseInteraction_w(hidden_dim, symmetry_type=symmetry_type, device=device)

        self.layers = [GRAFFConv(self.external_w, self.source_b, self.pairwise_w,
                            self_loops=self_loops) for i in range(num_layers)]
             
        self.step = step
        self.reset_parameters()
        self.to(device)

    def reset_parameters(self):
        self.enc.reset_parameters()
        self.dec.reset_parameters()
        self.external_w.reset_parameters()
        self.source_b.reset_parameters()
        self.pairwise_w.reset_parameters()


        
    def forward(self, data):

        x, edge_index = data.x, data.edge_index

        
        x = enc_out = self.enc(x)

        x0 = enc_out.clone()
        for layer in self.layers:
                
            x = x + self.step*F.relu(layer(x, edge_index, x0))

        output = self.dec(x)

        return output
    



class LinkPredictor(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers = 0, bias = False, dropout= 0, device = 'cpu'):
        super().__init__()
        
        self.num_layers = num_layers
        layers = []
        if self.num_layers != 0:
            
            layers.append(nn.Linear(input_dim, output_dim, bias = bias))
            for layer in range(self.num_layers):
                layers.append(nn.Linear(output_dim, output_dim, bias = bias))
        
            layers.append(nn.Linear(output_dim, 1, bias = bias))    
        else:
            layers.append(nn.Linear(input_dim, 1, bias = bias))    

    
        self.layers = nn.Sequential(*layers)
        self.dropout = dropout
        self.to(device)
        self.reset_parameters()
             
    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()
            
    def forward(self, x_i, x_j, training = False):
        
        out = x_i * x_j 
        if self.num_layers != 0:
            for layer_idx in range(len(self.layers)-1):
                out = self.layers[layer_idx](out)
                out = F.relu(out)
                out = F.dropout(out, p = self.dropout, training = training)
        out = self.layers[-1](out)

        out = torch.sigmoid(out)

        return out
    
    

class PhysicsGNN_LP(nn.Module):
    def __init__(self, dataset, hidden_dim, num_layers, step=0.1, symmetry_type='1', self_loops=False, device='cpu'):
        super().__init__()

        self.enc = torch.nn.Linear(
            dataset.num_features, hidden_dim, bias=False)

        self.external_w = External_W(hidden_dim, device=device)
        self.source_b = Source_b(device=device)
        self.pairwise_w = PairwiseInteraction_w(
            hidden_dim, symmetry_type=symmetry_type, device=device)

        self.layers = [GRAFFConv(self.external_w, self.source_b, self.pairwise_w,
                                 self_loops=self_loops) for i in range(num_layers)]

        self.step = step
        
        self.reset_parameters()
        self.to(device)

    def reset_parameters(self):
        self.enc.reset_parameters()
        self.external_w.reset_parameters()
        self.source_b.reset_parameters()
        self.pairwise_w.reset_parameters()


    def forward(self, data):

        x, edge_index = data.x.clone(), data.edge_index.clone()
        

        x = enc_out = self.enc(x)

        x0 = enc_out.clone()
        
        for layer in self.layers:

            x = x + self.step*F.relu(layer(x, edge_index, x0))

        return x
    
class GNN_LP(nn.Module):
    def __init__(self, dataset, hidden_dim, num_layers, GNN = GCNConv, self_loops = False, device='cpu'):
        super().__init__()

        self.enc = torch.nn.Linear(
            dataset.num_features, hidden_dim, bias=False)

        layers = [GNN(hidden_dim, hidden_dim, add_self_loops = self_loops) for i in range(num_layers)]
        
        self.layers = nn.Sequential(*layers)
        self.reset_parameters()
        self.to(device)

    def reset_parameters(self):
        self.enc.reset_parameters()
        for layer in self.layers:
            layer.reset_parameters()


    def forward(self, data):

        x, edge_index = data.x.clone(), data.edge_index.clone()
        

        x = self.enc(x)

        for layer in self.layers:

            x = x + F.relu(layer(x, edge_index))

        return x     

# class GNNStack(torch.nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout, emb=False):
#         super(GNNStack, self).__init__()
#         conv_model = torch_geometric.nn.SAGEConv

#         self.convs = nn.ModuleList()
#         self.convs.append(conv_model(input_dim, hidden_dim))
#         self.dropout = dropout
#         self.num_layers = num_layers
#         self.emb = emb

#         # Create num_layers GraphSAGE convs
#         assert (self.num_layers >= 1), 'Number of layers is not >=1'
#         for l in range(self.num_layers - 1):
#             self.convs.append(conv_model(hidden_dim, hidden_dim))

#         # post-message-passing processing 
#         self.post_mp = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim), nn.Dropout(self.dropout),
#             nn.Linear(hidden_dim, output_dim))

#     def forward(self, data):

#         x, edge_index = data.x, data.edge_index

#         for i in range(self.num_layers):
#             x = self.convs[i](x, edge_index)
#             x = F.relu(x)
#             x = F.dropout(x, p=self.dropout, training=self.training)

#         x = self.post_mp(x)

#         # Return final layer of embeddings if specified
#         if self.emb:
#             return x

#         # Else return class probabilities
#         return F.log_softmax(x, dim=1)

            
# class LinkPredictor(nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
#                  dropout):
#         super(LinkPredictor, self).__init__()

#         # Create linear layers
#         self.lins = nn.ModuleList()
#         self.lins.append(nn.Linear(in_channels, hidden_channels))
#         for _ in range(num_layers - 2):
#             self.lins.append(nn.Linear(hidden_channels, hidden_channels))
#         self.lins.append(nn.Linear(hidden_channels, out_channels))

#         self.dropout = dropout

#     def reset_parameters(self):
#         for lin in self.lins:
#             lin.reset_parameters()

#     def forward(self, x_i, x_j, training = False):
#         # x_i and x_j are both of shape (E, D)
#         x = x_i * x_j
#         for lin in self.lins[:-1]:
#             x = lin(x)
#             x = F.relu(x)
#             x = F.dropout(x, p=self.dropout, training=training)
#         x = self.lins[-1](x)
#         return torch.sigmoid(x)
        
        