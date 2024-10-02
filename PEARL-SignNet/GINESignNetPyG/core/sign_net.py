import torch, numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter
from core.transform import to_dense_list_EVD

import core.model_utils.masked_layers as masked_layers 
import core.model_utils.pyg_gnn_wrapper as elements
from core.model import GNN
from core.model_utils.transformer_module import TransformerEncoderLayer, PositionalEncoding 
from core.model_utils.elements import DiscreteEncoder

class GNN3d(nn.Module): 
    """
    Apply GNN on a 3-dimensional data x: n x k x d. 
    Equivalent to apply GNN on k independent nxd 2-d feature.
    * Assume no edge feature for now.
    """
    def __init__(self, n_in, n_out, n_layer, gnn_type='MaskedGINConv', NOUT=128, skipc=True):
        super().__init__()
        self.convs = nn.ModuleList([getattr(masked_layers, gnn_type)(n_in if i==0 else n_out, n_out, bias=False) for i in range(n_layer)])
        self.norms = nn.ModuleList([masked_layers.MaskedBN(n_out) for _ in range(n_layer)])
        self.skipc = skipc
        if skipc:
            self.output_encoder = nn.Linear(n_layer*n_out, n_out,bias=False)
            self.output_norm = masked_layers.MaskedBN(n_out)

    def reset_parameters(self):
        if self.skipc:
            self.output_encoder.reset_parameters()
            self.output_norm.reset_parameters()
        for conv, norm in zip(self.convs, self.norms): 
            conv.reset_parameters()
            norm.reset_parameters() 
    # TRY SKIP CONNECTIONS
    def forward(self, x, edge_index, edge_attr, mask=None):
        # x: n x k x d
        # mask: n x k
        x = x.transpose(0, 1) # k x n x d
        if mask is not None:
            mask = mask.transpose(0, 1) # k x n
        previous_x = 0
        skip_connections = []
        for conv, norm in zip(self.convs, self.norms):
            #TODO: current not work for continuous edge attri
            # x = conv(x, edge_index, enc(edge_attr), mask) # pass mask into
            x = conv(x, edge_index, edge_attr, mask) # pass mask into
            # assert x[~mask].max() == 0 
            #import pdb; pdb.set_trace()
            if mask is not None:
                x[~mask] = 0
            x = norm(x, mask)
            x = F.relu(x)
            if self.skipc:
                skip_connections.append(x)
            else:
                x = x + previous_x
                previous_x = x
        if self.skipc:
            x = torch.cat(skip_connections, dim=-1)
            x = self.output_encoder(x)
            x = self.output_norm(x) # maybe play w this
        return x.transpose(0, 1)

class GNN3d_S(nn.Module): 
    """
    Apply GNN on a 3-dimensional data x: n x k x d. 
    Equivalent to apply GNN on k independent nxd 2-d feature.
    * Assume no edge feature for now.
    """
    def __init__(self, n_in, n_out, n_layer, gnn_type='MaskedGINConv'):
        super().__init__()
        self.convs = nn.ModuleList([getattr(masked_layers, gnn_type)(n_in if i==0 else n_out, n_out, bias=False) for i in range(n_layer)])
        self.norms = nn.ModuleList([masked_layers.MaskedBN(n_out) for _ in range(n_layer)])
        self.edge_encoders = nn.ModuleList([DiscreteEncoder(n_in if i==0 else n_out) for i in range(n_layer)]) # only for categorical features

    def reset_parameters(self):
        for conv, norm, edge in zip(self.convs, self.norms, self.edge_encoders): 
            conv.reset_parameters()
            norm.reset_parameters() 
            edge.reset_parameters()
    
    def forward(self, x, edge_index, edge_attr, mask=None):
        # x: n x k x d
        # mask: n x k
        x = x.transpose(0, 1) # k x n x d
        if mask is not None:
            mask = mask.transpose(0, 1) # k x n
        previous_x = 0
        for conv, norm, enc in zip(self.convs, self.norms, self.edge_encoders):
            #TODO: current not work for continuous edge attri
            # x = conv(x, edge_index, enc(edge_attr), mask) # pass mask into
            x = conv(x, edge_index, edge_attr, mask) # pass mask into
            # assert x[~mask].max() == 0 
            if mask is not None:
                x[~mask] = 0
            x = norm(x, mask)
            x = F.relu(x, inplace=False)
            x = x+ previous_x
            previous_x = x
        return x.transpose(0, 1)

class SetTransformer(nn.Module):
    def __init__(self, nhid, nlayer):
        super().__init__()
        # self.pos_encoder = PositionalEncoding(nhid, freq=100)
        self.pos_encoder = masked_layers.MaskedMLP(1, nhid, nlayer=2)
        
        self.transformer_layers = nn.ModuleList(TransformerEncoderLayer(nhid, n_head=4) for _ in range(nlayer))
        self.out = nn.Sequential(nn.Linear(nhid, nhid, bias=False), nn.BatchNorm1d(nhid))

    def reset_parameters(self):
        # self.transformer_encoder.reset_parameters()
        # self.encoder.reset_parameters()
        self.pos_encoder.reset_parameters()

    def forward(self, x, pos, mask):
        # x: n x k x d
        # pos: n x k 
        # mask: n x k
        # x = self.encoder(x) + self.pos_encoder(pos, mask)
        # x[~mask] = 0
        x = x + pos
        if x[~mask].numel() > 0:
            assert x[~mask].max() == 0
        for layer in self.transformer_layers:
            if x[~mask].numel() > 0:
                assert x[~mask].max() == 0
            x,_ = layer(x, mask)
        x = torch.sum(x, dim=1) # n x d #### TODO: change later
        x = self.out(x)
        return x

class SignNet_S(nn.Module):
    """
        n x k node embeddings => n x n_hid 

        The output is sign invariant and permutation equivariant 
    """
    def __init__(self, n_hid, nl_phi, nl_rho=2):
        super().__init__()
        self.phi = GNN3d(1, n_hid, nl_phi, gnn_type='MaskedGINConv') 
        self.rho = SetTransformer(n_hid, nl_rho)

        self.eigen_encoder1 = masked_layers.MaskedMLP(1, n_hid, nlayer=1)
        self.eigen_encoder2 = masked_layers.MaskedMLP(1, n_hid, nlayer=2)

    def reset_parameters(self):
        self.phi.reset_parameters()
        self.rho.reset_parameters()
        self.eigen_encoder1.reset_parameters()
        self.eigen_encoder2.reset_parameters()

    def forward(self, data):
        eigS_dense, eigV_dense = to_dense_list_EVD(data.eigen_values, data.eigen_vectors, data.batch) 
        
        top_k = 8#min(8, eigS_dense.size(1))
        '''eigS_topk = eigS_dense[:, -top_k:]
        eigV_topk = eigV_dense[:, -top_k:]'''

        eigS_topk = eigS_dense[:, 1:top_k+1]
        eigV_topk = eigV_dense[:, 1:top_k+1]
        
        x = eigV_topk
        # get mask and prepare x

        size = scatter(torch.ones_like(x[:,0], dtype=torch.int64), data.batch, dim=0, reduce='add') # b x 1
        mask = torch.arange(x.size(1), device=x.device)[None, :] < size[:, None]                    # b x N_max
        mask_full = mask[data.batch]

        # transform eigens 
        x = x.unsqueeze(-1)
        # x = self.eigen_encoder1(x, mask_full)
        # assert x[~mask_full].max() == 0
        #print(x[~mask_full].numel())
        pos = self.eigen_encoder2(eigS_topk.unsqueeze(-1), mask_full)
        pos = 0 # ignore eigenvalues 

        # phi
        #print(x[~mask_full].numel())
        x = self.phi(x, data.edge_index, data.edge_attr, mask_full) + self.phi(-x, data.edge_index, data.edge_attr, mask_full)

        # rho = Transformer
        #print(x[~mask_full].numel())
        x = self.rho(x, pos=pos, mask=mask_full)

        return x # n x n_hid 



# RANDOM SIGNNET FOR RANDOM INPUT
class SignNet(nn.Module):
    """
        n x k node embeddings => n x n_hid 

        The output is sign invariant and permutation equivariant 
    """
    def __init__(self, n_hid, nl_phi, nl_rho=2, NOUT=128):
        super().__init__()
        self.phi = GNN3d(1, n_hid, nl_phi, gnn_type='MaskedGINConv', NOUT=NOUT) 
        #self.rho = SetTransformer(n_hid, nl_rho)

    def reset_parameters(self):
        self.phi.reset_parameters()
        #self.rho.reset_parameters()

    def forward(self, data, rand):
        #eigS_dense, eigeigV_dense = to_dense_list_EVD(data.eigen_values, data.eigen_vectors, data.batch) 
        x1 = rand
        # get mask and prepare x
        #x = x1.squeeze()
        #size = scatter(torch.ones_like(x[:,0], dtype=torch.int64), data.batch, dim=0, reduce='add') # b x 1
        #mask = torch.arange(x.size(1), device=x.device)[None, :] < size[:, None]                    # b x N_max
        #mask_full = mask[data.batch]
        mask_full = None
        x = x1

        # phi
        x = self.phi(x, data.edge_index, data.edge_attr, mask_full) #+ self.phi(-x, data.edge_index, data.edge_attr, mask_full)

        # rho = Transformer
        #x = self.rho(x.transpose(0, 1), pos=0, mask=mask_full)
        x = x.mean(dim=1)
        return x#.mean(dim=1) # n x n_hid 
    
# SIGNNET capping eigenvectors, without transformer or GNN3D
class SignNet2(nn.Module):
    """
        n x k node embeddings => n x n_hid 

        The output is sign invariant and permutation equivariant 
    """
    def __init__(self, n_hid, nl_phi, nl_rho=2):
        super().__init__()
        self.max_eigenvectors =30
        self.phi = GNN3d(self.max_eigenvectors, n_hid, nl_phi, gnn_type='MaskedGINConv') 
        #self.rho = SetTransformer(1, nl_rho)
        #self.eigen_encoder1 = masked_layers.MaskedMLP(self.max_eigenvectors, n_hid, nlayer=1)
        #self.eigen_encoder2 = masked_layers.MaskedMLP(1, n_hid, nlayer=2)
        self.out = nn.Sequential(nn.Linear(n_hid, n_hid, bias=False), nn.BatchNorm1d(n_hid))
        #self.max_eigenvectors = 1

    def reset_parameters(self):
        self.phi.reset_parameters()
        #self.rho.reset_parameters()
        #self.eigen_encoder1.reset_parameters()
        #self.eigen_encoder2.reset_parameters()
        #self.out.reset_parameters()

    def forward(self, data, no_transformer=True):
        # Convert eigenvalues and eigenvectors to dense format
        eigS_dense, eigV_dense = to_dense_list_EVD(data.eigen_values, data.eigen_vectors, data.batch)
        #import pdb; pdb.set_trace()
        # Limit the number of eigenvectors to 8 per graph
        max_eigenvectors = self.max_eigenvectors
        num_graphs, num_nodes = eigV_dense.shape
        if num_nodes < max_eigenvectors:
            # Create a new tensor for the padded eigenvectors 
            padded_eigV_dense = eigV_dense.new_zeros((num_graphs, max_eigenvectors))
            
            # Copy the original eigenvectors into the new tensor
            padded_eigV_dense[:, :num_nodes] = eigV_dense

            # Update eigV_dense to the padded version
            eigV_dense = padded_eigV_dense

        # If the number of eigenvectors is greater or equal to max_eigenvectors, truncate
        eigV_dense = eigV_dense[:, :max_eigenvectors]
        
        x = eigV_dense
        
        # Get mask and prepare x
        size = scatter(torch.ones_like(x[:,0], dtype=torch.int64), data.batch, dim=0, reduce='add') # b x 1
        mask = torch.arange(x.size(1), device=x.device)[None, :] < size[:, None]                    # b x N_max
        mask_full = mask[data.batch].squeeze()

        # Transform eigens
        #x = x.unsqueeze(-1)
        # Ignore eigenvalues
        pos = 0  # You could also process eigS_dense if necessary

        # Apply phi transformation for sign invariance
        #x = self.eigen_encoder1(x, None) + self.eigen_encoder1(-x, None)
        x = self.phi(x, data.edge_index, data.edge_attr, mask_full) #+ self.phi(-x, data.edge_index, data.edge_attr, mask_full)
        #x = torch.sum(x, dim=1)

        if no_transformer:
            x = self.out(x)
        else:
            x = self.rho(x, pos=0, mask=mask_full)
        return  x


'''def forward(self, data):
        eigS_dense, eigV_dense = to_dense_list_EVD(data.eigen_values, data.eigen_vectors, data.batch) 
        x = eigV_dense
        # get mask and prepare x
        size = scatter(torch.ones_like(x[:,0], dtype=torch.int64), data.batch, dim=0, reduce='add') # b x 1
        mask = torch.arange(x.size(1), device=x.device)[None, :] < size[:, None]                    # b x N_max
        mask_full = mask[data.batch]

        # transform eigens 
        x = x.unsqueeze(-1)
        # x = self.eigen_encoder1(x, mask_full)
        # assert x[~mask_full].max() == 0
        #pos = self.eigen_encoder2(eigS_dense.unsqueeze(-1), mask_full)
        pos = 0 # ignore eigenvalues 

        # phi
        x = self.phi(x, data.edge_index, data.edge_attr, mask_full) + self.phi(-x, data.edge_index, data.edge_attr, mask_full)
        x = torch.sum(x, dim=1)
        # rho = Transformer
        #x = self.rho(x, pos=pos, mask=mask_full)

        return self.out(x)''' # n x n_hid

class SignNet2(nn.Module):
    """
        n x k node embeddings => n x n_hid 
        The output is sign invariant and permutation equivariant 
    """
    def __init__(self, n_hid, nl_phi, nl_rho=2, max_eigenvectors=35):
        super().__init__()
        self.max_eigenvectors = max_eigenvectors
        # MLPs for processing eigenvectors (one for each sign)
        self.eigen_encoder_pos = masked_layers.MaskedMLP(1, n_hid, nlayer=nl_phi)
        self.eigen_encoder_neg = masked_layers.MaskedMLP(1, n_hid, nlayer=nl_phi)

        # Positional encoding for eigenvalues
        self.pos_encoder = masked_layers.MaskedMLP(1, n_hid, nlayer=2)

        # SetTransformer for permutation equivariance
        self.out = nn.Sequential(nn.Linear(n_hid, n_hid, bias=False), nn.BatchNorm1d(n_hid))

    def reset_parameters(self):
        self.eigen_encoder_pos.reset_parameters()
        self.eigen_encoder_neg.reset_parameters()
        self.pos_encoder.reset_parameters()
        #self.rho.reset_parameters()

    def forward(self, data):
        eigS_dense, eigV_dense = to_dense_list_EVD(data.eigen_values, data.eigen_vectors, data.batch) 
        
        # Pad or truncate eigenvectors to have exactly `max_eigenvectors` per graph
        eigV_dense = eigV_dense[:, :self.max_eigenvectors]  # Truncate to max_eigenvectors
        if eigV_dense.size(1) < self.max_eigenvectors:
            padding = torch.zeros(eigV_dense.size(0), self.max_eigenvectors - eigV_dense.size(1), device=eigV_dense.device)
            eigV_dense = torch.cat([eigV_dense, padding], dim=1)
        
        x = eigV_dense.unsqueeze(-1)  # Add feature dimension

        # Pad or truncate eigenvalues to match the eigenvectors
        eigS_dense = eigS_dense[:, :self.max_eigenvectors]
        if eigS_dense.size(1) < self.max_eigenvectors:
            eigS_padding = torch.zeros(eigS_dense.size(0), self.max_eigenvectors - eigS_dense.size(1), device=eigS_dense.device)
            eigS_dense = torch.cat([eigS_dense, eigS_padding], dim=1)
        
        pos = eigS_dense.unsqueeze(-1)  # Add feature dimension

        original_size = scatter(torch.ones_like(data.eigen_values, dtype=torch.int64), data.batch, dim=0, reduce='sum')

        # Step 2: Create a range tensor to compare against the number of valid eigenvectors
        range_tensor = torch.arange(self.max_eigenvectors, device=eigV_dense.device).unsqueeze(0)  # Shape: (1, max_eigenvectors)

        # Step 3: Expand the original size to match the shape of the range tensor
        original_size_expanded = original_size.unsqueeze(1)  # Shape: (batch_size, 1)

        # Step 4: Create the mask by comparing range_tensor to original_size_expanded
        mask = range_tensor < original_size_expanded  # Shape: (batch_size, max_eigenvectors)

        # Step 5: Expand this mask to the full batch, considering all nodes in the batch
        mask_full = mask[data.batch] 

        # Process eigenvectors with MLPs
        x_pos = self.eigen_encoder_pos(x, mask_full)
        x_neg = self.eigen_encoder_neg(-x, mask_full)
        
        # Sum the outputs for sign invariance
        x = x_pos + x_neg

        # Positional encoding from eigenvalues
        #pos_encoded = self.pos_encoder(pos, mask_full)
        pos_encoded = 0

        # Apply SetTransformer with positional encoding
        x = torch.sum(x, dim=1)

        return self.out(x)


class SignNetGNN_S(nn.Module):
    def __init__(self, node_feat, edge_feat, n_hid, n_out, nl_signnet, nl_gnn):
        super().__init__()
        self.n_out = n_out
        self.sign_net = SignNet(n_hid, nl_signnet, nl_rho=1)
        self.gnn = GNN(node_feat, edge_feat, n_hid, n_out, nlayer=nl_gnn, gnn_type='GINConv') #GINEConv

    def reset_parameters(self):
        self.sign_net.reset_parameters()
        self.gnn.reset_parameters()

    def forward(self, data):
        pos = self.sign_net(data)
        return self.gnn(data, pos)
    

class SignNetGNN(nn.Module):
    def __init__(self, node_feat, edge_feat, n_hid_3d, n_hid, n_out, nl_signnet, nl_gnn):
        super().__init__()
        self.n_out = n_out
        self.sign_net = SignNet(n_hid_3d, nl_signnet, nl_rho=1, NOUT=n_hid)
        self.is_hid_neq = (n_hid != n_hid_3d)
        if self.is_hid_neq:
            self.out = nn.Sequential(nn.Linear(n_hid_3d, n_hid, bias=False), nn.BatchNorm1d(n_hid))
        self.gnn = GNN(node_feat, edge_feat, n_hid, n_out, nlayer=nl_gnn, gnn_type='GINConv') #GINEConv

    def reset_parameters(self):
        if self.is_hid_neq:
            for layer in self.out:
                layer.reset_parameters()
        self.sign_net.reset_parameters()
        self.gnn.reset_parameters()

    def forward(self, data, r):
        pos = self.sign_net(data, r)
        if self.is_hid_neq:
            for layer in self.out:
                pos = layer(pos)
        return self.gnn(data, pos)