import numpy as np
import torch
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F

from models.layers import GraphConvolution, MLP


device = f"cuda:0" if torch.cuda.is_available() else "cpu"
device = torch.device(device)


class GCN(nn.Module):
    def __init__(
        self,
        nfeat,
        nhid,
        nclass,
        nlayers,
        nnodes,
        dropout,
        model_type,
        structure_info,
        variant=False,
        # init_layers_X=1,
    ):
        super(GCN, self).__init__()
        self.gcns, self.mlps = nn.ModuleList(), nn.ModuleList()
        self.model_type, self.structure_info, self.nlayers, self.nnodes = (
            model_type,
            structure_info,
            nlayers,
            nnodes,
        )
        self.gcns.append(
            GraphConvolution(
                nfeat,
                nhid,
                nnodes,
                model_type=model_type,
                variant=variant,
                structure_info=structure_info,
            )
        )
        self.gcns.append(
            GraphConvolution(
                1 * nhid,
                nclass,
                nnodes,
                model_type=model_type,
                output_layer=1,
                variant=variant,
                structure_info=structure_info,
            )
        )

        self.dropout = dropout

    # def forward(self, x, adj_low, adj_high, adj_low_unnormalized):
    def forward(self, x, adj_low, adj_high,adj_low_unnormalized):
       
        x = F.dropout(x, self.dropout, training=self.training)
        fea1 = self.gcns[0](x, adj_low, adj_high, adj_low_unnormalized)
        fea1 = F.dropout((F.relu(fea1)), self.dropout, training=self.training)
        fea2 = self.gcns[1](fea1, adj_low, adj_high, adj_low_unnormalized)
  
        return fea2

