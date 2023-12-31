import torch
import torch.nn.functional as F
from torch import device
from torch.nn import ModuleList, Linear
from torch_geometric.nn import GCNConv
from torch_geometric.nn.pool import global_mean_pool

class GCN(torch.nn.Module):

    def __init__(self,
            in_channels: int = None,
            hidden_channels: int = 32,
            out_channels: int = None,
            pooling_fn: callable = global_mean_pool,
            device: device = None,
            nb_layers: int = 3,):
        super(GCN, self).__init__()

        self.pooling_fn = pooling_fn
        self.device = device
        self.to(device)

        torch.manual_seed(12345)
        self.convs = ModuleList()
        for layer in range(nb_layers):
            if layer == 0:
                self.convs.append(GCNConv(in_channels, hidden_channels))
            else:
                self.convs.append(GCNConv(hidden_channels, hidden_channels))

        self.linear = Linear(hidden_channels, out_channels)
        self.embeddings = None

    def forward(self, x, edge_index, batch):
        x = x.to(dtype=torch.float32)

        for index, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if index != len(self.convs) - 1:
                x = x.relu()

        embeddings = self.pooling_fn(x, batch)
        self.embeddings = embeddings

        embeddings = F.dropout(embeddings, p=0.5, training=self.training)
        result = self.linear(embeddings)

        return result

    def reset_parameters(self):
        for layer in self.convs:
            layer.reset_parameters()
        self.linear.reset_parameters()

class GCN_2(GCN):
    def __init__(
            self,**kwargs,
    ):
        super().__init__(nb_layers=2, **kwargs)

class GCN_3(GCN):
    def __init__(
            self,
            **kwargs,
    ):
        super().__init__(nb_layers=3, **kwargs)