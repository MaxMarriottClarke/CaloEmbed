"""EdgeConv over a static kNN graph built in a learned latent space.

Port of the reference contrastive Net from
geant4sim/scripts/training/Contrastive/src/model.py, with the input width
configurable (the reference hardcoded 5 features) and the embedding size
named out_dim instead of contrastive_dim.
"""

import torch.nn as nn
import torch.nn.functional as F
from torch_cluster import knn_graph
from torch_geometric.nn import EdgeConv

from . import register_model


@register_model("edgeconv")
class LatentKNNEdgeConv(nn.Module):
    """Encode nodes → build kNN graph in that latent space once → run
    EdgeConv message passing on the fixed graph → project to embedding.
    """

    def __init__(self, in_dim: int, hidden_dim: int = 64, num_layers: int = 4,
                 dropout: float = 0.3, out_dim: int = 8, k: int = 20):
        super().__init__()
        self.k = k
        self.dropout = dropout

        self.lc_encode = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
        )

        def build_mlp():
            return nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.ELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )

        self.convs = nn.ModuleList([
            EdgeConv(nn=build_mlp(), aggr="max")
            for _ in range(num_layers)
        ])

        self.output = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ELU(),
            nn.Dropout(p=dropout),
            nn.Linear(64, 32),
            nn.ELU(),
            nn.Dropout(p=dropout),
            nn.Linear(32, out_dim),
        )

    def forward(self, data):
        x = self.lc_encode(data.x)

        # kNN is a non-differentiable index selection; detach + fp32 keeps
        # torch_cluster happy under AMP without changing gradients.
        edge_index = knn_graph(x.detach().float(), k=self.k, batch=data.batch, loop=False)

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        return self.output(x)
