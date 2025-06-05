import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv


class GCNLinkPred(torch.nn.Module):
    def __init__(self, gcn_in, gcn_hidden=128, gcn_out=32, gcn_layers=2,
                 link_hidden=64, link_out=1, link_layers=1, dropout=0.3):
        super(GCNLinkPred, self).__init__()
        self.gcn = GCN(gcn_in, gcn_hidden, gcn_out, gcn_layers, dropout)
        self.link_predictor = LinkPredictor(
            gcn_out, link_hidden, link_out, link_layers, dropout)

    def forward(self, x, edge_index, edge_label_index):
        node_emb = self.gcn(x, edge_index)

        x_i = node_emb[edge_label_index[0]]
        x_j = node_emb[edge_label_index[1]]

        return self.link_predictor(x_i, x_j).squeeze()


class SAGELinkPred(torch.nn.Module):
    def __init__(self, sage_in, sage_hidden=128, sage_out=32, sage_layers=2,
                 link_hidden=64, link_out=1, link_layers=1, dropout=0.3, residual=True):
        super(SAGELinkPred, self).__init__()
        self.sage = SAGE(sage_in, sage_hidden, sage_out,
                         sage_layers, dropout, residual)
        self.link_predictor = LinkPredictor(
            sage_out, link_hidden, link_out, link_layers, dropout)

    def forward(self, x, adj_t, edge_label_index):
        node_emb = self.sage(x, adj_t)

        x_i = node_emb[edge_label_index[0]]
        x_j = node_emb[edge_label_index[1]]

        return self.link_predictor(x_i, x_j).squeeze()


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(GCN, self).__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")
        if dropout < 0 or dropout > 1:
            raise ValueError("dropout must be between 0 and 1")

        self.convs = torch.nn.ModuleList()
        self.dropout = dropout

        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.convs[-1](x, edge_index)
        return x


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, residual):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(
            SAGEConv(in_channels, hidden_channels, root_weight=residual))
        for _ in range(num_layers - 2):
            self.convs.append(
                SAGEConv(hidden_channels, hidden_channels, root_weight=residual))
        self.convs.append(
            SAGEConv(hidden_channels, out_channels, root_weight=residual))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x


class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(LinkPredictor, self).__init__()

        self.lins = torch.nn.ModuleList()
        # MLP
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j):
        # hadamard product
        x = x_i * x_j
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)
