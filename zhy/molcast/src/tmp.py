import torch
import numpy as np
import dgl
from torch import nn, optim
import torch.nn.functional as F
from layers import NodeEmbedding, EdgeEmbedding, RBFLayer, ReadLayer, DTNNLayer


class EnhancedDTNN(nn.Module):
    def __init__(self, dim_node=128, dim_edge=128, cutoff_low=0, cutoff_high=10, n_centers=30, n_conv=3, norm=False):
        """
        dim_node: e.g. 64, 128, etc.
        dim_edge: the same as dim_node
        default: dim_node = dim_edge
        """
        super(EnhancedDTNN, self).__init__()
        self.n_conv = n_conv
        self.norm = norm
        self.node_embedding = NodeEmbedding(dim_embedded=dim_node)  # 1 --> 128
        self.edge_embedding = EdgeEmbedding(dim_embedded=dim_edge)  # 1 --> 128
        self.rbf_embedding = RBFLayer(cutoff_low, cutoff_high,
                                      n_centers)  # (self, low=0, high=30, gap=0.1, dim=1) dim 是嵌入前的维度
        # self.reset_embedding_parameter() # 嵌入层Xavier初始化效果并不太好
        self.conv_layers = nn.ModuleList([DTNNLayer(dim_node, dim_edge + n_centers) for i in range(n_conv)])
        self.read_layer = ReadLayer(dim_in=dim_node, dim_hidden=dim_node)  # 对于每个节点特征进行线性变换（双层FC网络）
        self.reset_readlayer_parameter()
        self.reset_dtnn_parameter()

    def reset_readlayer_parameter(self):
        # nn.init.xavier_normal_(self.read_layer.fc1.weight, gain=nn.init.calculate_gain('relu'))
        # nn.init.xavier_normal_(self.read_layer.fc2.weight)
        # 发现uniform更好，但是normal更稳定
        nn.init.xavier_uniform_(self.read_layer.fc1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.read_layer.fc2.weight)

    def forward(self, g: dgl.DGLGraph):
        self.node_embedding(g, name='h')  # 类型嵌入并与电荷特征做 concatenate, g.ndata['feat'] size: [N, 128]
        self.edge_embedding(g, name='h')  # 类型嵌入, 'h' g.edage['h'] size: [M, 128]
        self.rbf_embedding(g)  # 距离的RBF嵌入, g.edata['rbf'] size: [M, K]
        if self.conv_type == 'dtnn':
            g.edata['h'] = torch.cat((g.edata['h'], g.edata['rbf']), 1)  # g.edata['feat'] size: [M, 128+K]
            for i in range(self.n_conv):
                self.conv_layers[i](g)
            self.read_layer(g)  # 最终得到各个节点标量特征 ‘E’/'h'
        elif self.conv_type == 'mgcn':
            g.ndata['h_0'] = g.ndata['h']
            for i in range(self.n_conv):
                self.conv_layers[i](g, i + 1)  # ndata['h_1'], ..., ndata['h_n_conv']
            node_feat = tuple(g.ndata['h_{}'.format(i)] for i in range(self.n_conv + 1))
            g.ndata['h'] = torch.cat(node_feat, dim=1)  # [N, dim]
            h = F.softplus(self.read_node_fc_1(g.ndata['h']), beta=1, threshold=20)  # [N, 64]
            h = self.read_node_fc_2(h)  # [N, ]
            g.ndata['h'] = h

        return dgl.sum_nodes(g, 'h').flatten()

    def reset_dtnn_parameter(self):
        """
        初始化权重参数，全部 ~ Normal Xavier 随机初始化
        """
        for i in range(self.n_conv):
            nn.init.xavier_normal_(self.conv_layers[i].fc_node_1.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.xavier_normal_(self.conv_layers[i].fc_edge_1.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.xavier_normal_(self.conv_layers[i].fc_node_2.weight)
            nn.init.xavier_normal_(self.conv_layers[i].fc_edge_2.weight)
            nn.init.xavier_normal_(self.conv_layers[i].fc_combine.weight, gain=nn.init.calculate_gain('tanh'))

    def reset_embedding_parameter(self):
        """
        初始化嵌入层参数
        """
        nn.init.xavier_normal_(self.node_embedding.embedding.weight)
        nn.init.xavier_normal_(self.edge_embedding.embedding.weight)

    def set_mean_std(self, mean: float, std: float):
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)
