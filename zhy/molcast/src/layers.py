import torch
from torch import nn
import numpy as np
import dgl
import torch.nn.functional as F


class NodeEmbedding(nn.Module):
    def __init__(self, dim_dict=20, dim_embedded=128, pre_train=None):
        super(NodeEmbedding, self).__init__()
        if pre_train is not None:
            self.embedding = nn.Embedding.from_pretrained(pre_train, padding_idx=0)
        else:
            self.embedding = nn.Embedding(dim_dict, dim_embedded - 1, padding_idx=0)

    def forward(self, g, name='feat'):
        # g.ndata['Z']: [num_nodes, ]
        embedded = self.embedding(g.ndata['Z'])  # [num_nodes, 127]
        g.ndata[name] = torch.cat((embedded, g.ndata['C'].view(-1, 1)), dim=(embedded.dim() - 1))  # dim=1


class EdgeEmbedding(nn.Module):
    """
    Convert the edge to embedding.
    The edge links same pair of atoms share the same initial embedding.
    """

    def __init__(self, dim_dict=400, dim_embedded=128, pre_train=None):
        # def __init__(self, dim=128, edge_num=3000, pre_train=None):
        """
        Randomly init the edge embeddings.
        Args:
            dim: the dim of embeddings
            edge_num: the maximum type of edges
            pre_train: the pre_trained embeddings
        """
        super(EdgeEmbedding, self).__init__()
        # self._dim = dim
        # self._edge_num = edge_num
        if pre_train is not None:
            self.embedding = nn.Embedding.from_pretrained(pre_train, padding_idx=0)
        else:
            # self.embedding = nn.Embedding(edge_num, dim, padding_idx=0)
            self.embedding = nn.Embedding(dim_dict, dim_embedded, padding_idx=0)

    def forward(self, g, name="feat"):
        g.edata[name] = self.embedding(g.edata["type"])  # 边嵌入
        # return g.edata[name]


class RBFLayer(nn.Module):
    """
    Radial basis functions Layer.
    e(d) = exp(- gamma * ||d - mu_k||^2)
    default settings:
        gamma = 10
        0 <= mu_k <= 30 for k=1~300
    """

    def __init__(self, low, high, n_centers=30, dim=1):
        super(RBFLayer, self).__init__()
        self.centers = torch.linspace(low, high, n_centers)
        self.centers = nn.Parameter(self.centers, requires_grad=False) ########################
        self.gap = self.centers[1] - self.centers[0]
        # self._fan_out = self.dim * self.n_center
        self.fan_out = dim * n_centers

    # def __init__(self, low=0, high=30, gap, dim=1):
    #     # 默认会嵌入得到 M×K 的边特征
    #     super().__init__()
    #     self._low = low
    #     self._high = high
    #     self._gap = gap
    #     self._dim = dim
    #
    #     self._n_centers = int(np.ceil((high - low) / gap))  # 就是维度 k 值
    #     # centers = np.linspace(low, high, self._n_centers)
    #     # self.centers = th.tensor(centers, dtype=th.float, requires_grad=False)
    #     self.centers = torch.linspace(low, high, self._n_centers)
    #     self.centers = nn.Parameter(self.centers, requires_grad=False)  # 径向基函数中心值变为可训练的
    #     self._fan_out = self._dim * self._n_centers
    #
    #     self._gap = self.centers[1] - self.centers[0]

    def dis2rbf(self, edges):
        # dist = edges.data["dist"]
        # radial = dist - self.centers
        # coef = -1 / self._gap
        coef = -1 / self.gap
        rbf = torch.exp(coef * ((edges.data['dist'].view(-1, 1) - self.centers.view(1, -1)) ** 2))
        return {"rbf": rbf}

    def forward(self, g):
        """Convert distance scalar to rbf vector"""
        g.apply_edges(self.dis2rbf)
        return g.edata["rbf"]


class ReadLayer(nn.Module):
    def __init__(self, dim_in, dim_hidden):
        super(ReadLayer, self).__init__()
        self.fc1 = nn.Linear(dim_in, dim_hidden)
        self.fc2 = nn.Linear(dim_hidden, 1)

    def forward(self, g: dgl.DGLGraph):
        g.apply_nodes(self.linear_trans)  # 不同图的节点数目不一样，单节点特征的维度一样
        # return dgl.sum_nodes(g, 'E')

    def linear_trans(self, nodes):
        # h1 = torch.tanh(self.fc1(nodes.data['h']))
        h1 = torch.relu(self.fc1(nodes.data['h'])) # ReLU效果更好, tanh难收敛
        h2 = self.fc2(h1)
        return {'h': h2}


class DTNNLayer(nn.Module):
    def __init__(self, dim_node, dim_edge, norm=None):
        super(DTNNLayer, self).__init__()
        self.fc_node_1 = nn.Linear(dim_node, 128)
        self.fc_node_2 = nn.Linear(128, 128)
        self.fc_edge_1 = nn.Linear(dim_edge, 128)
        self.fc_edge_2 = nn.Linear(128, 128)
        self.fc_combine = nn.Linear(128, dim_node)
        self.fc_update_edge = nn.Linear(dim_node, dim_edge,bias=False)

    def forward(self, g: dgl.DGLGraph):
        # g.ndata['h'] : [N, dim_node]
        g.update_all(self.msg_func, self.reduce_func)
        g.apply_edges(self.update_edges)

    def update_edges(self, edges):
        h = 0.8*edges.data['h'] + (1-0.8) * self.fc_update_edge(edges.src['h'] * edges.dst['h'])
        return {'h': h}

    def msg_func(self, edges):
        # print('edges.src[h]', edges.src['h'].size())
        # print('edges.data[feat]',edges.data['feat'].size())

        m1 = self.fc_node_2(F.relu(self.fc_node_1(edges.src['h'])))  # [M,dim_node] --> [M,256]
        m2 = self.fc_edge_2(F.relu(self.fc_edge_1(edges.data['h'])))  # [M,dim_edge] --> [M,256]
        # print('m1',m1.size())
        # print('m2',m2.size())
        m = torch.tanh(self.fc_combine(m1 * m2))

        # print('m',m.size())
        return {'m': m}

    def reduce_func(self, nodes):
        return {'h': nodes.mailbox['m'].sum(dim=1) + nodes.data['h']}


