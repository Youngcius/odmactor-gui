import torch
import numpy as np
import dgl
from torch import nn, optim
import torch.nn.functional as F
import sklearn
from torch.utils.data import DataLoader
from sklearn.base import BaseEstimator
import xgboost
from layers import NodeEmbedding, EdgeEmbedding, RBFLayer, ReadLayer, DTNNLayer
from dgl.nn.pytorch import GraphConv
import dgl.function as fn


#
# class MoleculePredictor(nn.Module, BaseEstimator):
#     '''
#     多种模型：GraphSAGE，GCN，MPNN，XGBoost
#     输入数据：均为 torch.Tensor (float32)
#     '''
#
#     def __init__(self, type='GraphSAGE'):
#         super(MoleculePredictor, self).__init__()
#         self.type = type
#         if self.type == 'GraphSAGE':
#             pass
#         elif self.type == 'GCN':
#             pass
#         elif self.type == 'XGBoost':
#             self.xgb = xgboost.XGBRegressor()
#         else:
#             print('Please input a correct model type!')
#
#     def forward(self, data: torch.Tensor, labels: torch.Tensor):
#         pass
#
#     def fit(self, X: np.ndarray, y: np.ndarray):
#         if self.type == 'XGBoost':
#             return self.xgb.fit(X, y)
#
#     def predict(self, X: np.ndarray):
#         if self.type == 'XGBoost':
#             return self.xgb.predict(X)


class MoleculePredictor(nn.Module):
    def __init__(self, dim_node=128, dim_edge=128, cutoff_low=0, cutoff_high=10, n_centers=30, n_conv=3,
                 conv_type='dtnn',
                 norm=False):
        """
        dim_node: e.g. 64, 128, etc.
        dim_edge: the same as dim_node
        default: dim_node = dim_edge
        """
        super(MoleculePredictor, self).__init__()
        self.n_conv = n_conv
        self.norm = norm
        self.conv_type = conv_type
        self.node_embedding = NodeEmbedding(dim_embedded=dim_node)  # 1 --> 128
        self.edge_embedding = EdgeEmbedding(dim_embedded=dim_edge)  # 1 --> 128
        self.rbf_embedding = RBFLayer(cutoff_low, cutoff_high,
                                      n_centers)  # (self, low=0, high=30, gap=0.1, dim=1) dim 是嵌入前的维度
        # self.reset_embedding_parameter() # 嵌入层Xavier初始化效果并不太好
        self.conv_layers = nn.ModuleList()
        if conv_type == 'dtnn':
            """
            键距离&类型，原子类型嵌入：效果并不好
            """
            for i in range(n_conv):
                self.conv_layers.append(DTNNLayer(dim_node, dim_edge + n_centers))
            self.read_layer = ReadLayer(dim_in=dim_node, dim_hidden=dim_node)  # 对于每个节点特征进行线性变换（双层FC网络）
            self.reset_readlayer_parameter()
            self.reset_dtnn_parameter()
        elif conv_type == 'gcn':
            # for i in range(n_conv):
            #     self.conv_layers.append(GraphConv(in_feats=dim_edge + n_centers, out_feats=dim_edge + n_centers))
            #     self.read_node_fc_1 = nn.Linear(dim_node * (self.n_conv + 1), 64)
            #     self.read_node_fc_2 = nn.Linear(64, 1)
            pass
        elif conv_type == 'schnet':
            pass
        elif conv_type == 'mgcn':
            # dim_node == dim_edge
            for i in range(n_conv):
                if dim_edge != dim_node:
                    raise ValueError('Please set dim_node == dim_edge!')
                self.conv_layers.append(MultiLevelInteraction(self.rbf_embedding.fan_out, dim_node))
                self.read_node_fc_1 = nn.Linear(dim_node * (self.n_conv + 1), 64)
                self.read_node_fc_2 = nn.Linear(64, 1)
        else:
            raise TypeError('Please input a correct type name for convolutional layer (dtnn, gcn, schnet or mgcn)')

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

        # graphs = dgl.unbatch(g)
        # if self.norm:
        # return torch.stack([dgl.sum_nodes(gi, 'h') for gi in graphs]).flatten() * self.std + self.mean
        # else:
        # return torch.stack([dgl.sum_nodes(gi, 'h') for gi in graphs]).flatten()
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

    # def reset_embedding_parameter(self):
    #     """
    #     初始化嵌入层参数
    #     """
    #     nn.init.xavier_normal_(self.node_embedding.embedding.weight)
    #     nn.init.xavier_normal_(self.edge_embedding.embedding.weight)

    # def set_mean_std(self, mean: float, std: float):
    #     self.mean = torch.tensor(mean)
    #     self.std = torch.tensor(std)


class MultiLevelInteraction(nn.Module):
    """
    The multilevel interaction in the MGCN model.
    """

    def __init__(self, rbf_dim, dim=64):
        super().__init__()

        self._atom_dim = dim

        self.activation = nn.Softplus(beta=0.5, threshold=14)

        self.node_layer1 = nn.Linear(dim, dim, bias=True)
        self.edge_layer1 = nn.Linear(dim, dim, bias=True)
        self.conv_layer = VEConv(rbf_dim, dim)
        self.node_layer2 = nn.Linear(dim, dim)
        self.node_layer3 = nn.Linear(dim, dim)

    def forward(self, g, level=1):
        g.ndata["h_new"] = self.node_layer1(g.ndata["h_{}".format(level - 1)])  # 前一层节点特征的线性变换
        node = self.conv_layer(g)  # 在卷积中默认更新了 edata['h'] (线性变换) # node 即 g.ndata['h_new']
        g.edata["h"] = self.activation(self.edge_layer1(g.edata["h"]))  # 边特征更新、激活
        node_1 = self.node_layer2(node)
        node_1a = self.activation(node_1)
        new_node = self.node_layer3(node_1a)

        g.ndata["h_{}".format(level)] = g.ndata["h_{}".format(level - 1)] + new_node

        return g.ndata['h_{}'.format(level)]


class VEConv(nn.Module):
    """
    The Vertex-Edge convolution layer in MGCN which take edge & vertex features
    in consideratoin at the same time.
    """

    def __init__(self, rbf_dim, dim=64, update_edge=True):
        """
        Args:
            rbf_dim: the dimension of the RBF layer
            dim: the dimension of linear layers
            update_edge: whether update the edge emebedding in each conv-layer
        """
        super().__init__()
        self._rbf_dim = rbf_dim
        self._dim = dim
        self._update_edge = update_edge

        self.linear_layer1 = nn.Linear(self._rbf_dim, self._dim)  # NN(d_ij)
        self.linear_layer2 = nn.Linear(self._dim, self._dim)  # NN(d_ij)
        self.linear_layer3 = nn.Linear(self._dim, self._dim)

        self.activation = nn.Softplus(beta=0.5, threshold=14)

    def update_rbf(self, edges):
        """
        
        """
        rbf = edges.data["rbf"]  # rbf不更新，但需要用到rbf的线性变化来做节点卷积
        h = self.linear_layer1(rbf)
        h = self.activation(h)
        h = self.linear_layer2(h)
        return {"h": h}

    def update_edge(self, edges):
        edge_f = edges.data["h"]
        h = self.linear_layer3(edge_f)
        return {"h": h}

    def forward(self, g: dgl.DGLGraph):
        g.apply_edges(self.update_rbf)  # edata['h']
        if self._update_edge:
            g.apply_edges(self.update_edge)  #
        # new_node 待卷积层的线性变换
        # g.update_all(message_func=[
        #     fn.u_mul_e("h_new", "h", "m_0"),
        #     fn.copy_e("h", "m_1")
        # ],
        #              reduce_func=[
        #                  fn.sum("m_0", "h_new_0"),
        #                  fn.sum("m_1", "h_new_1")
        #              ])
        g.update_all(fn.u_mul_e('h_new', 'h', 'm_0'), fn.sum('m_0', 'h_new_0'))
        g.update_all(fn.copy_e('h', 'm_1'), fn.sum('m_1', 'h_new_1'))

        g.ndata["h_new"] = g.ndata.pop("h_new_0") + g.ndata.pop("h_new_1")

        return g.ndata["h_new"]


# ###############################


###################################


def train(graph_model: nn.Module, dataloader: DataLoader, optimizer, idx_label=0):
    """
    每一个 epoch 中进行 train()
    """
    loss_func = nn.MSELoss()
    mae_func = nn.L1Loss()
    loss_train = []
    mae_train = []
    for i, (graphs, labels) in enumerate(dataloader):
        labels = labels[:, idx_label]
        predict = graph_model(graphs)
        # print('batch {}, label: {}'.format(i, labels.shape))
        # print('batch {}, predict: {}'.format(i, predict.shape))
        loss = loss_func(predict, labels)
        # print('batch {}, loss: {}'.format(i, loss))

        mae = mae_func(predict.detach(), labels.detach())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_train.append(loss.item())
        mae_train.append(mae.item())
        if (i + 1) % 100 == 0:
            print('\t\tbatch: {}, loss: {:.4f}, MAE: {:.4f}'.format(i + 1, loss.item(), mae.item()))
    loss_train = np.mean(loss_train)
    mae_train = np.mean(mae_train)
    print('Training:\t loss: {:.4f}, MAE: {:.4f}'.format(loss_train, mae_train))
    return loss_train, mae_train


def test(graph_model: nn.Module, dataloader: DataLoader, idx_label=0):
    """
    每一个 epoch 中进行 test()
    """
    loss_func = nn.MSELoss()
    mae_func = nn.L1Loss()
    loss_test = []
    mae_test = []
    with torch.no_grad():
        for i, (graphs, labels) in enumerate(dataloader):
            labels = labels[:, idx_label]
            predict = graph_model(graphs)
            loss = loss_func(predict, labels)
            mae = mae_func(predict.detach(), labels.detach())
            loss_test.append(loss.item())
            mae_test.append(mae.item())
        loss_test = np.mean(loss_test)
        mae_test = np.mean(mae_test)
        print('Evaluating:\t loss: {:.4f}, MAE: {:.4f}'.format(loss_test, mae_test))

    return loss_test, mae_test
