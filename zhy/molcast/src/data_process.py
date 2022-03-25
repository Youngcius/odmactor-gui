import dgl.function as fn
from dgl.nn import SAGEConv
from dgl.utils import expand_as_pair, check_eq_shape
from dgl.data.utils import save_info, load_info
import scipy.stats as sps
import matplotlib.pyplot as plt
import xgboost
import sklearn
from sklearn.model_selection import train_test_split
from PIL import Image
from ase.io import write
import logging
import os
import re
import shutil
import tarfile
import tempfile
import torch
from torch import nn, optim
import dgl
from dgl.data import DGLDataset
from dgl.data import split_dataset
import torch.nn.functional as F
import numpy as np
from ase.io.extxyz import read_xyz, write_xyz
from ase.io.png import write_png
from ase.units import Debye, Bohr, Hartree, eV
import schnetpack as spk
from schnetpack.datasets import DownloadableAtomsData, QM9
from ase.visualize import view
from rdkit import Chem
from rdkit.Chem import AllChem, Draw


# 读取单个文件
def read_data(fname: str, fmt: str):
    if fmt == 'xyz':
        ######################
        # 基于QM9数据只实现该部分
        ######################
        pass
    elif fmt == 'mol':
        pass
    elif fmt == 'smile':
        pass


# 保存单个文件
def save_data(fname: str, fmt: str):
    if fmt == 'xyz':
        pass
    elif fmt == 'mol':
        pass
    elif fmt == 'smiles':
        pass


def batcher():
    def batcher_dev(batch):
        graphs, labels = zip(*batch)
        labels = torch.stack(labels, 0)
        batch_graphs = dgl.batch(graphs)
        # 保存批次信息
        num_batch_nodes = []
        num_batch_edges = []
        for g in graphs:
            num_batch_nodes.append(g.num_nodes())
            num_batch_edges.append(g.num_edges())
        num_batch_nodes = torch.LongTensor(num_batch_nodes)
        num_batch_edges = torch.LongTensor(num_batch_edges)
        batch_graphs.set_batch_num_nodes(num_batch_nodes)
        batch_graphs.set_batch_num_edges(num_batch_edges)
        return batch_graphs, labels
        # return AlchemyBatcher(graph=batch_graphs, label=labels)

    return batcher_dev


class QM9Dataset(DGLDataset):
    # 原始文本第二行为"scalar"类型的属性，共17个，后15个有意义
    # 能量参数的单位都是 Hatree
    A = "rotational_constant_A"  # 转动频率：GHz
    B = "rotational_constant_B"
    C = "rotational_constant_C"
    # ----------------------------
    mu = "dipole_moment"
    alpha = "isotropic_polarizability"  # 各向同性极化率
    homo = "homo"  # 最高占据分子轨道
    lumo = "lumo"  # 最低未占据分子轨道（lumo > homo）
    gap = "gap"  # lumo - homo
    r2 = "electronic_spatial_extent"  # 电子空间距离度量: a0^2
    zpve = "zpve"  # 零点振动能
    U0 = "energy_U0"  # OK 内能
    U = "energy_U"  # RT 内能
    H = "enthalpy_H"  # RT 焓
    G = "free_energy"  # RT 吉布斯函数
    Cv = "heat_capacity"  # RT 热容

    reference = {zpve: 0, U0: 1, U: 2, H: 3, G: 4, Cv: 5}
    # prop_names = ['A', 'B', 'C', 'mu', 'alpha', 'homo',
    #   'lumo', 'gap', 'r2', 'zpve', 'U0', 'U', 'H', 'G', 'Cv']

    prop_names = [A, B, C, mu, alpha, homo,
                  lumo, gap, r2, zpve, U0, U, H, G, Cv]
    units = [1.0, 1.0, 1.0, Debye, Bohr ** 3, Hartree, Hartree, Hartree,
             Bohr ** 2, Hartree, Hartree, Hartree, Hartree, Hartree, 1.0]

    # def __init__(self, url=None, raw_dir=None, save_dir=None, fore_reload=None, verbose=False):

    def __init__(self, raw_dir='../data/'):
        # super(QM9Dataset, self).__init__(name='qm9', url=url, raw_dir=raw_dir,
        #  save_dir=save_dir, fore_reload=fore_reload, verbose=verbose)
        super(QM9Dataset, self).__init__(name='qm9', raw_dir=raw_dir)

        pass

    def process(self):
        # process raw data to graphs, labels, splitting masks
        # 假定原始数据已经存在self.raw_dir
        # 该函数在类的实例化时自动执行
        # if self.has_cache():
        #     self.load()
        # else:
        self.dict = self._load_data()
        self.graphs, self.labels = self._load_graphs()

    def _load_data(self):
        Smiles = []
        InChI = []
        num_atoms = []  # 每个分子包含的原子个数
        # properties = {pn: [] for p in prop_names}# 属性都存在字典数据中
        properties = []  # 暂时用列表（数组）存储属性值，每个元素为长度15的列表
        # coordinates = [] # element size： [n, 3] ndarray
        # Z_numbers = [] # 原子序数和空间坐标由 atom object 中的 numbers 和 positions 属性获得
        charges = []  # element length： n
        harmonics = []  # 自由度：3n-6
        all_atoms = []
        data_path = os.path.join(self.raw_dir, 'dsgdb9nsd.xyz')
        fnames = [os.path.join(data_path, fname)
                  for fname in os.listdir(data_path)]
        tmpdir = tempfile.mkdtemp("gdb9")  # 临时目录为利用 read_xyz 读取xyz文件
        # print('file numebr:', len(fnames))
        for i, fname in enumerate(fnames):
            ##################
            # if i > 1000:
            #     break
            # if (i+1)%50==0:
            #     print('Parsed: {:6d} / 1000'.format(i+1))
            ##################
            if (i + 1) % 10000 == 0:
                # logging.info("Parsed: {:6d} / 133885".format(i + 1))
                print("Parsed: {:6d} / 133885".format(i + 1))
            tmp = os.path.join(tmpdir, "tmp.xyz")

            with open(fname, "r") as f:
                lines = f.readlines()
                lines = [line.replace("*^", "e") for line in lines]
                # atom number in every molecule
                num_atoms.append(int(lines[0]))
                harmonics.append(list(map(float, lines[-3].split())))
                Smiles.append(lines[-2].split())
                InChI.append(lines[-1].split())
                # l = lines[1].split()[2:]
                properties.append(
                    list(map(float, lines[1].split()[2:])))  # 后15个属性值
                # 坐标&电荷数据
                coord = np.array([])
                charge = []
                for i in range(2, 2 + int(lines[0])):
                    charge.append(float(lines[i].split()[-1]))
                    # coord = np.append(coord, list(map(float, l[1:-1])))
                charges.append(charge)
                # corrdinates.append(coord)
                with open(tmp, "wt") as fout:
                    for line in lines:
                        # fout.write(line.replace("*^", "e"))
                        fout.write(line)

            with open(tmp, "r") as f:
                ats = list(read_xyz(f, 0))[0]  # atoms object (ASE)
            all_atoms.append(ats)

        shutil.rmtree(tmpdir)  # 删除 gdb9 临时目录
        print('parsing 已经完成!')
        return {'fnames': fnames, 'Smiles': Smiles, 'InChI': InChI, 'properties': properties,
                'charges': charges, 'all_atoms': all_atoms, 'harmonics': harmonics, 'num_atoms': num_atoms}

    def _load_graphs(self):
        labels = torch.tensor(self.dict['properties']).float()
        # 暂时不需要 self.dict['harmonics'][i] 作为标签
        graphs = []
        num_graphs = len(labels)
        # with open('debug.txt', 'a') as f:

        for i in range(num_graphs):
            # f.write(str(i) + '\n')
            if (i+1)%10000==0:
                print('Graphs created: {:6d} / 133885'.format(i+1))
            ats = self.dict['all_atoms'][i]
            # 需要根据 A 合电荷构造库伦矩阵吗????????
            # mol = Chem.AddHs(Chem.MolFromInchi(self.dict['InChI'][i][0]))
            mol = Chem.AddHs(Chem.MolFromSmiles(self.dict['Smiles'][i][0]))

            A = Chem.rdmolops.GetAdjacencyMatrix(mol)  # ndarray 邻接矩阵
            atoms_info = [(atom.GetIdx(), atom.GetAtomicNum(), atom.GetSymbol())
                          for atom in mol.GetAtoms()]  # idx, Z, Symbol
            bonds_info = [(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) for bond in mol.GetBonds(
            )] + [(bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()) for bond in mol.GetBonds()]
            g = dgl.graph(bonds_info)
            # 构造节点初始特征
            # g.ndata['X'] = torch.tensor(ats.positions).float()  # positions
            g.ndata['C'] = torch.tensor(
                self.dict['charges'][i]).float()  # charges
            g.ndata['Z'] = torch.tensor(ats.numbers).long()  # Z
            # 构造 边 初始特征
            g.edata['dist'] = torch.empty(g.num_edges())
            g.edata['type'] = torch.empty(g.num_edges()).long()
            for i, (u, v) in enumerate(zip(g.edges()[0], g.edges()[1])):
                # (u, v) 的特征,即第 i 条边, size: torch.Size([])
                Z_u, Z_v = g.ndata['Z'][u].item(), g.ndata['Z'][v].item()
                # g.edata['dist'][i] = torch.norm(g.ndata['X'][u] - g.ndata['X'][v])
                g.edata['dist'][i] = np.linalg.norm(ats.positions[u] - ats.positions[v])
                g.edata['type'][i] = Z_u * Z_v + (np.abs(Z_u - Z_v) - 1) ** 2 / 4 # 自动转换为long类型

            graphs.append(g)

        return graphs, labels

    def get_prop_stat(self) -> dict:
        """
        Returns:
            nobs : int or ndarray of ints
               Number of observations (length of data along `axis`).
               When 'omit' is chosen as nan_policy, each column is counted separately.
            minmax: tuple of ndarrays or floats
               Minimum and maximum value of data array.
            mean : ndarray or float
               Arithmetic mean of data along axis.
            variance : ndarray or float
               Unbiased variance of the data along axis, denominator is number of
               observations minus one.
            skewness : ndarray or float
               Skewness, based on moment calculations with denominator equal to
               the number of observations, i.e. no degrees of freedom correction.
            kurtosis : ndarray or float
               Kurtosis (Fisher).  The kurtosis is normalized so that it is
               zero for the normal distribution.  No degrees of freedom are used.
        """
        return sps.describe(self.dict['properties'])._asdict()

    # def get_type_atoms_with_idx(self,idx)->list:
    #     g = self.graphs[idx]
    #     g.ndata
    #
    #     return
    def get_dist_min_max_with_idx(self, idx) -> tuple:
        dist = self.graphs[idx].edata['dist']
        return (dist.min().item(), dist.max().item())

    def save(self):
        # if self.has_cache():
        #     return
        # if not os.path.exists(self.save_path):
        #     os.mkdir(self.save_path)
        # 保存图和标签到 self.save_path
        graph_path = os.path.join(self.save_path, 'dgl_graph.bin')
        dgl.save_graphs(graph_path, self.graphs, {'labels': self.labels})
        # 在Python字典里保存其他信息
        info_path = os.path.join(self.save_path, 'info.pkl')
        save_info(info_path, self.dict)
        print('....saved...')

    def load(self):
        # 从目录 `self.save_path` 里读取处理过的数据
        graph_path = os.path.join(self.save_path, 'dgl_graph.bin')
        self.graphs, label_dict = dgl.load_graphs(graph_path)
        self.labels = label_dict['labels']
        info_path = os.path.join(self.save_path, 'info.pkl')
        self.dict = load_info(info_path)
        print('....loaded...')

    def has_cache(self):
        # 检查在 `self.save_path` 里是否有处理过的数据文件
        graph_path = os.path.join(self.save_path, 'dgl_graph.bin')
        info_path = os.path.join(self.save_path, 'info.pkl')
        return os.path.exists(graph_path) and os.path.exists(info_path)

    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx]

    def __len__(self):
        return len(self.graphs)

    @property
    def num_labels(self):
        return 15
