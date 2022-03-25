from torchvision import datasets
import networkx as nx
from dgl.data import RedditDataset, QM7bDataset
import dgl.function as fn
from dgl.nn import SAGEConv
from dgl.utils import expand_as_pair, check_eq_shape
import matplotlib.pyplot as plt
import xgboost
import sklearn
from sklearn.model_selection import train_test_split
from PIL import Image
from ase.io import write
import IPython
import logging
import os
import re
import shutil
import tarfile
import tempfile
from urllib import request
import torch
from torch import nn, optim
import dgl
from dgl.data import DGLDataset
from dgl.dataloading import DataLoader
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

#################################

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

prop_names = [A, B, C, mu, alpha, homo, lumo, gap, r2, zpve, U0, U, H, G, Cv]
units = [1.0, 1.0, 1.0, Debye, Bohr ** 3, Hartree, Hartree, Hartree,
         Bohr ** 2, Hartree, Hartree, Hartree, Hartree, Hartree, 1.0]


# raw_path = '../data/dsgdb9nsd.xyz'
# ordered_files = sorted(os.listdir(raw_path), key=lambda x: (int(re.sub("\D", "", x)), x))
# all_properties = []
# irange = np.arange(len(ordered_files), dtype=np.int) # 0 -- 133885
Smiles = []
InChI = []
atom_nums = []
# properties = {pn: [] for p in prop_names}# 属性都存在字典数据中
properties = []  # 暂时用列表（数组）存储属性值，每个元素为长度15的列表
# coordinates = [] # element size： [n, 3] ndarray
# Z_numbers = [] # 原子序数和空间坐标由 atom object 中的 numbers 和 positions 属性获得
charges = []  # element length： n
harmonics = []  # 自由度：3n-6
all_atoms = []
data_path = '../data/dsgdb9nsd.xyz/'
fnames = [os.path.join(data_path, fname) for fname in os.listdir(data_path)]
print('file numebr:', len(fnames))

tmpdir = tempfile.mkdtemp("gdb9")  # 临时目录为利用 read_xyz 读取xyz文件

# parsing
for i, fname in enumerate(fnames):
    ##################
    if i > 1000:
        break
    ##################
    if (i + 1) % 10000 == 0:
        # logging.info("Parsed: {:6d} / 133885".format(i + 1))
        print("Parsed: {:6d} / 133885".format(i + 1))
    tmp = os.path.join(tmpdir, "tmp.xyz")

    with open(fname, "r") as f:
        lines = f.readlines()
        lines = [line.replace("*^", "e") for line in lines]
        atom_nums.append(int(lines[0]))  # atom number in every molecule
        harmonics.append(list(map(float, lines[-3].split())))
        Smiles.append(lines[-2].split())
        InChI.append(lines[-1].split())
        # l = lines[1].split()[2:]
        properties.append(list(map(float, lines[1].split()[2:])))  # 后15个属性值
        # 坐标&电荷数据
        coord = np.array([])
        charge = []
        for i in range(2, 2+int(lines[0])):
            charge.append(float(lines[i].split()[-1]))
            # coord = np.append(coord, list(map(float, l[1:-1])))
        charges.append(charge)
        # corrdinates.append(coord)
        with open(tmp, "wt") as fout:
            for line in lines:
                fout.write(line)
    with open(tmp, "r") as f:
        ats = list(read_xyz(f, 0))[0]  # atoms object (ASE)
    all_atoms.append(ats)

shutil.rmtree(tmpdir)  # 删除 gdb9 临时目录




energy = [prop[10] for prop in properties]

plt.scatter(atom_nums, energy)


type(torch.randn(10).std().item())
torch.tensor(10)

a.tolist()

F.mse_loss(a.float(),a.tolist())
F.mse_loss(b,b).item()
F.mse_loss([1,2],[1,1])



_=100











######################################################
ats = all_atoms[100]
ats.symbols
ats.info
print(ats)
dir(ats)

ts = torch.tensor([1,2,3,3,3,4])
ts.unique()
np.count_nonzero(ts.unique())


ats = all_atoms[21725]
mol = Chem.AddHs(Chem.MolFromInchi(InChI[21725][0]))
mol2 = Chem.AddHs(Chem.MolFromSmiles(Smiles[21725][0]))
mol2
mol
ats.positions
mol.GetNumAtoms()
mol
with open(fnames[100],mode='r') as f:
    txt = f.read()
print(txt)

from plotnine.data import meat, mtcars
meat.describe()
arr = mtcars.iloc[:,1:].to_numpy()
import scipy.stats as sps
desc = sps.describe(mtcars.iloc[:,1:].to_numpy())._asdict()
mtcars.describe()


for k,v in desc.items():
    print(k,'\t', np.round(v,2))

mtcars.describe()

import pandas as pd
desc
?sps.describe
print(desc)
desc._asdict()

iris
ats = all_atoms[100]
mol = Chem.AddHs(Chem.MolFromSmiles(Smiles[100][0]))


A = Chem.rdmolops.GetAdjacencyMatrix(mol)  # ndarray 邻接矩阵
atoms_info = [(atom.GetIdx(), atom.GetAtomicNum(), atom.GetSymbol())
                for atom in mol.GetAtoms()]  # idx, Z, Symbol
bonds_info = [(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) for bond in mol.GetBonds(
)] + [(bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()) for bond in mol.GetBonds()]
g = dgl.graph(bonds_info)
nx.draw(g.to_networkx(),with_labels=True)
plt.title('Smiles Symbol: CC1CC1O')
plt.savefig('nx.png',dpi=400)
a,b=np.asarray([(1,2),(3,4)]).max(0)
t1 = torch.randn(3,4)
t2 = torch.rand(3,2)
torch.cat((t1,t2),1)

Draw.MolToFile('mol.')

type(all_atoms[0])
type(mol)
type(g)
sps.describe([[1,2,3],[2,3,4]])
dgl.DGLGraph
ts.max().item()
##############################
# 构建数据集
##############################
#  from dgl.data import qm9


class QM9Dataset(DGLDataset):
    def __init__(self, url=None, raw_dir=None, save_dir=None, fore_reload=None, verbose=False):
        super(QM9Dataset, self).__init__(name='QM9', url=url, raw_dir=raw_dir,
                                         save_dir=save_dir, fore_reload=fore_reload, verbose=verbose)

        pass

    def __getitem__(self, idx):

        pass

    def __len__(self):
        pass

    def download(self):
        # download raw data to local disk
        pass

    def process(self):
        # process raw data to graphs, labels, splitting masks
        pass

    def load(self):
        # load processed data from directory `self.save_path`
        pass

    def has_cache(self):
        # check whether there are processed data in `self.save_path`
        pass


class QM7bDataset(DGLDataset):
    r"""QM7b dataset for graph property prediction (regression)

    This dataset consists of 7,211 molecules with 14 regression targets.
    Nodes means atoms and edges means bonds. Edge data 'h' means
    the entry of Coulomb matrix.

    Reference: `<http://quantum-machine.org/datasets/>`_

    Statistics:

    - Number of graphs: 7,211
    - Number of regression targets: 14
    - Average number of nodes: 15
    - Average number of edges: 245
    - Edge feature size: 1

    Parameters
    ----------
    raw_dir : str
        Raw file directory to download/contains the input data directory.
        Default: ~/.dgl/
    force_reload : bool
        Whether to reload the dataset. Default: False
    verbose: bool
        Whether to print out progress information. Default: True.

    Attributes
    ----------
    num_labels : int
        Number of labels for each graph, i.e. number of prediction tasks

    Raises
    ------
    UserWarning
        If the raw data is changed in the remote server by the author.

    Examples
    --------
    >>> data = QM7bDataset()
    >>> data.num_labels
    14
    >>>
    >>> # iterate over the dataset
    >>> for g, label in data:
    ...     edge_feat = g.edata['h']  # get edge feature
    ...     # your code here...
    ...
    >>>
    """

    # _url = 'http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/qm7b.mat'
    _url = 'file:///E:/VSCode/Python/torch-study/data/data-source/qm7/qm7b.mat'
    _sha1_str = '4102c744bb9d6fd7b40ac67a300e49cd87e28392'

    def __init__(self, raw_dir=None, force_reload=False, verbose=False):
        super(QM7bDataset, self).__init__(name='qm7b',
                                          url=self._url,
                                          raw_dir=raw_dir,
                                          force_reload=force_reload,
                                          verbose=verbose)

    def process(self):
        # 该函数假定原始数据已经位于 self.raw_dir 目录中
        mat_path = self.raw_path + '.mat'
        self.graphs, self.label = self._load_graph(mat_path)

    def _load_graph(self, filename):
        data = io.loadmat(filename)
        labels = F.tensor(data['T'], dtype=F.data_type_dict['float32'])  # 属性特征
        feats = data['X']  # 空间位置
        num_graphs = labels.shape[0]
        graphs = []
        for i in range(num_graphs):
            edge_list = feats[i].nonzero()
            g = dgl_graph(edge_list)
            # 边特征
            g.edata['h'] = F.tensor(feats[i][edge_list[0], edge_list[1]].reshape(-1, 1),
                                    dtype=F.data_type_dict['float32'])
            graphs.append(g)
        return graphs, labels

    def save(self):
        """save the graph list and the labels"""
        graph_path = os.path.join(self.save_path, 'dgl_graph.bin')
        save_graphs(str(graph_path), self.graphs, {'labels': self.label})

    def has_cache(self):
        graph_path = os.path.join(self.save_path, 'dgl_graph.bin')
        return os.path.exists(graph_path)

    def load(self):
        graphs, label_dict = load_graphs(
            os.path.join(self.save_path, 'dgl_graph.bin'))
        self.graphs = graphs
        self.label = label_dict['labels']

    def download(self):
        file_path = os.path.join(self.raw_dir, self.name + '.mat')
        download(self.url, path=file_path)
        if not check_sha1(file_path, self._sha1_str):
            raise UserWarning('File {} is downloaded but the content hash does not match.'
                              'The repo may be outdated or download may be incomplete. '
                              'Otherwise you can create an issue for it.'.format(self.name))

    @property
    def num_labels(self):
        """Number of labels for each graph, i.e. number of prediction tasks."""
        return 14

    def __getitem__(self, idx):
        r""" Get graph and label by index

        Parameters
        ----------
        idx : int
            Item index

        Returns
        -------
        (:class:`dgl.DGLGraph`, Tensor)
        """
        return self.graphs[idx], self.label[idx]

    def __len__(self):
        r"""Number of graphs in the dataset.

        Return
        -------
        int
        """
        return len(self.graphs)


##############################
# 构建神经网络
##############################
# - Number of graphs: 7,211
# - Number of regression targets: 14
# - Average number of nodes: 15
# - Average number of edges: 245
# - Edge feature size: 1
qm7 = QM7bDataset()
g = qm7[0]
g.ndata['feat']
?QM7bDataset
len(qm7)
qm7[1]
qm7.num_labels
gg = g[0].to_networkx()
nx.draw(gg)
g[0].edges()
g[1]
g[0]

for g, _ in qm7:
    print(g)

features = []

torch.stack, g[1]))

features=[]
gs=[]
for i in range(len(qm7)): vb
    features.append(qm7[i][1].tolist())
    gs.append(qm7[i][0])

num_nodes=[]
num_edges=[]
for i in range(len(gs)):
    num_nodes.append(gs[i].num_nodes())
    num_edges.append(gs[i].num_edges())

max(num_nodes)  # 32
min(num_nodes)  # 4
max(num_edges)  # 529
min(num_edges)  # 16


features=np.asarray(features)
features.std(axis=0)
features.max(axis=0)
features.min(axis=0)
plt.hist(features[:, 0])
plt.hist(features[:, 1])
plt.hist(features[:, 2])
plt.hist(features[:, 3])

import scipy
ddd=scipy.io.loadmat(
    r'E:\VSCode\Python\torch-study\data\data-source\qm7\qm7b.mat')
type(ddd)
ddd.keys()
ddd['X'].shape  # (7211, 23, 23)
ddd['T'].shape  # (7211, 14)

datasets.MNIST()

ddd['T'].max(axis=1)
features.max(axis=1)

################################################################################
v=view(ats, viewer='x3d')

type(v)

IPython.display.HTML
dir(v)


with open('view.html', 'w') as f:
    f.write(v.data)


# eps --> png
ats=all_atoms[-1]
ats.write('ats-1.eps')
os.system('magick -density {} {} {}'.format(300, 'ats-1.eps', 'ats-1.png'))

print(v.data)
ats.write('asd.png')


eps=Image.open('ats-1.eps')
type(eps)

eps.save('ats-111.png')


# RDKit分子可视化
mol1=AllChem.MolFromInchi(InChI[-1][1])
Draw.MolToFile(mol1, 'asd.png', size=(500, 500))

AllChem.MolFromSmiles(Smiles[0][0])


##############################
# 使用sklean中的XGBoost做回归预测
##############################


atom_nums

asd=plt.hist(atom_nums, bins=15)
properties[0]

prop=np.asarray(properties)

plt.hist(prop[:, -1], bins=15)


datasets.MNIST()





nn.init.calculate_gain('tanh')
fc1 = nn.Linear(100,100)
# nn.init.normal_(fc1.weight)
nn.init.xavier_uniform_(fc1.weight,gain=1)
plt.hist(fc1.weight.data.numpy()[0])


embed = nn.Embedding(30,64)
plt.hist(embed.weight.data.numpy()[0])

nn.init.xavier_normal









