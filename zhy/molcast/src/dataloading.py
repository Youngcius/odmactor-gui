import data_process
from torchvision import datasets
import networkx as nx
from dgl.data import RedditDataset, QM7bDataset
import dgl.function as fn
from dgl.nn import SAGEConv
from dgl.utils import expand_as_pair, check_eq_shape
from dgl.data.utils import save_info, load_info
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
from torch.utils.data import DataLoader
# from dgl.dataloading.pytorch import GraphDataLoader
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

?torch.utils.data.DataLoader
from dgl.data import split_dataset

trainset, evalset, testset = split_dataset(qm9,[0.8,0.1,0.1])
trainset[0]
qm9[-1]
testset[-1]
len(evalset)
len(testset)

ss = nn.Sequential(nn.Linear(10,2))
oop = optim.Adam(ss.parameters())
type(oop)
ss.state_dict()
torch.save(ss.state_dict(),'ss.pkl')
ss.load_state_dict(torch.load('ss.pkl'))
ss.state_dict()

nx.draw(g.to_networkx().to_undirected(),node_size=6)

a.detach()

qm9loader = DataLoader(qm9,batch_size=16,shuffle=True,collate_fn=batcher())


for i, (g,l) in enumerate(qm9loader):
    if i > 0:
        break
    print(i,g,l)
type(g)
g.ndata['X'].shape
g.ndata.keys()

dgl.

133885/16

dgl.__version__
torch.__version__
len(harmonics[100])
from dgl.dataloading import GraphDataLoader
from dgl.dataloading.pytorch import GraphDataLoader
torch.linspace(0)
nn.Parameter(b,requires_grad=False)

qm9.dict['all_atoms'][0].positions.min()
torch.cat((a,b),dim=1)

qm9.prop_names
max(qm9.dict['harmonics'][0])

embed = nn.Embedding(10,3)
a = torch.randint(0,10,size=(1000,))
a.unique()
a = embed(a)
b = torch.randn(20).view(-1,1)
b
a.mean()
torch.stack((a,b),dim=1)
torch.cat((a,b),dim=1)
embed(a)

qm9 = data_process.QM9Dataset()
qm9.dict['charges']
qm9.graphs[10000].ndata

# 图网络预测
# 需要注意的是批次化图中的节点和边属性没有批次大小对应的维度。
qm9loader = DataLoader(qm9, batch_size=16, shuffle=True)

type(qm9loader)
type()


qm9[0]

with open(qm9.dict['fnames'][0],'r') as f:
    txt = f.read()

print(txt)


qm9.raw_path
qm9.raw_dir
qm9.
os.path.join('../data/', 'asd')

nx.draw(qm9[0][0].to_networkx(), with_labels=True)

# raw_path = '../data/dsgdb9nsd.xyz'
# ordered_files = sorted(os.listdir(raw_path), key=lambda x: (int(re.sub("\D", "", x)), x))
# all_properties = []
# irange = np.arange(len(ordered_files), dtype=np.int) # 0 -- 133885

g = qm9[0][0]
g.edata['']
for i,j in zip(g.edges()[0], g.edges()[1]):
    print(i,j)
    print(g.ndata['Z'][i], g.ndata['Z'][j])
    print('-----------')
g.ndata['X'][0]
g.edges()

torch.norm(torch.Tensor([1,0,1]))

torch.cat((a,b))

torch.abs(-100)
abs

#########################################################
#########################################################
mol = Chem.MolFromInchi(InChI[100][0])
mol = Chem.AddHs(mol)
mol.GetAtomWithIdx(1).GetIdx()
mol.GetNumAtoms()

A = Chem.rdmolops.GetAdjacencyMatrix(mol)
with open(fnames[100], 'r') as f:
    txt = f.read()

print(txt)
ats.positions

atoms_info = [(atom.GetIdx(), atom.GetAtomicNum(), atom.GetSymbol())
              for atom in mol.GetAtoms()]  # idx, Z, Symbol
atom_index = list(range(mol.GetNumAtoms()))
atoms_info
print(txt)

bonds_info = [(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
              for bond in mol.GetBonds()]
bonds_info * 2
bonds_info = bonds_info + bonds_info  # 变为双向图
atom_index


qm7[0][0].edges()

g = dgl.graph(bonds_info)
g.edges()
g.nodes()
nx.draw(g.to_networkx().to_undirected(), with_labels=True,)
mol
A

atoms_info

ats = all_atoms[100]
ats.positions

dir(ats)
ats._neighbors
ats.get_dipole_moment()
ats.lattice()

all_atoms[0].info

ats.todict()


#########################################################
#########################################################
