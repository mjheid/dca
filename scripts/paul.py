import pandas as pd
import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt

anno = pd.read_csv('/home/kaies/csb/dca/data/paul/paul_annotation.csv')
anno['celltypes'] = anno['celltypes'].replace(r'^1Ery.*$', 0, regex=True)
anno['celltypes'] = anno['celltypes'].replace(r'^2Ery.*$', 1, regex=True)
anno['celltypes'] = anno['celltypes'].replace(r'^3Ery*$', 2, regex=True)
anno['celltypes'] = anno['celltypes'].replace(r'^4Ery.*$', 3, regex=True)
anno['celltypes'] = anno['celltypes'].replace(r'^5Ery.*$', 4, regex=True)
anno['celltypes'] = anno['celltypes'].replace(r'^6Ery.*$', 5, regex=True)
anno['celltypes'] = anno['celltypes'].replace(r'^7MEP.*$', 6, regex=True)
anno['celltypes'] = anno['celltypes'].replace(r'^8Mk.*$', 7, regex=True)
anno['celltypes'] = anno['celltypes'].replace(r'^9GMP.*$', 8, regex=True)
anno['celltypes'] = anno['celltypes'].replace(r'^10GMP.*$', 9, regex=True)
anno['celltypes'] = anno['celltypes'].replace(r'^11DC.*$', 10, regex=True)
anno['celltypes'] = anno['celltypes'].replace(r'^12Baso.*$', 11, regex=True)
anno['celltypes'] = anno['celltypes'].replace(r'^13Baso.*$', 12, regex=True)
anno['celltypes'] = anno['celltypes'].replace(r'^14Mo.*$', 13, regex=True)
anno['celltypes'] = anno['celltypes'].replace(r'^15Mo.*$', 14, regex=True)
anno['celltypes'] = anno['celltypes'].replace(r'^16Neu.*$', 15, regex=True)
anno['celltypes'] = anno['celltypes'].replace(r'^17Neu*$', 16, regex=True)
anno['celltypes'] = anno['celltypes'].replace(r'^18Eos.*$', 17, regex=True)
anno['celltypes'] = anno['celltypes'].replace(r'^19Lymph.*$', 18, regex=True)

groups = list(range(19))
cdict = {0: '1Ery', 1: '2Ery', 2: '3Ery', 3: '4Ery', 4: '5Ery',
        5: '6Ery', 6: '7MEP', 7: '8Mk', 8: '9GMP', 9: '10GMP',
        10: '11DC', 11: '12Baso', 12: '13Baso', 13: '14Mo',
        14: '15Mo', 15: '16Neu', 16: '17Neu', 17: '18Eos',
        18: '19Lymph'}

import torch
from federated_dca.models import NBAutoEncoder, ZINBAutoEncoder
from federated_dca.datasets import GeneCountData
import random
import os

# Seed
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
os.environ['PYTHONHASHSEED'] = '0'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

dataset = GeneCountData('/home/kaies/csb/dca/data/paul/paul_original.csv', transpose=True, first_col_names=False)
dataset.set_mode('test')
eval_dataloader = torch.utils.data.DataLoader(dataset, batch_size=dataset.__len__())

dca = ZINBAutoEncoder(dataset.gene_num, 64, 2).to(device)
dca = NBAutoEncoder(dataset.gene_num, 16, 2)
dca.load_state_dict(torch.load('data/checkpoints/paul_nb.pt'))
dca.eval()

for data, target, size_factor in eval_dataloader:
        x = dca.encoder(data)
        x = list(dca.bottleneck.modules())[1](x)

bndata = x.detach().numpy()
anno['bnd.1'] = bndata[:,0]
anno['bnd.2'] = bndata[:,1]

fig, ax = plt.subplots(figsize=(10,10))

for g in groups:
    ganno = anno[anno['celltypes']==g]
    plt.scatter(ganno['bnd.1'], ganno['bnd.2'], label=cdict[g])#, c=ganno['celltypes'], label=cdict[g], s=1)
ax.legend()
fig.tight_layout()
plt.show()
plt.close()

adata_ae = sc.datasets.paul15()
adata_ae.obsm['bnd'] = bndata
genes = adata_ae.var_names.to_native_types()
genes[genes == 'Sfpi1'] = 'Pu.1'
adata_ae.var_names = pd.Index(genes)
sc.pp.log1p(adata_ae)
sc.pp.pca(adata_ae)
sc.pp.neighbors(adata_ae, n_neighbors=20, use_rep='bnd')
sc.tl.dpt(adata_ae, n_branchings=1)
#sc.pl.diffmap(adata_ae, color='dpt_pseudotime', title='Diffusion Pseudotime of GMP-MEP branches', color_map='viridis', use_raw=False)

fig, ax = plt.subplots(figsize=(10,10))

anno['dpt'] = adata_ae.obs['dpt_pseudotime'].values

plt.scatter(anno['bnd.1'], anno['bnd.2'], c=anno['dpt'])
fig.tight_layout()
plt.show()
plt.close()
