import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt

anno = pd.read_csv('/home/kaies/csb/dca/data/zheng/68k_pbmc_barcodes_annotation.tsv', sep='\t', header=0)
anno['celltype'] = anno['celltype'].replace(r'^CD8\+.*$', 0, regex=True)
anno['celltype'] = anno['celltype'].replace(r'^CD4\+.*$', 1, regex=True)
anno['celltype'] = anno['celltype'].replace(r'^CD14\+ Monocyte.*$', 2, regex=True)
anno['celltype'] = anno['celltype'].replace(r'^CD19\+ B.*$', 3, regex=True)
anno['celltype'] = anno['celltype'].replace(r'^CD34\+.*$', 4, regex=True)
anno['celltype'] = anno['celltype'].replace(r'^CD56\+ NK.*$', 5, regex=True)
anno['celltype'] = anno['celltype'].replace(r'^Dendritic.*$', 6, regex=True)

groups = [0, 1, 2, 3, 4, 5, 6]
cdict = {0: ['red', 'CD8+'], 1: ['yellow', 'CD4+'], 2: ['blue', 'CD14+ Monocyte'], 
            3: ['green', 'CD19+ B'], 4: ['purple', 'CD34+'], 5: ['brown', 'CD56+ NK'], 6: ['orange', 'Dendritic']}

fig, ax = plt.subplots(figsize=(10,10))

for g in groups:
    ganno = anno[anno['celltype']==g]
    plt.scatter(ganno['TSNE.1'], ganno['TSNE.2'], c=cdict[g][0], label=cdict[g][1], s=1)
ax.legend(markerscale=10)
fig.tight_layout()
plt.show()
plt.close()

import torch
from federated_dca.models import ZINBAutoEncoder
from federated_dca.datasets import GeneCountData
device = 'cuda' if torch.cuda.is_available() else 'cpu'

dca = ZINBAutoEncoder(1000, 64, 2).to(device)
dca.load_state_dict(torch.load('data/checkpoints/zheng.pt'))

dataset = GeneCountData('/home/kaies/csb/dca/data/zheng/zheng_original.csv', transpose=True, first_col_names=False)
dataset.set_mode('test')
eval_dataloader = torch.utils.data.DataLoader(dataset, batch_size=dataset.__len__())
for data, target, size_factor in eval_dataloader:
        x = dca.encoder(data)
        x = dca.bottleneck(x)

bndata = x.detach().numpy()
anno['bnd.1'] = bndata[:,0]
anno['bnd.2'] = bndata[:,1]

fig, ax = plt.subplots(figsize=(10,10))

for g in groups:
    ganno = anno[anno['celltype']==g]
    plt.scatter(ganno['bnd.1'], ganno['bnd.2'], c=cdict[g][0], label=cdict[g][1], s=1)
ax.legend(markerscale=10)
fig.tight_layout()
plt.show()
plt.close()
