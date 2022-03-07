from dca.mod_train import train

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
import pandas as pd
import scipy as sp
import pandas as pd
from sklearn.metrics import silhouette_score

sc.settings.set_figure_params(dpi=120)

anno = pd.read_csv('/home/kaies/csb/dca/data/twogroupsimulation/twogroupsimulation_annotation.csv')

sim_raw = sc.read('/home/kaies/csb/dca/data/twogroupsimulation/twogroupsimulation_witDropout.csv', first_column_names=True).transpose()
sim_raw.obs['Group'] = anno['Group'].values
sc.pp.filter_genes(sim_raw, min_counts=1)

dca_zinb = train('/home/kaies/csb/dca/data/twogroupsimulation/twogroupsimulation_witDropout.csv')
dca_zinb.obs['Group'] = anno['Group'].values

sim_true = sc.read('/home/kaies/csb/dca/data/twogroupsimulation/twogroupsimulation_withoutDropout.csv', first_column_names=True).transpose()
sim_true.obs['Group'] = anno['Group'].values
sim_true = sim_true[:, sim_raw.var_names].copy()
sim_true

sim_raw_norm = sim_raw.copy()
sc.pp.normalize_total(sim_raw_norm)
sc.pp.log1p(sim_raw_norm)
sc.pp.pca(sim_raw_norm)

sim_true_norm = sim_true.copy()
sc.pp.normalize_total(sim_true_norm)
sc.pp.log1p(sim_true_norm)
sc.pp.pca(sim_true_norm)

dca_zinb_norm = dca_zinb.copy()
sc.pp.normalize_total(dca_zinb_norm)
sc.pp.log1p(dca_zinb_norm)
sc.pp.pca(dca_zinb_norm)

print(sim_raw)
print(sim_true)
print(dca_zinb)

adatas = [sim_true_norm, sim_raw_norm, dca_zinb_norm]
adata_labels = ['Without\ndropout', 'With\ndropout', 'Denoised\n(ZINB)']

fig, axs = plt.subplots(1, len(adatas), figsize=(14,4))

for i, (lbl, ad, ax) in enumerate(zip(adata_labels, adatas, axs)):
    sc.pl.pca_scatter(ad, color='Group', size=20, title=lbl, ax=ax, show=False, legend_loc='none')
    if i!=0: 
        ax.set_xlabel('')
        ax.set_ylabel('')
        
plt.tight_layout()
plt.show()
plt.close()

sils = np.array([silhouette_score(ad.obsm['X_pca'][:, :2], 
                 ad.obs.Group) for ad in adatas])

f, ax = plt.subplots(figsize=(4,4))
ax.grid(axis='x')

# Choose the width of each bar and their positions
width = 4
x_pos = [10,20,30]
 
# Make the plot
ax.bar(x_pos, sils, width=width)
ax.set_xticklabels([''] + adata_labels)

for x, t in zip(x_pos, sils):
    ax.text(x-1, t+0.005, '%.2f' % t)

plt.show()
plt.close()

anno = pd.read_csv('/home/kaies/csb/dca/data/sixgroupsimulation/sixgroupsimulation_annotation.csv')

sim_raw = sc.read('/home/kaies/csb/dca/data/sixgroupsimulation/sixgroupsimulation_witDropout.csv', first_column_names=True).transpose()
sim_raw.obs['Group'] = anno['Group'].values
sc.pp.filter_genes(sim_raw, min_counts=1)

dca_zinb = train('/home/kaies/csb/dca/data/sixgroupsimulation/sixgroupsimulation_witDropout.csv')
dca_zinb.obs['Group'] = anno['Group'].values

sim_true = sc.read('/home/kaies/csb/dca/data/sixgroupsimulation/sixgroupsimulation_withoutDropout.csv', first_column_names=True).transpose()
sim_true.obs['Group'] = anno['Group'].values
sim_true = sim_true[:, sim_raw.var_names].copy()
sim_true

sim_raw_norm = sim_raw.copy()
sc.pp.normalize_total(sim_raw_norm)
sc.pp.log1p(sim_raw_norm)
sc.pp.pca(sim_raw_norm)
sc.pp.neighbors(sim_raw_norm)
sc.tl.umap(sim_raw_norm)

sim_true_norm = sim_true.copy()
sc.pp.normalize_total(sim_true_norm)
sc.pp.log1p(sim_true_norm)
sc.pp.pca(sim_true_norm)
sc.pp.neighbors(sim_true_norm)
sc.tl.umap(sim_true_norm)

dca_zinb_norm = dca_zinb.copy()
sc.pp.normalize_total(dca_zinb_norm)
sc.pp.log1p(dca_zinb_norm)
sc.pp.pca(dca_zinb_norm)
sc.pp.neighbors(dca_zinb_norm)
sc.tl.umap(dca_zinb_norm)

print(sim_raw)
print(sim_true)
print(dca_zinb)

adatas6 = [sim_true_norm, sim_raw_norm, dca_zinb_norm]
adata_labels6 = ['Without\ndropout', 'With\ndropout', 'Denoised\n(ZINB)']

fig, axs = plt.subplots(1, len(adatas6), figsize=(16,4))
for i, (lbl, ad, ax) in enumerate(zip(adata_labels6, adatas6, axs)):
    sc.pl.umap(ad, color='Group', size=20, title=lbl, ax=ax, show=False, legend_loc='none')
    if i!=0: 
        ax.set_xlabel('')
        ax.set_ylabel('')
        
plt.tight_layout()
plt.show()
plt.close()

sils6 = np.array([silhouette_score(ad.obsm['X_umap'], 
                 ad.obs.Group) for ad in adatas6])

f, ax = plt.subplots(figsize=(4,4))
ax.grid(axis='x')

# Choose the width of each bar and their positions
width = 4
x_pos = [10,20,30]
 
# Make the plot
ax.bar(x_pos, sils6, width=width)
ax.set_xticklabels([''] + adata_labels6)

for x, t in zip(x_pos, sils6):
    ax.text(x-1.3, t+0.005, '%.2f' % t)

plt.show()
plt.close()
