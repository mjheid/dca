from federated_dca.train import train

import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc
import pandas as pd
from sklearn.metrics import silhouette_score

sc.settings.set_figure_params(dpi=120)

cellinfo = pd.read_csv('/home/kaies/csb/dca/data/twogroupsimulation/cellinfo.csv', index_col=0)
counts = pd.read_csv('/home/kaies/csb/dca/data/twogroupsimulation/counts.csv', index_col=0)
dropout = pd.read_csv('/home/kaies/csb/dca/data/twogroupsimulation/dropout.csv', index_col=0)
geneinfo = pd.read_csv('/home/kaies/csb/dca/data/twogroupsimulation/geneinfo.csv', index_col=0)
truecounts = pd.read_csv('/home/kaies/csb/dca/data/twogroupsimulation/truecounts.csv', index_col=0)

cellinfo.drop(cellinfo.columns[[0]], axis=1, inplace=True)
#counts.drop(counts.columns[[0]], axis=1, inplace=True)
#dropout.drop(dropout.columns[[0]], axis=1, inplace=True)
geneinfo.drop(geneinfo.columns[[0]], axis=1, inplace=True)
#truecounts.drop(truecounts.columns[[0]], axis=1, inplace=True)

sim_raw = sc.AnnData(counts.values, obs=cellinfo, var=geneinfo)
sim_raw.obs_names = cellinfo.index
sim_raw.var_names = geneinfo.index
sc.pp.filter_genes(sim_raw, min_counts=1)

# remove zero-genes from dropout data frame too
dropout_gt = dropout.loc[:, sim_raw.var_names].values

sim_true = sc.AnnData(truecounts.values, obs=cellinfo, var=geneinfo)
sim_true.obs_names = cellinfo.index
sim_true.var_names = geneinfo.index
sim_true = sim_true[:, sim_raw.var_names].copy()


sim_raw_norm = sim_raw.copy()
sc.pp.normalize_total(sim_raw_norm)
sc.pp.log1p(sim_raw_norm)
sc.pp.pca(sim_raw_norm)

sim_true_norm = sim_true.copy()
sc.pp.normalize_total(sim_true_norm)
sc.pp.log1p(sim_true_norm)
sc.pp.pca(sim_true_norm)

dca_zinb = train(sim_raw, name='2szinb', transpose=False)

dca_zinb_norm = dca_zinb.copy()
sc.pp.normalize_total(dca_zinb_norm)
sc.pp.log1p(dca_zinb_norm)
sc.pp.pca(dca_zinb_norm)

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


de_genes = np.where(sim_true_norm.var.loc[:, 'DEFacGroup1':'DEFacGroup2'].values.sum(1) != 2.0)[0]

obs_idx = np.random.choice(list(range(sim_raw_norm.n_obs)), 300, replace=False)
idx = np.argsort(sim_true_norm.obs.Group.values[obs_idx])
obs_idx = obs_idx[idx]

ax = sc.pl.clustermap(sim_true_norm[obs_idx, de_genes], 'Group', use_raw=False,
                      standard_scale=1, row_cluster=False, show=False, xticklabels=False, yticklabels=False)
ax.ax_row_dendrogram.set_visible(False)

gene_order = ax.dendrogram_col.reordered_ind # preserve gene order from true counts to make heatmaps easily comparable
de_genes = de_genes[gene_order]

ax = sc.pl.clustermap(sim_raw_norm[obs_idx, de_genes], 'Group', use_raw=False, 
                      standard_scale=1, row_cluster=False, col_cluster=False, show=False, xticklabels=False, yticklabels=False)

ax = sc.pl.clustermap(dca_zinb_norm[obs_idx, de_genes], 'Group', use_raw=False, 
                      standard_scale=1, row_cluster=False, col_cluster=False, show=False, xticklabels=False, yticklabels=False)
ax.ax_row_dendrogram.set_visible(False)
plt.show()
plt.close()

from kern_smooth import densCols

f, axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(14,7), constrained_layout=True)

adatas_raw = [sim_raw, dca_zinb]
adata_raw_labels = ['Counts with dropout', 'DCA (ZINB)']
true_counts = sim_true.X.flatten()+1

for i, (ax, ad, lbl) in enumerate(zip(axes[0], adatas_raw, adata_raw_labels)):
    x = ad.X.flatten()+1
    y = true_counts
    c = densCols(np.log10(x), np.log10(y))
    ax.scatter(x, y, s=0.1, alpha=0.1, c=c, marker='o')
    if i == 0: 
        ax.set_ylabel('True counts')
    else:
        ax.plot([min(x), max(x)], [min(x), max(x)], 'k-', linewidth=0.5)
    if i == 1:
        ax.set_title('All counts')        
    cor = np.corrcoef(np.log(x), np.log(y))[0, 1]
    ax.annotate(f'r={cor.round(3)}', (1, 20000))        
        
    ax.loglog()

true_dropout = true_counts[dropout_gt.flatten()==1]
for i, (ax, ad, lbl) in enumerate(zip(axes[1], adatas_raw, adata_raw_labels)):
    x = ad.X[dropout_gt==1]+1
    y = true_dropout
    c = densCols(np.log10(x), np.log10(y))
    ax.scatter(x, y, s=0.1, alpha=0.1, c=c, marker='o')

    if i==0: 
        ax.set_ylabel('True counts')
    else: 
        cor = np.corrcoef(np.log(x), np.log(y))[0, 1]
        ax.annotate(f'r={cor.round(3)}', (1, 20000))                              
        ax.plot([min(x), max(x)], [min(x), max(x)], 'k-', linewidth=0.5)
    if i == 1:
        ax.set_title('Only dropout events')        

    ax.loglog()
    ax.set_xlabel(lbl)
plt.show()
plt.close()

cellinfo = pd.read_csv('/home/kaies/csb/dca/data/sixgroupsimulation/cellinfo.csv', index_col=0)
counts = pd.read_csv('/home/kaies/csb/dca/data/sixgroupsimulation/counts.csv', index_col=0)
dropout = pd.read_csv('/home/kaies/csb/dca/data/sixgroupsimulation/dropout.csv', index_col=0)
geneinfo = pd.read_csv('/home/kaies/csb/dca/data/sixgroupsimulation/geneinfo.csv', index_col=0)
truecounts = pd.read_csv('/home/kaies/csb/dca/data/sixgroupsimulation/truecounts.csv', index_col=0)

cellinfo.drop(cellinfo.columns[[0]], axis=1, inplace=True)
#counts.drop(counts.columns[[0]], axis=1, inplace=True)
#dropout.drop(dropout.columns[[0]], axis=1, inplace=True)
geneinfo.drop(geneinfo.columns[[0]], axis=1, inplace=True)
#truecounts.drop(truecounts.columns[[0]], axis=1, inplace=True)

sim_raw6 = sc.AnnData(counts.values, obs=cellinfo, var=geneinfo)
sim_raw6.obs_names = cellinfo.index
sim_raw6.var_names = geneinfo.index
sc.pp.filter_genes(sim_raw6, min_counts=1)

sim_true6 = sc.AnnData(truecounts.values, obs=cellinfo, var=geneinfo)
sim_true6.obs_names = cellinfo.index
sim_true6.var_names = geneinfo.index
sim_true6 = sim_true6[:, sim_raw6.var_names].copy()

sim_raw_norm6 = sim_raw6.copy()
sc.pp.normalize_total(sim_raw_norm6)
sc.pp.log1p(sim_raw_norm6)
sc.pp.pca(sim_raw_norm6)
sc.pp.neighbors(sim_raw_norm6)
sc.tl.umap(sim_raw_norm6)

sim_true_norm6 = sim_true6.copy()
sc.pp.normalize_total(sim_true_norm6)
sc.pp.log1p(sim_true_norm6)
sc.pp.pca(sim_true_norm6)
sc.pp.neighbors(sim_true_norm6)
sc.tl.umap(sim_true_norm6)

dca_zinb6 = train(sim_raw6, name='6szinb', transpose=False)

dca_zinb_norm6 = dca_zinb6.copy()
sc.pp.normalize_total(dca_zinb_norm6)
sc.pp.log1p(dca_zinb_norm6)
sc.pp.pca(dca_zinb_norm6)
sc.pp.neighbors(dca_zinb_norm6)
sc.tl.umap(dca_zinb_norm6)

adatas6 = [sim_true_norm6, sim_raw_norm6, dca_zinb_norm6]
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


# stratified sampling
total_samples = 300
total_cats = len(sim_true_norm6.obs.Group.cat.categories)
obs_idx = []
for c in sim_true_norm6.obs.Group.cat.categories:
    obs_idx.extend(list(np.random.choice(list(np.where(sim_true_norm6.obs.Group == c)[0]), 
                                         total_samples//total_cats, 
                                         replace=False)))

ax = sc.pl.clustermap(sim_true_norm6[obs_idx, de_genes], 'Group', use_raw=False,
                      standard_scale=1, row_cluster=False, show=False, xticklabels=False, yticklabels=False)

gene_order = ax.dendrogram_col.reordered_ind # preserve gene order from true counts to make heatmaps comparable
de_genes = de_genes[gene_order]

ax = sc.pl.clustermap(sim_raw_norm6[obs_idx, de_genes], 'Group', use_raw=False, 
                      standard_scale=1, row_cluster=False, col_cluster=False, show=False, xticklabels=False, yticklabels=False)

ax = sc.pl.clustermap(dca_zinb_norm6[obs_idx, de_genes], 'Group', use_raw=False, 
                      standard_scale=1, row_cluster=False, col_cluster=False, show=False, xticklabels=False, yticklabels=False)

plt.show()
plt.close()
