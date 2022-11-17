import pandas as pd
import scanpy as sc
import numpy as np

from federated_dca.datasets import write_text_matrix, GeneCountData

dataset_true = GeneCountData('/home/kaies/csb/data/dca/data/twogroupsimulation/twogroupsimulation_witDropout.csv', filter_min_counts=False, transpose=True, loginput=False, norminput=False, size_factor=False)
dataset = GeneCountData('/home/kaies/csb/data/dca/data/twogroupsimulation/twogroupsimulation_witDropout.csv', filter_min_counts=False, transpose=True, loginput=True, norminput=True, size_factor=True)
pd.DataFrame(dataset.adata.obs['size_factors'].values).to_csv('anno_two.csv')

group1idx = np.where(dataset.adata.obs.index.values=='Group1')[0]
group2idx = np.where(dataset.adata.obs.index.values=='Group2')[0]
group3idx = np.where(dataset.adata.obs.index.values=='Group3')[0]
group4idx = np.where(dataset.adata.obs.index.values=='Group4')[0]
group5idx = np.where(dataset.adata.obs.index.values=='Group5')[0]
group6idx = np.where(dataset.adata.obs.index.values=='Group6')[0]

pd.DataFrame(dataset.adata.obs['size_factors'][group1idx].values).to_csv('anno_1.csv')
pd.DataFrame(dataset.adata.obs['size_factors'][group2idx].values).to_csv('anno_2.csv')
pd.DataFrame(dataset.adata.obs['size_factors'][group3idx].values).to_csv('anno_3.csv')
pd.DataFrame(dataset.adata.obs['size_factors'][group4idx].values).to_csv('anno_4.csv')
pd.DataFrame(dataset.adata.obs['size_factors'][group5idx].values).to_csv('anno_5.csv')
pd.DataFrame(dataset.adata.obs['size_factors'][group6idx].values).to_csv('anno_6.csv')

colnames = dataset.adata.var_names.values
rownames = dataset.adata.obs_names.values
write_text_matrix(dataset.adata.X,
                    'norm_two.csv',
                    rownames=rownames, colnames=colnames, transpose=False)
write_text_matrix(dataset_true.adata.X,
                    'data_two.csv',
                    rownames=rownames, colnames=colnames, transpose=False)

group = dataset.adata.X[group2idx]
rownames = dataset.adata.obs_names.values[group2idx]
write_text_matrix(group,
                    'norm_2.csv',
                    rownames=rownames, colnames=colnames, transpose=False)

group = dataset.adata.X[group3idx]
rownames = dataset.adata.obs_names.values[group3idx]
write_text_matrix(group,
                    'norm_3.csv',
                    rownames=rownames, colnames=colnames, transpose=False)

group = dataset.adata.X[group4idx]
rownames = dataset.adata.obs_names.values[group4idx]
write_text_matrix(group,
                    'norm_4.csv',
                    rownames=rownames, colnames=colnames, transpose=False)

group = dataset.adata.X[group5idx]
rownames = dataset.adata.obs_names.values[group5idx]
write_text_matrix(group,
                    'norm_5.csv',
                    rownames=rownames, colnames=colnames, transpose=False)


group = dataset.adata.X[group6idx]
rownames = dataset.adata.obs_names.values[group6idx]
write_text_matrix(group,
                    'norm_6.csv',
                    rownames=rownames, colnames=colnames, transpose=False)

group = dataset.adata.X[group1idx]
rownames = dataset.adata.obs_names.values[group1idx]
write_text_matrix(group,
                    'norm_1.csv',
                    rownames=rownames, colnames=colnames, transpose=False)

dataset = dataset_true

group = dataset.adata.X[group2idx]
rownames = dataset.adata.obs_names.values[group2idx]
write_text_matrix(group,
                    'data_2.csv',
                    rownames=rownames, colnames=colnames, transpose=False)

group = dataset.adata.X[group3idx]
rownames = dataset.adata.obs_names.values[group3idx]
write_text_matrix(group,
                    'data_3.csv',
                    rownames=rownames, colnames=colnames, transpose=False)

group = dataset.adata.X[group4idx]
rownames = dataset.adata.obs_names.values[group4idx]
write_text_matrix(group,
                    'data_4.csv',
                    rownames=rownames, colnames=colnames, transpose=False)

group = dataset.adata.X[group5idx]
rownames = dataset.adata.obs_names.values[group5idx]
write_text_matrix(group,
                    'data_5.csv',
                    rownames=rownames, colnames=colnames, transpose=False)


group = dataset.adata.X[group6idx]
rownames = dataset.adata.obs_names.values[group6idx]
write_text_matrix(group,
                    'data_6.csv',
                    rownames=rownames, colnames=colnames, transpose=False)

group = dataset.adata.X[group1idx]
rownames = dataset.adata.obs_names.values[group1idx]
write_text_matrix(group,
                    'data_1.csv',
                    rownames=rownames, colnames=colnames, transpose=False)
# print('kek')
