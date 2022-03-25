from federated_dca.train import train, train_nb
from federated_dca.loss import ZINBLoss
import torch

# train('/home/kaies/csb/dca/data/test.csv', EPOCH=1, name='test0', loginput=False,
#         norminput=False, batchsize=1, test_split=False)

# loss = ZINBLoss()
# target = torch.tensor([[ 20.,   0.,   0.,  50., 200.,   0.,  30.,   0.,  90.,  10.]])
# mean = torch.tensor([[4.4575e-01, 3.2911e+03, 1.0000e+06, 2.2900e-03, 1.6775e+02, 1.0000e-05,
#          1.0000e+06, 1.0000e-05, 5.5417e+05, 1.0000e-05]])
# disp = torch.tensor([[3.9533e+01, 1.5560e+01, 1.1123e+01, 6.9109e+00, 3.4450e+01, 9.3383e+00,
#          1.0000e-04, 2.5055e+01, 1.9007e+01, 2.5116e+01]])
# drop = torch.tensor([[8.1262e-16, 5.1825e-06, 9.9999e-01, 8.1314e-14, 1.0000e+00, 1.0000e+00,
#          7.6104e-01, 9.7419e-01, 2.2012e-07, 1.0000e+00]])

# l = loss(target, mean, disp, drop)

import scanpy as sc
import pandas as pd

# anno = pd.read_csv('/home/kaies/csb/dca/data/zheng/zheng_annotation.csv')
# anno['clusters'] = anno['clusters'].replace(r'^CD8\+.*$', 'CD8+', regex=True)
# anno['clusters'] = anno['clusters'].replace(r'^CD4\+.*$', 'CD8+', regex=True)
# anno['clusters'] = anno['clusters'].replace(r'^CD14\+ Monocyte.*$', 'CD8+', regex=True)
# anno['clusters'] = anno['clusters'].replace(r'^CD19\+ B.*$', 'CD8+', regex=True)
# anno['clusters'] = anno['clusters'].replace(r'^CD34\+.*$', 'CD8+', regex=True)
# anno['clusters'] = anno['clusters'].replace(r'^CD56\+ NK.*$', 'CD8+', regex=True)
# anno['clusters'] = anno['clusters'].replace(r'^Dendritic.*$', 'CD8+', regex=True)
# adata =  sc.read('/home/kaies/csb/dca/data/zheng/zheng_original.csv', first_column_names=True).transpose()

dca_zinb = train_nb('/home/kaies/csb/dca/data/paul/paul_original.csv', name='paul_nb', bottleneck_size=2, encoder_size=16, EPOCH=1000)
#dca_zinb = train('/home/kaies/csb/dca/data/stoeckius/stoeckius_original.csv', name='stoeckius')
#dca_zinb.obs['Group'] = anno['clusters'].values
