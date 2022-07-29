import torch
import numpy as np
import scipy as sp
import pandas as pd
import scanpy as sc
from sklearn.model_selection import train_test_split



class GeneCountData(torch.utils.data.Dataset):
    """Dataset of GeneCounts for DCA"""

    def __init__(self, path='data/francesconi/francesconi_withDropout.csv', device='cpu',
                transpose=True, check_count=False, test_split=True, loginput=True,
                 norminput=True, filter_min_counts=True, first_col_names=True, size_factor=True):
        """
        Args:
            
        """
        adata = read_dataset(path,
                            transpose=transpose, # assume gene x cell by default
                            check_counts=check_count,
                            test_split=True,
                            first_col_names=first_col_names)

        adata = normalize(adata,
                            filter_min_counts=filter_min_counts,
                            size_factors=size_factor,
                            logtrans_input=loginput,
                            normalize_input=norminput)
        
        self.adata = adata

        self.data = torch.from_numpy(np.array(adata.X)).to(device)
        self.size_factors = torch.from_numpy(np.array(adata.obs.size_factors)).to(device)
        self.target = torch.from_numpy(np.array(adata.raw.X)).to(device)
        self.gene_num = self.data.shape[1]

        if test_split:
            adata = adata[adata.obs.dca_split == 'train']

            train_idx, test_idx = train_test_split(np.arange(adata.n_obs), test_size=0.1, random_state=42)
            spl = pd.Series(['train'] * adata.n_obs)
            spl.iloc[test_idx] = 'test'
            adata.obs['dca_split'] = spl.values

            self.val_data = torch.from_numpy(np.array(adata[adata.obs.dca_split == 'test'].X)).to(device)
            self.val_target = torch.from_numpy(np.array(adata[adata.obs.dca_split == 'test'].raw.X)).to(device)
            self.val_size_factors = torch.from_numpy(np.array(adata[adata.obs.dca_split == 'test'].obs.size_factors)).to(device)

            self.train_data = torch.from_numpy(np.array(adata[adata.obs.dca_split == 'train'].X)).to(device)
            self.train_target = torch.from_numpy(np.array(adata[adata.obs.dca_split == 'train'].raw.X)).to(device)
            self.train_size_factors = torch.from_numpy(np.array(adata[adata.obs.dca_split == 'train'].obs.size_factors)).to(device)
        
        self.train = 0
        self.val = 1
        self.test = 2
        self.mode = self.test
    
    def set_mode(self, mode):
        if mode == self.train:
            self.mode = self.train
        elif mode == self.val:
            self.mode = self.val
        elif mode == self.test:
            self.mode = self.test


    def __len__(self):
        if self.mode == self.train:
            return self.train_data.shape[0]
        elif self.mode == self.val:
            return self.val_data.shape[0]
        else:
            return self.data.shape[0]

    def __getitem__(self, idx):
        if self.mode == self.train:
            data = self.train_data[idx]
            target = self.train_target[idx]
            size_factors = self.train_size_factors[idx]
        elif self.mode == self.val:
            data = self.val_data[idx]
            target = self.val_target[idx]
            size_factors = self.val_size_factors[idx]
        else:
            data = self.data[idx]
            target = self.target[idx]
            size_factors = self.size_factors[idx]

        return data, target, size_factors


class threadedGeneCountData(GeneCountData):
    def __init__(self, path='data/francesconi/francesconi_withDropout.csv', device='cpu',
                transpose=True, check_count=False, test_split=True, loginput=False,
                 norminput=False, filter_min_counts=False, first_col_names=True, size_factor=False):

        self.adata_true = read_dataset(path[0],
                            transpose=transpose, # assume gene x cell by default
                            check_counts=check_count,
                            test_split=False,
                            first_col_names=first_col_names)
        adata_raw = read_dataset(path[1],
                            transpose=transpose, # assume gene x cell by default
                            check_counts=check_count,
                            test_split=False,
                            first_col_names=first_col_names)
        self.adata_raw = normalize(adata_raw,
                            filter_min_counts=filter_min_counts,
                            size_factors=size_factor,
                            logtrans_input=loginput,
                            normalize_input=norminput)
        self.sf = pd.read_csv(path[2])

        self.data = torch.from_numpy(np.array(self.adata_raw.X)).to(device)
        self.size_factors = torch.from_numpy(np.array(self.sf['size_factors'].values)).to(device)
        self.target = torch.from_numpy(np.array(self.adata_true.X)).to(device)
        self.gene_num = self.data.shape[1]

        if test_split:
            train_idx, test_idx = train_test_split(np.arange(self.adata_true.n_obs), test_size=0.1, random_state=42)
            # spl = pd.Series(['train'] * self.adata_true.n_obs)
            # spl.iloc[test_idx] = 'test'
            # self.adata_true.obs['dca_split'] = spl.values
            # self.adata_raw.obs['dca_split'] = spl.values
            # self.sf['dca_split'] = spl.values

            self.val_data = torch.from_numpy(np.array(self.adata_raw[self.sf.dca_split == 1].X)).to(device)
            self.val_target = torch.from_numpy(np.array(self.adata_true[self.sf.dca_split == 1].X)).to(device)
            self.val_size_factors = torch.from_numpy(np.array(self.sf[self.sf.dca_split == 1]['size_factors'])).to(device)

            self.train_data = torch.from_numpy(np.array(self.adata_raw[self.sf.dca_split == 0].X)).to(device)
            self.train_target = torch.from_numpy(np.array(self.adata_true[self.sf.dca_split == 0].X)).to(device)
            self.train_size_factors = torch.from_numpy(np.array(self.sf[self.sf.dca_split == 0]['size_factors'])).to(device)
    
        self.train = 0
        self.val = 1
        self.test = 2
        self.mode = self.test


class classiGeneCountData(GeneCountData):
    def __init__(self, path='data/francesconi/francesconi_withDropout.csv', device='cpu',
                transpose=True, check_count=False, test_split=True, loginput=False,
                 norminput=False, filter_min_counts=False, first_col_names=True, size_factor=False):
        
        self.adata_true = read_dataset(path[0],
                            transpose=transpose, # assume gene x cell by default
                            check_counts=check_count,
                            test_split=False,
                            first_col_names=first_col_names)
        adata_raw = read_dataset(path[1],
                            transpose=transpose, # assume gene x cell by default
                            check_counts=check_count,
                            test_split=False,
                            first_col_names=first_col_names)
        self.adata_raw = normalize(adata_raw,
                            filter_min_counts=filter_min_counts,
                            size_factors=size_factor,
                            logtrans_input=loginput,
                            normalize_input=norminput)
        self.sf = pd.read_csv(path[2])

        classes = self.sf['celltype'].unique()
        dic = {}
        for i in list(range(len(classes))):
            t = torch.tensor([0]*len(classes))
            t[i] = 1
            dic[classes[i]] = t
        self.sf['class_tensor'] = [torch.tensor([0]*len(classes), dtype=int)] * self.sf.shape[0]
        self.sf = self.sf.assign(class_tensor=self.sf.celltype.map(dic).fillna(self.sf.class_tensor))

        self.data = torch.from_numpy(np.array(self.adata_raw.X)).to(device)
        #why do i have to do this??? there should be a better way, but from_numpy doesnt work
        size_factor = torch.zeros((self.sf.shape[0], len(classes)))
        for i in list(range(self.sf.shape[0])):
            size_factor[i] = self.sf['class_tensor'].values[i]
        self.size_factors = size_factor.to(device)
        self.target = torch.from_numpy(np.array(self.adata_true.X)).to(device)
        self.gene_num = self.data.shape[1]

        if test_split:
            train_idx, test_idx = train_test_split(np.arange(self.adata_true.n_obs), test_size=0.3, random_state=42)
            spl = pd.Series(['train'] * self.adata_true.n_obs)
            spl.iloc[test_idx] = 'test'
            self.adata_true.obs['dca_split'] = spl.values
            self.adata_raw.obs['dca_split'] = spl.values
            self.sf['dca_split'] = spl.values

            self.val_data = torch.from_numpy(np.array(self.adata_raw[self.adata_raw.obs.dca_split == 'test'].X)).to(device)
            self.val_target = torch.from_numpy(np.array(self.adata_true[self.adata_true.obs.dca_split == 'test'].X)).to(device)
            self.val_size_factors = self.size_factors[self.sf.dca_split == 'test'].to(device)

            self.train_data = torch.from_numpy(np.array(self.adata_raw[self.adata_raw.obs.dca_split == 'train'].X)).to(device)
            self.train_target = torch.from_numpy(np.array(self.adata_true[self.adata_true.obs.dca_split == 'train'].X)).to(device)
            self.train_size_factors = self.size_factors[self.sf.dca_split == 'train'].to(device)
    
        self.train = 0
        self.val = 1
        self.test = 2
        self.mode = self.test


def read_dataset(adata, transpose=False, test_split=False, copy=False, check_counts=True, first_col_names=True):

    if isinstance(adata, sc.AnnData):
        if copy:
            adata = adata.copy()
    elif isinstance(adata, str):
        adata = sc.read(adata, first_column_names=first_col_names)
    else:
        raise NotImplementedError

    if check_counts:
        # check if observations are unnormalized using first 10
        X_subset = adata.X[:10]
        norm_error = 'Make sure that the dataset (adata.X) contains unnormalized count data.'
        if sp.sparse.issparse(X_subset):
            assert (X_subset.astype(int) != X_subset).nnz == 0, norm_error
        else:
            assert np.all(X_subset.astype(int) == X_subset), norm_error

    if transpose: adata = adata.transpose()

    if test_split:
        train_idx, test_idx = train_test_split(np.arange(adata.n_obs), test_size=0.1, random_state=42)
        spl = pd.Series(['train'] * adata.n_obs)
        spl.iloc[test_idx] = 'test'
        adata.obs['dca_split'] = spl.values
    else:
        adata.obs['dca_split'] = 'train'

    adata.obs['dca_split'] = adata.obs['dca_split'].astype('category')
    print('dca: Successfully preprocessed {} genes and {} cells.'.format(adata.n_vars, adata.n_obs))

    return adata


def normalize(adata, filter_min_counts=True, size_factors=True, normalize_input=True, logtrans_input=True):

    if filter_min_counts:
        sc.pp.filter_genes(adata, min_counts=1)
        sc.pp.filter_cells(adata, min_counts=1)

    if size_factors or normalize_input or logtrans_input:
        adata.raw = adata.copy()
    else:
        adata.raw = adata

    if size_factors:
        sc.pp.normalize_per_cell(adata)
        adata.obs['size_factors'] = adata.obs.n_counts / np.median(adata.obs.n_counts)
    else:
        adata.obs['size_factors'] = 1.0

    if logtrans_input:
        sc.pp.log1p(adata)

    if normalize_input:
        sc.pp.scale(adata)

    return adata

def read_genelist(filename):
    genelist = list(set(open(filename, 'rt').read().strip().split('\n')))
    assert len(genelist) > 0, 'No genes detected in genelist file'
    print('dca: Subset of {} genes will be denoised.'.format(len(genelist)))

    return genelist

def write_text_matrix(matrix, filename, rownames=None, colnames=None, transpose=False):
    if transpose:
        matrix = matrix.T
        rownames, colnames = colnames, rownames

    pd.DataFrame(matrix, index=rownames, columns=colnames).to_csv(filename,
                                                                  sep=',',
                                                                  index=(rownames is not None),
                                                                  header=(colnames is not None),
                                                                  float_format='%.6f')
