import  torch.utils.data.Dataset as Dataset
import torch
import numpy as np
import io



class GeneCountData(torch.utils.data.Dataset):
    """Dataset of GeneCounts for DCA"""

    def __init__(self, path='data/francesconi/francesconi_withDropout.csv', transpose=False, check_count=False,
                test_split=False, loginput=True, norminput=True, cuda=None):
        """
        Args:
            
        """
        adata = io.read_dataset(path,
                            transpose=False, # assume gene x cell by default
                            check_counts=False,
                            test_split=False)

        adata = io.normalize(adata,
                            size_factors=True,
                            logtrans_input=loginput,
                            normalize_input=norminput)
        
        self.adata = adata
        # TODO: to cuda
        self.data = torch.from_numpy(np.array(adata.X))
        self.size_factors = torch.from_numpy(np.array(adata.obs.size_factors))
        self.gene_num = self.data.shape[1]
        

    def __len__(self):

        return self.data.shape[0]

    def __getitem__(self, idx):

        data = self.data[idx]
        size_factors = self.size_factors[idx]

        return data, size_factors

#data_loader = GeneCountData()