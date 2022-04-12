##Federated deep count autoencoder for denoising scRNA-seq data

A deep count autoencoder network to denoise scRNA-seq data and remove the dropout effect by taking the count structure, overdispersed nature and sparsity of the data into account using a deep autoencoder with zero-inflated negative binomial (ZINB) loss function.

See [manuscript](https://www.nature.com/articles/s41467-018-07931-2) and [tutorial](https://nbviewer.ipython.org/github/theislab/dca/blob/master/tutorial.ipynb) for more details.

### Installation

#### pip



### Usage



### Results

Output folder contains the main output file (representing the mean parameter of ZINB distribution) as well as some additional matrices in TSV format:

