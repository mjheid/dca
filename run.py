import os, argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Autoencoder')

    parser.add_argument('-input', type=str, default='/data/input/',help='Input is raw count data in TSV/CSV '
                        'or H5AD (anndata) format. '
                        'Row/col names are mandatory. Note that TSV/CSV files must be in '
                        'gene x cell layout where rows are genes and cols are cells (scRNA-seq '
                        'convention).'
                        'Use the -t/--transpose option if your count matrix in cell x gene layout. '
                        'H5AD files must be in cell x gene format (stats and scanpy convention).')
    parser.add_argument('-clients', type=int, default=2, help='')
    parser.add_argument('-pg', '--path_global', type=str, default='/data/global/')
    parser.add_argument('--name', type=str, default='test')
    parser.add_argument('--seed', type=str, default=42)

    parser.add_argument('-t', '--transpose', type=bool, default=False)
    parser.add_argument('--loginput', type=bool, default=False)
    parser.add_argument('--norminput', type=bool, default=False)
    parser.add_argument('--test_split', type=float, default=0.1)
    parser.add_argument('--filter_min_counts', type=bool, default=False)
    parser.add_argument('-sf', '--size_factor', type=bool, default=False)
    parser.add_argument('-b', '--batchsize', type=int, default=32)
    parser.add_argument('--encoder_size', type=int, default=64)
    parser.add_argument('--bottleneck_size', type=int, default=32)
    parser.add_argument('--ridge', type=float, default=0.0)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--reduce_lr', type=int, default=20)
    parser.add_argument('--early_stopping', type=int, default=25)
    parser.add_argument('-e', '--epoch', type=int, default=500)
    parser.add_argument('--model', type=str, default='zinb')
    parser.add_argument('-pf', '--param_factor', type=float, default=0.1)
    parser.add_argument('-g', '--gridsearch', type=bool, default=False)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    inputfiles = args.input
    num_clients = args.clients
    transpose = args.transpose
    loginput = args.loginput
    norminput = args.norminput
    test_split = args.test_split
    filter_min_counts = args.filter_min_counts
    size_factor = args.size_factor
    batch_size = args.batchsize
    encoder_size = args.encoder_size
    bottleneck_size = args.bottleneck_size
    ridge = args.ridge
    name = args.name
    lr = args.lr
    reduce_lr = args.reduce_lr
    early_stopping = args.early_stopping
    EPOCH = args.epoch
    modeltype = args.model
    path_global = args.path_global
    param_factor = args.param_factor
    seed = args.seed

    from federated_dca.train import train_with_clients, train
    import scanpy as sc
    from sklearn.metrics import silhouette_score

    if num_clients > 1:
        adata, best_total_loss, model, epoch = train_with_clients(inputfiles=inputfiles,
                    num_clients=num_clients,
                    transpose=transpose,
                    loginput=loginput,
                    norminput=norminput,
                    test_split=test_split,
                    filter_min_counts=filter_min_counts,
                    size_factor=size_factor,
                    batch_size=batch_size,
                    encoder_size=encoder_size,
                    bottleneck_size=bottleneck_size,
                    ridge=ridge,
                    name=name,
                    lr=lr,
                    reduce_lr=reduce_lr,
                    early_stopping=early_stopping,
                    EPOCH=EPOCH,
                    modeltype=modeltype,
                    path_global=path_global,
                    param_factor=param_factor,
                    seed=seed)
    else:
        adata, best_total_loss, model, epoch = train(path=inputfiles,
                    EPOCH=EPOCH,
                    lr=lr,
                    batchsize=batch_size,
                    transpose=transpose,
                    reduce_lr=reduce_lr,
                    early_stopping=early_stopping,
                    name=name,
                    loginput=loginput,
                    test_split=test_split,
                    norminput=norminput,
                    ridge=ridge,
                    seed=seed,
                    encoder_size=encoder_size,
                    bottleneck_size=bottleneck_size)
    
    if args.gridsearch:
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
        sc.pp.pca(adata)
        sc.pp.neighbors(adata)
        sc.tl.umap(adata)
        adata.obs['Group'] = adata.obs.index.values
        sil_score = silhouette_score(adata.obsm['X_umap'], adata.obs.Group)
        print(f'SIL score: {sil_score}')

        import os.path
        import torch

        directory = os.path.abspath(os.getcwd())
        if os.path.exists(os.path.join(directory, f'{name}.pt')):
            checkpoint = torch.load(f'{name}.pt')
            prev_best_total_loss = checkpoint['best_total_loss']
            prev_best_sil_score = checkpoint['best_sil_score']
        else:
            prev_best_total_loss = float('inf')
            prev_best_sil_score = float('-inf')
        
        if sil_score > prev_best_sil_score:
            torch.save({
                'model': model.state_dict(),
                'best_total_loss': best_total_loss,
                'best_sil_score': sil_score,
                'batch_size': batch_size,
                'lr': lr,
                'reduce_lr': reduce_lr,
                'early_stopping': early_stopping,
                'seed': seed,
                'encoder_size': encoder_size,
                'bottleneck_size': bottleneck_size,
                'epoch': epoch,
                'model_type': modeltype,
                'param_factor': param_factor,
                'num_clients': num_clients
            }, f'{name}.pt')
            adatas6 = adata
            adata_labels6 = 'Denoised\n(ZINB)'
            import matplotlib.pyplot as plt
            fig, axs = plt.subplots(1, 1, figsize=(6,4))
            sc.pl.umap(adatas6, color='Group', size=20, title=adata_labels6, ax=axs, show=False, legend_loc='none')
            plt.tight_layout()
            plt.savefig(f'{name}_cluster.pdf')
            plt.close()
        with open('data/checkpoints/log.txt', 'a') as logfile:
            logfile.write(f'Name: {name}, Epoch: {epoch}, model: {modeltype}, loss: {best_total_loss}, sil: {sil_score}, lr: {lr}, batch: {batch_size}, r_lr: {reduce_lr}, e_st: {early_stopping}, pf: {param_factor}, cl: {num_clients}' + os.linesep)
