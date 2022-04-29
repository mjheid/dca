import os, sys, argparse
from federated_dca.train import train

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
    model = args.model
    path_global = args.path_global
    param_factor = args.param_factor
    seed = args.seed

    from federated_dca.train import train_with_clients

    train_with_clients(inputfiles=inputfiles,
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
                modeltype=model,
                path_global=path_global,
                param_factor=param_factor,
                seed=seed)
