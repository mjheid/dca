import os, sys, argparse
from federated_dca.train import train

def parse_args():
    parser = argparse.ArgumentParser(description='Autoencoder')

    parser.add_argument('input', type=str, help='Input is raw count data in TSV/CSV '
                        'or H5AD (anndata) format. '
                        'Row/col names are mandatory. Note that TSV/CSV files must be in '
                        'gene x cell layout where rows are genes and cols are cells (scRNA-seq '
                        'convention).'
                        'Use the -t/--transpose option if your count matrix in cell x gene layout. '
                        'H5AD files must be in cell x gene format (stats and scanpy convention).')
    parser.add_argument('outputdir', type=str, help='The path of the output directory')
    parser.add_argument('-t', '--transpose', dest='transpose',
            action='store_true', help='Transpose input matrix (default: False)')
    parser.add_argument('--testsplit', dest='testsplit',
            action='store_true', help="Use one fold as a test set (default: False)")

    # training options
    parser.add_argument('--type', type=str, default='nb-conddisp',
            help="Type of autoencoder. Possible values: normal, poisson, nb, "
                 "nb-shared, nb-conddisp (default), nb-fork, zinb, "
                 "zinb-shared, zinb-conddisp( zinb-fork")
    parser.add_argument('--threads', type=int, default=None,
            help='Number of threads for training (default is all cores)')
    parser.add_argument('-b', '--batchsize', type=int, default=32,
            help="Batch size (default:32)")
    parser.add_argument('--sizefactors', dest='sizefactors',
            action='store_true', help="Normalize means by library size (default: True)")
    parser.add_argument('--nosizefactors', dest='sizefactors',
            action='store_false', help="Do not normalize means by library size")
    parser.add_argument('--norminput', dest='norminput',
            action='store_true', help="Zero-mean normalize input (default: True)")
    parser.add_argument('--nonorminput', dest='norminput',
            action='store_false', help="Do not zero-mean normalize inputs")
    parser.add_argument('--loginput', dest='loginput',
            action='store_true', help="Log-transform input (default: True)")
    parser.add_argument('--nologinput', dest='loginput',
            action='store_false', help="Do not log-transform inputs")
    parser.add_argument('-d', '--dropoutrate', type=str, default='0.0',
            help="Dropout rate (default: 0)")
    parser.add_argument('--batchnorm', dest='batchnorm', action='store_true',
            help="Batchnorm (default: True)")
    parser.add_argument('--nobatchnorm', dest='batchnorm', action='store_false',
            help="Do not use batchnorm")
    parser.add_argument('--l2', type=float, default=0.0,
            help="L2 regularization coefficient (default: 0.0)")
    parser.add_argument('--l1', type=float, default=0.0,
            help="L1 regularization coefficient (default: 0.0)")
    parser.add_argument('--l2enc', type=float, default=0.0,
            help="Encoder-specific L2 regularization coefficient (default: 0.0)")
    parser.add_argument('--l1enc', type=float, default=0.0,
            help="Encoder-specific L1 regularization coefficient (default: 0.0)")
    parser.add_argument('--ridge', type=float, default=0.0,
            help="L2 regularization coefficient for dropout probabilities (default: 0.0)")
    parser.add_argument('--gradclip', type=float, default=5.0,
            help="Clip grad values (default: 5.0)")
    parser.add_argument('--activation', type=str, default='relu',
            help="Activation function of hidden units (default: relu)")
    parser.add_argument('--optimizer', type=str, default='RMSprop',
            help="Optimization method (default: RMSprop)")
    parser.add_argument('--init', type=str, default='glorot_uniform',
            help="Initialization method for weights (default: glorot_uniform)")
    parser.add_argument('-e', '--epochs', type=int, default=300,
            help="Max number of epochs to continue training in case of no "
                 "improvement on validation loss (default: 300)")
    parser.add_argument('--earlystop', type=int, default=15,
            help="Number of epochs to stop training if no improvement in loss "
                 "occurs (default: 15)")
    parser.add_argument('--reducelr', type=int, default=10,
            help="Number of epochs to reduce learning rate if no improvement "
            "in loss occurs (default: 10)")
    parser.add_argument('-s', '--hiddensize', type=str, default='64,32,64',
            help="Size of hidden layers (default: 64,32,64)")
    parser.add_argument('--inputdropout', type=float, default=0.0,
            help="Input layer dropout probability"),
    parser.add_argument('-r', '--learningrate', type=float, default=None,
            help="Learning rate (default: 0.001)")
    parser.add_argument('--saveweights', dest='saveweights',
            action='store_true', help="Save weights (default: False)")
    parser.add_argument('--no-saveweights', dest='saveweights',
            action='store_false', help="Do not save weights")
    parser.add_argument('--hyper', dest='hyper',
            action='store_true', help="Optimizer hyperparameters (default: False)")
    parser.add_argument('--hypern', dest='hypern', type=int, default=1000,
            help="Number of samples drawn from hyperparameter distributions during optimization. "
                 "(default: 1000)")
    parser.add_argument('--hyperepoch', dest='hyperepoch', type=int, default=100,
            help="Number of epochs used in each hyperpar optimization iteration. "
                 "(default: 100)")
    parser.add_argument('--debug', dest='debug',
            action='store_true', help="Enable debugging. Checks whether every term in "
                                      "loss functions is finite. (default: False)")
    parser.add_argument('--checkcounts', dest='checkcounts', action='store_true',
            help="Check if the expression matrix has raw (unnormalized) counts (default: True)")
    parser.add_argument('--nocheckcounts', dest='checkcounts', action='store_false',
            help="Do not check if the expression matrix has raw (unnormalized) counts")

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
