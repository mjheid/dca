from pyexpat import model
import numpy as np
import torch
from torch.utils.data import DataLoader
from federated_dca.datasets import GeneCountData
from federated_dca.loss import ZINBLoss, NBLoss
from federated_dca.models import ZINBAutoEncoder, NBAutoEncoder
from federated_dca.utils import save_and_load_init_model, aggregate
import random
import os


def train(path='', EPOCH=500, lr=0.001, batch=32,
        transpose=True, reduce_lr=10, early_stopping=15,
        name='dca', name2=None, loginput=True, test_split=True,
        norminput=True, batchsize=32, ridge=0.0, seed=42,
        save_and_load=False, encoder_size=64, bottleneck_size=32):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Seed
    seed = seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = '0'

    dataset = GeneCountData(path, device, transpose=transpose, test_split=test_split,
                            loginput=loginput, norminput=norminput)
    input_size = dataset.gene_num

    dataset.set_mode(dataset.train)
    trainDataLoader = DataLoader(dataset, batch_size=batchsize, shuffle=True)
    dataset.set_mode(dataset.val)
    valDataLoader = DataLoader(dataset, batch_size=batchsize)
    dca = ZINBAutoEncoder(input_size=input_size, encoder_size=encoder_size, bottleneck_size=bottleneck_size).to(device)
    # dca = NBAutoEncoder(input_size=input_size, encoder_size=64, bottleneck_size=32).to(device)
    if save_and_load:
        dca = save_and_load_init_model(dca, name)
    optimizer = torch.optim.RMSprop(dca.parameters(), lr=lr)
    # loss_zinb = NBLoss()
    loss_zinb = ZINBLoss(ridge_lambda=ridge)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=reduce_lr)

    best_val_loss = float('inf')
    earlystopping = True
    es_count = 0
    # dataset.set_mode('test')
    # eval_dataloader = DataLoader(dataset, batch_size=dataset.__len__())
    for epoch in range(EPOCH):
        if  earlystopping:
            train_loss = 0
            dca.train()
            dataset.set_mode(dataset.train)
            for data, target, size_factor in trainDataLoader:

                mean, disp, drop = dca(data, size_factor)
                loss = loss_zinb(target, mean, disp, drop)
                # mean, disp = dca(data)
                # loss = loss_zinb(data, mean, disp)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            avg_loss = train_loss / len(trainDataLoader)
            print(f'Epoch: {epoch}, Avg train loss: {avg_loss}')

            val_loss = 0
            with torch.no_grad():
                dca.eval()
                dataset.set_mode(dataset.val)
                for data, target, size_factor in valDataLoader:
                    mean, disp, drop = dca(data, size_factor)
                    loss = loss_zinb(target, mean, disp, drop)
                    # mean, disp = dca(data)
                    # loss = loss_zinb(data, mean, disp)

                    val_loss += loss.item()
                
                avg_loss = val_loss / len(valDataLoader)
                scheduler.step(avg_loss)
                print(f'Epoch: {epoch}, Avg val loss: {avg_loss}')
                if avg_loss < best_val_loss:
                    best_val_loss = avg_loss
                    es_count = 0
                    torch.save(dca.state_dict(), 'data/checkpoints/'+name+'.pt')
                else:
                    es_count += 1
            if es_count >= early_stopping:
                earlystopping = False
        else:
            pass
    
    print(f'Best val loss: {best_val_loss}')

    dca = ZINBAutoEncoder(input_size=input_size, encoder_size=encoder_size, bottleneck_size=bottleneck_size).to(device)
    #dca = NBAutoEncoder(input_size=input_size, encoder_size=64, bottleneck_size=32).to(device)
    dca.load_state_dict(torch.load('data/checkpoints/'+name+'.pt'))
    dca.eval()

    # if name2:
    #     dataset = GeneCountData(name2, device, transpose=transpose)
    # else:
    #     dataset = GeneCountData('/home/kaies/csb/dca/data/twogroupsimulation/twogroupsimulation_witDropout.csv', device, transpose=transpose)
    # input_size = dataset.gene_num

    dataset.set_mode(dataset.test)
    eval_dataloader = DataLoader(dataset, batch_size=dataset.__len__())
    for data, target, size_factor in eval_dataloader:
        mean, disp, drop = dca(data, size_factor)
        #mean, disp = dca(data)
    adata = dataset.adata.copy()
    adata.X = mean.detach().numpy()
    
    return adata
    

def train_nb(path='', EPOCH=500, lr=0.001, batch=32,
        transpose=True, reduce_lr=10, early_stopping=15,
        name='nb', name2=None, loginput=True, test_split=True,
        norminput=True, batchsize=32, ridge=0.0, seed=42,
        save_and_load=False, encoder_size=64, bottleneck_size=32):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Seed
    seed = seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = '0'

    dataset = GeneCountData(path, device, transpose=transpose, test_split=test_split,
                            loginput=loginput, norminput=norminput)
    input_size = dataset.gene_num

    dataset.set_mode(dataset.train)
    trainDataLoader = DataLoader(dataset, batch_size=batchsize, shuffle=True)
    dataset.set_mode(dataset.val)
    valDataLoader = DataLoader(dataset, batch_size=batchsize)
    dca = NBAutoEncoder(input_size=input_size, encoder_size=encoder_size, bottleneck_size=bottleneck_size).to(device)
    if save_and_load:
        dca = save_and_load_init_model(dca, name)
    optimizer = torch.optim.RMSprop(dca.parameters(), lr=lr)
    loss_zinb = NBLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=reduce_lr)

    best_val_loss = float('inf')
    earlystopping = True
    es_count = 0
    for epoch in range(EPOCH):
        if  earlystopping:
            train_loss = 0
            dca.train()
            dataset.set_mode(dataset.train)
            for data, target, size_factor in trainDataLoader:

                mean, disp = dca(data, size_factor)
                loss = loss_zinb(target, mean, disp)
                # mean, disp = dca(data)
                # loss = loss_zinb(data, mean, disp)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            avg_loss = train_loss / len(trainDataLoader)
            print(f'Epoch: {epoch}, Avg train loss: {avg_loss}')

            val_loss = 0
            with torch.no_grad():
                dca.eval()
                dataset.set_mode(dataset.val)
                for data, target, size_factor in valDataLoader:
                    mean, disp= dca(data, size_factor)
                    loss = loss_zinb(target, mean, disp)

                    val_loss += loss.item()
                
                avg_loss = val_loss / len(valDataLoader)
                scheduler.step(avg_loss)
                print(f'Epoch: {epoch}, Avg val loss: {avg_loss}')
                if avg_loss < best_val_loss:
                    best_val_loss = avg_loss
                    es_count = 0
                    torch.save(dca.state_dict(), 'data/checkpoints/'+name+'.pt')
                else:
                    es_count += 1
            if es_count >= early_stopping:
                earlystopping = False
        else:
            pass
    
    print(f'Best val loss: {best_val_loss}')

    dca = NBAutoEncoder(input_size=input_size, encoder_size=encoder_size, bottleneck_size=bottleneck_size).to(device)
    dca.load_state_dict(torch.load('data/checkpoints/'+name+'.pt'))
    dca.eval()

    # if name2:
    #     dataset = GeneCountData(name2, device, transpose=transpose)
    # else:
    #     dataset = GeneCountData('/home/kaies/csb/dca/data/twogroupsimulation/twogroupsimulation_witDropout.csv', device, transpose=transpose)
    # input_size = dataset.gene_num

    dataset.set_mode(dataset.test)
    eval_dataloader = DataLoader(dataset, batch_size=dataset.__len__())
    for data, target, size_factor in eval_dataloader:
        mean, disp = dca(data, size_factor)
    adata = dataset.adata.copy()
    adata.X = mean.detach().numpy()
    
    return adata

def train(args):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
    path = os.listdir(path)


    # Seed
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = f'{seed}'

    datasets = [GeneCountData(path, device, transpose=transpose,
                        loginput=loginput, norminput=norminput, test_split=test_split,
                        filter_min_counts=filter_min_counts, size_factor=size_factor) for path in inputfiles]
    global_dataset = GeneCountData(path_global, device, transpose=transpose,
                        loginput=loginput, norminput=norminput, test_split=test_split,
                        filter_min_counts=filter_min_counts, size_factor=size_factor)
    input_size = datasets[0].gene_num

    [dataset.set_mode(dataset.train) for dataset in datasets]
    trainDataLoaders = [DataLoader(dataset, batch_size=batch_size, shuffle=True) for dataset in datasets]
    client_lens = [len(trainDataLoader.__len__()) for trainDataLoader in trainDataLoaders]
    [dataset.set_mode(dataset.val) for dataset in datasets]
    valDataLoaders = [DataLoader(dataset, batch_size=batch_size) for dataset in datasets]
    globalDataLoader = DataLoader(global_dataset, batch_size=batch_size)
    if model == 'zinb':
        global_model = ZINBAutoEncoder(input_size=input_size, encoder_size=encoder_size, bottleneck_size=bottleneck_size).to(device)
        client_models = [ZINBAutoEncoder(input_size=input_size, encoder_size=encoder_size, bottleneck_size=bottleneck_size).to(device)
                        for _ in list(range(num_clients))]
        loss = ZINBLoss(ridge_lambda=ridge)
    else:
        global_model = NBAutoEncoder(input_size=input_size, encoder_size=encoder_size, bottleneck_size=bottleneck_size).to(device)
        client_models = [NBAutoEncoder(input_size=input_size, encoder_size=encoder_size, bottleneck_size=bottleneck_size).to(device)
                        for _ in list(range(num_clients))]
        loss = NBLoss()
    
    [model.load_state_dict(global_model.state_dict()) for model in client_models]
    
    optimizers = [torch.optim.RMSprop(model.parameters(), lr=lr) for model in client_models]
    schedulers = [torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=reduce_lr) for optimizer in optimizers]

    best_val_loss = [float('inf') for _ in list(range(num_clients+1))]
    earlystopping = [True for _ in list(range(num_clients+1))]
    es_count = [0 for _ in list(range(num_clients+1))]
    global_model.eval()

    for epoch in range(EPOCH):
        if  all(earlystopping):
            train_loss = 0
            [model.train() for model in client_models]
            [dataset.set_mode(dataset.train) for dataset in datasets]
            print(f'Epoch: {epoch}')
            for client in list(range(num_clients)):
                for data, target, size_factor in trainDataLoaders[client]:
                    if name == 'zinb':
                        mean, disp, drop = client_models[client](data, size_factor)
                        l = loss(target, mean, disp, drop)
                    else:
                        mean, disp = client_models[client](data)
                        loss = loss(data, mean, disp)

                    optimizers[client].zero_grad()
                    loss.backward()
                    optimizers[client].step()

                    train_loss += l.item()

                avg_loss = train_loss / len(trainDataLoaders[client])
                print(f'Client: {client}, Avg train loss: {avg_loss}')

            val_loss = 0
            with torch.no_grad():
                [model.eval() for model in client_models]
                [dataset.set_mode(dataset.val) for dataset in datasets]
                for client in list(range(num_clients)):
                    for data, target, size_factor in valDataLoaders[client]:
                        if model == 'zinb':
                            mean, disp, drop = client_models[client](data, size_factor)
                            l = loss(target, mean, disp, drop)
                        else:
                            mean, disp = client_models[client](data)
                            l = loss(data, mean, disp)

                        val_loss += l.item()
                    
                    avg_loss = val_loss / len(valDataLoaders[client])
                    schedulers[client].step(avg_loss)
                    print(f'Client: {client}, Avg val loss: {avg_loss}')
                    if avg_loss < best_val_loss[client]:
                        best_val_loss[client] = avg_loss
                        es_count[client] = 0
                        torch.save(client_models[client].state_dict(), 'data/checkpoints/'+name+f'_{client}.pt')
                    else:
                        es_count[client] += 1
                    earlystopping[client] = not es_count[client] >= early_stopping
        else:
            pass
        with torch.no_grad():
            test_loss = 0
            aggregate(global_model, client_models, client_lens)
            for data, target, size_factor in globalDataLoader:
                if model == 'zinb':
                    mean, disp, drop = global_model(data, size_factor)
                    l = loss(target, mean, disp, drop)
                else:
                    mean, disp = global_model(data)
                    l = loss(data, mean, disp)

                test_loss += l.item()
            avg_loss = test_loss / len(globalDataLoader)
            print(f'Global avg test loss: {avg_loss}')
            if avg_loss < best_val_loss[-1]:
                best_val_loss[-1] = avg_loss
                es_count[-1] = 0
                torch.save(global_model.state_dict(), 'data/checkpoints/'+name+f'_global.pt')
            else:
                es_count[client] += 1
            earlystopping[-1] = not es_count[-1] >= early_stopping
    
    print(f'Best val loss: {best_val_loss}')
    