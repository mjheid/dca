import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
from federated_dca.datasets import GeneCountData, threadedGeneCountData, classiGeneCountData
from federated_dca.loss import ZINBLoss, NBLoss
from federated_dca.models import ZINBAutoEncoder, NBAutoEncoder, Classifier
from federated_dca.utils import save_and_load_init_model, train_client, global_agg, sort_paths
import random
import os


def train(path='', EPOCH=500, lr=0.001,
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

    path = os.path.abspath(os.getcwd()) + path

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
    loss_zinb = ZINBLoss(ridge_lambda=ridge, device=device)
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
                    torch.save({'model': dca.state_dict(),
                                'epoch': epoch}, 'data/checkpoints/'+name+'.pt')
                else:
                    es_count += 1
            if es_count >= early_stopping:
                earlystopping = False
        else:
            pass
    
    print(f'Best val loss: {best_val_loss}')

    dca = ZINBAutoEncoder(input_size=input_size, encoder_size=encoder_size, bottleneck_size=bottleneck_size).to(device)
    #dca = NBAutoEncoder(input_size=input_size, encoder_size=64, bottleneck_size=32).to(device)
    dca.load_state_dict(torch.load('data/checkpoints/'+name+'.pt')['model'])
    epoch = torch.load('data/checkpoints/'+name+'.pt')['epoch']
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
        loss = loss_zinb(target, mean, disp, drop)
        #mean, disp = dca(data)
    adata = dataset.adata.copy()
    adata.X = mean.detach().cpu().numpy()
    
    return adata, loss.item(), dca, epoch
    

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
    adata.X = mean.detach().cpu().numpy()
    
    return adata


def train_with_clients(inputfiles='/data/input/', num_clients=2, transpose=False, loginput=False, norminput=False,
            test_split=0.1, filter_min_counts=False, size_factor=False, batch_size=32,
            encoder_size=64, bottleneck_size=32, ridge=0.0, name='dca',
            lr=0.001, reduce_lr=10, early_stopping=15, EPOCH=500, local_epoch=1,
            modeltype='zinb', path_global='/data/global/', param_factor=1, seed=42):

    if torch.cuda.is_available():
        mp.set_start_method('spawn')
        device = 'cuda'  
    else:
        device =  'cpu'
    directory = os.path.abspath(os.getcwd())
    inputfiles = sort_paths(directory+inputfiles)
    global_input = sort_paths(directory+path_global, client=False)[0]

    # Seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = f'{seed}'

    datasets = [threadedGeneCountData(path, device, transpose=transpose,
                        loginput=loginput, norminput=norminput, test_split=test_split,
                        filter_min_counts=filter_min_counts, size_factor=size_factor) for path in inputfiles]
    global_dataset = threadedGeneCountData(global_input, device, transpose=transpose,
                        loginput=loginput, norminput=norminput, test_split=test_split,
                        filter_min_counts=filter_min_counts, size_factor=size_factor)
    input_size = datasets[0].gene_num

    [dataset.set_mode(dataset.train) for dataset in datasets]
    trainDataLoaders = [DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True) for dataset in datasets]
    client_lens = [trainDataLoader.__len__() for trainDataLoader in trainDataLoaders]
    [dataset.set_mode(dataset.val) for dataset in datasets]
    valDataLoaders = [DataLoader(dataset, batch_size=batch_size) for dataset in datasets]
    globalDataLoader = DataLoader(global_dataset, batch_size=batch_size)
    if modeltype == 'zinb':
        global_model = ZINBAutoEncoder(input_size=input_size, encoder_size=encoder_size, bottleneck_size=bottleneck_size).to(device)
        client_models = [ZINBAutoEncoder(input_size=input_size, encoder_size=encoder_size, bottleneck_size=bottleneck_size).to(device)
                        for _ in list(range(num_clients))]
        loss = ZINBLoss(ridge_lambda=ridge, device=device)
    else:
        global_model = NBAutoEncoder(input_size=input_size, encoder_size=encoder_size, bottleneck_size=bottleneck_size).to(device)
        client_models = [NBAutoEncoder(input_size=input_size, encoder_size=encoder_size, bottleneck_size=bottleneck_size).to(device)
                        for _ in list(range(num_clients))]
        loss = NBLoss(device=device)
    
    [model.load_state_dict(global_model.state_dict()) for model in client_models]
    #[model.load_state_dict(torch.load('/home/kaies/csb/data/dca/initial_gc.pt')) for model in client_models]
    #global_model.load_state_dict(torch.load('/home/kaies/csb/data/dca/initial_gc.pt'))
    
    optimizers = [torch.optim.RMSprop(model.parameters(), lr=lr) for model in client_models]
    schedulers = [torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=reduce_lr) for optimizer in optimizers]

    earlystopping = [mp.Event() for _ in list(range(num_clients+1))]
    [e.set() for e in earlystopping]
    es_count = [0 for _ in list(range(num_clients+1))]
    global_model.eval()

    [model.share_memory() for model in client_models]
    processes = []
    events = []
    aggregate_flag = mp.Event()
    for rank in range(num_clients+1):
        if rank < num_clients:
            e = mp.Event()
            events.append(e)
            p = mp.Process(target=train_client, args=(client_models[rank],
                            trainDataLoaders[rank], loss, optimizers[rank],
                            valDataLoaders[rank], e, aggregate_flag, es_count[rank],
                            earlystopping[rank], earlystopping[rank-1], earlystopping[-1], early_stopping, EPOCH, datasets[rank], name, rank, 
                            schedulers[rank], modeltype, local_epoch))
            p.start()
            processes.append(p)
        else:
            p = mp.Process(target=global_agg, args=(client_models,
                            global_model, loss, globalDataLoader,
                            name, client_lens, param_factor, aggregate_flag, events,
                            es_count[rank], earlystopping[-1], earlystopping[rank-1], early_stopping, EPOCH, modeltype))
            p.start()
            processes.append(p)
    for p in processes:
        p.join()
    
    global_model.load_state_dict(torch.load('data/checkpoints/'+name+f'_global.pt')['model'])
    epoch = torch.load('data/checkpoints/'+name+f'_global.pt')['epoch']
    global_dataset.set_mode(global_dataset.test)
    eval_dataloader = DataLoader(global_dataset, batch_size=global_dataset.__len__())
    for data, target, size_factor in eval_dataloader:
        if modeltype == 'zinb':
            mean, disp, drop = global_model(data, size_factor)
            l = loss(target, mean, disp, drop)
        else:
            mean, disp = global_model(data)
            l = loss(data, mean, disp)

    adata = global_dataset.adata_true.copy()
    adata.obs['Group'] = global_dataset.sf['celltype'].values
    adata.X = mean.detach().cpu().numpy()

    return adata, l.item(), global_model, epoch


def train_classifier(path='/data/global/', EPOCH=500, lr=0.1,
        transpose=True, reduce_lr=10, early_stopping=15,
        name='test', loginput=True, norminput=True,
        batchsize=32, seed=42, input_size=199, 
        conv_out=64, kernel=1, num_classes=6, first_col_names=False):
    
    # Seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = f'{seed}'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    directory = os.path.abspath(os.getcwd())
    global_input = sort_paths(directory+path, client=False)[0]
    # global_input = ['/home/kaies/csb/data/dca/data/sixgroupsimulation/sixgroupsimulation_withoutDropout.csv',
    #                     '/home/kaies/csb/data/dca/data/sixgroupsimulation/sixgroupsimulation_withoutDropout.csv',
    #                     '/home/kaies/csb/data/dca/data/global/anno_1.csv']

    dataset = classiGeneCountData(global_input, device, transpose=transpose, first_col_names=first_col_names,
                                    loginput=loginput, norminput=norminput)
    dataset.set_mode(dataset.train)
    trainDataLoader = torch.utils.data.DataLoader(dataset, batch_size=batchsize, shuffle=True)
    dataset.set_mode(dataset.val)
    valDataLoader = torch.utils.data.DataLoader(dataset, batch_size=batchsize)
    dataset.set_mode(dataset.test)
    testDataLoader = torch.utils.data.DataLoader(dataset, batch_size=dataset.__len__())

    cls = Classifier(input_size, conv_out, kernel, num_classes).to(device)

    loss = torch.nn.CrossEntropyLoss()

    opt = torch.optim.SGD(cls.parameters(), lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', patience=reduce_lr)
    run = True
    run_i = 0
    best_val = float('inf')
    for epoch in list(range(EPOCH)):
        if run:
            print(epoch)
            dataset.set_mode(dataset.train)
            train = 0
            for data, target, size_factor in trainDataLoader:
                prediction = cls(data)

                l = loss(prediction, torch.argmax(size_factor, dim=1))

                opt.zero_grad()
                l.backward()
                opt.step()

                train += l.item()
            avg_loss = train / len(trainDataLoader)
            print(f'Avg loss: {avg_loss}')

            val = 0
            dataset.set_mode(dataset.val)
            with torch.no_grad():
                for data, target, size_factor in valDataLoader:
                    prediction = cls(data)

                    l = loss(prediction, torch.argmax(size_factor, dim=1))

                    val += l.item()
                avg_val_loss = val / len(valDataLoader)
                scheduler.step(avg_val_loss)
                print(f'Avg val loss: {avg_val_loss}')
                if avg_val_loss < best_val:
                    best_val = avg_val_loss
                    run_i = 0
                    torch.save(cls.parameters(), 'data/checkpoints/'+name+f'_cls.pt')
                else:
                    run_i += 1
                run == run_i < early_stopping
        
    with torch.no_grad():
        dataset.set_mode(dataset.test)
        cls.load_state_dict(torch.load('data/checkpoints/'+name+f'_cls.pt'))
        for data, target, size_factor in testDataLoader:
            prediction = cls(data)
        result = torch.eq(torch.argmax(prediction, dim=1), torch.argmax(size_factor, dim=1))
        print(torch.sum(result))
        print(result.shape[0])
        print(torch.sum(result)/result.shape[0])
        print(torch.argmax(prediction, dim=1))
