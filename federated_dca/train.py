import numpy as np
import torch
from torch.utils.data import DataLoader
from federated_dca.datasets import GeneCountData
from federated_dca.loss import ZINBLoss, NBLoss
from federated_dca.models import ZINBAutoEncoder, NBAutoEncoder
from federated_dca.utils import save_and_load_init_model
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

    dataset.set_mode('train')
    trainDataLoader = DataLoader(dataset, batch_size=batchsize, shuffle=True)
    dataset.set_mode('val')
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
            dataset.set_mode('test')
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
                dataset.set_mode('val')
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

    dataset.set_mode('test')
    eval_dataloader = DataLoader(dataset, batch_size=dataset.__len__())
    for data, target, size_factor in eval_dataloader:
        mean, disp, drop = dca(data, size_factor)
        #mean, disp = dca(data)
    adata = dataset.adata
    adata.X = mean.detach().numpy()
    
    return adata
    
