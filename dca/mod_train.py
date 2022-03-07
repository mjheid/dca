import torch
from torch.utils.data import DataLoader
from dca.datasets import GeneCountData
from dca.my_loss import ZINBLoss, NBLoss
from dca.models import ZINBAutoEncoder, NBAutoEncoder

def train(path='', EPOCH=300, lr=0.001, batch=32,
        transpose=True, reduce_lr=10, early_stopping=15):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    dataset = GeneCountData(path, device, transpose=transpose)
    input_size = dataset.gene_num

    dataset.set_mode('train')
    trainDataLoader = DataLoader(dataset, batch_size=32, shuffle=True)
    dataset.set_mode('val')
    valDataLoader = DataLoader(dataset, batch_size=32)
    dca = ZINBAutoEncoder(input_size=input_size, encoder_size=64, bottleneck_size=32).to(device)
    # dca = NBAutoEncoder(input_size=input_size, encoder_size=64, bottleneck_size=32).to(device)
    optimizer = torch.optim.RMSprop(dca.parameters(), lr=lr)
    # loss_zinb = NBLoss()
    loss_zinb = ZINBLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=reduce_lr)

    best_val_loss = float('inf')
    earlystopping = True
    es_count = 0

    for epoch in range(EPOCH):
        if  earlystopping:
            train_loss = 0
            dca.train()
            dataset.set_mode('train')
            for data, size_factor in trainDataLoader:

                mean, disp, drop = dca(data)
                loss = loss_zinb(data, mean, disp, drop)
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
                for data, size_factor in valDataLoader:
                    mean, disp, drop = dca(data)
                    loss = loss_zinb(data, mean, disp, drop)
                    # mean, disp = dca(data)
                    # loss = loss_zinb(data, mean, disp)

                    val_loss += loss.item()
                
                avg_loss = val_loss / len(valDataLoader)
                scheduler.step(avg_loss)
                print(f'Epoch: {epoch}, Avg val loss: {avg_loss}')
                if avg_loss < best_val_loss:
                    best_val_loss = avg_loss
                    es_count = 0
                    torch.save(dca.state_dict(), 'dca.pt')
                else:
                    es_count += 1
            if es_count >= early_stopping:
                earlystopping = False
        else:
            pass
    
    print(f'Best val loss: {best_val_loss}')

    dca = ZINBAutoEncoder(input_size=input_size, encoder_size=64, bottleneck_size=32).to(device)
    #dca = NBAutoEncoder(input_size=input_size, encoder_size=64, bottleneck_size=32).to(device)
    dca.load_state_dict(torch.load('dca.pt'))
    dca.eval()

    dataset.set_mode('test')
    eval_dataloader = DataLoader(dataset, batch_size=dataset.__len__())
    for data, size_factor in eval_dataloader:
        mean, disp, drop = dca(data)
        #mean, disp = dca(data)
    adata = dataset.adata
    adata.X = mean.detach().numpy()
    
    return adata
    
