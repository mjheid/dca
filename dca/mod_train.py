import torch
from torch.utils.data import DataLoader
from datasets import GeneCountData
from my_loss import ZINBLoss
from models import ZINBAutoEncoder

def train(path='', EPOCH=100, lr=0.001):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_dataset = GeneCountData(path, device, train=0.9)
    val_dataset = GeneCountData(path, device, val=0.1)
    input_size = train_dataset.gene_num

    trainDataLoader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    valDataLoader = DataLoader(val_dataset, batch_size=32)
    dca = ZINBAutoEncoder(input_size=input_size, encoder_size=64, bottleneck_size=32).to(device)
    optimizer = torch.optim.RMSprop(dca.parameters(), lr=lr)
    loss_zinb = ZINBLoss()
    #loss_zinb = torch.nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min')

    for epoch in range(EPOCH):
        train_loss = 0
        for data, size_factor in trainDataLoader:

            mean, disp, drop = dca(data)
            loss = loss_zinb(data, mean, disp, drop)
            #loss = loss_zinb(mean, data)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_loss = train_loss / len(trainDataLoader)
        print(f'Epoch: {epoch}, Avg train loss: {avg_loss}')

        val_loss = 0
        with torch.no_grad():
            for data, size_factor in valDataLoader:
                mean, disp, drop = dca(data)
                loss = loss_zinb(data, mean, disp, drop)
                #loss = loss_zinb(mean, data)

                val_loss += loss.item()
            
            avg_loss = val_loss / len(valDataLoader)
            scheduler.step(avg_loss)
            print(f'Epoch: {epoch}, Avg val loss: {avg_loss}')

train('/home/kaies/csb/dca/data/francesconi/francesconi_withDropout.csv')