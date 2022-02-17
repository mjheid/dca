import torch
from torch.utils.data import DataLoader
from datasets import GeneCountData
from my_loss import ZINBLoss
from models import ZINBAutoEncoder

def train(epoch):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    path = 'data/francesconi/francesconi_withDropout.csv'
    dataset = GeneCountData(path, device)
    input_size = dataset.gene_num

    trainDataLoader = DataLoader(dataset, batch_size=32, shuffle=True)

    dca = ZINBAutoEncoder(input_size, encoder_size=64, bottleneck_size=32).to(device)

