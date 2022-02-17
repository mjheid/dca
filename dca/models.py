from turtle import forward
import torch

def MeanAct(x):
    return torch.clamp(torch.exp(x), 1e-5, 1e6)

def DispAct(x):
    return torch.clamp(torch.nn.Softplus(x), 1e-4, 1e4)

class ZINBAutoEncoder(torch.nn.Module):
    def __init__(self, input_size, encoder_size, bottleneck_size):
        super().__init__()

        self.input_size = input_size

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(self.input_size, encoder_size),
            torch.nn.ReLU()
            )
        
        self.bottleneck = torch.nn.Sequential(
            torch.nn.Linear(encoder_size, bottleneck_size),
            torch.nn.ReLU()
            )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(bottleneck_size, encoder_size),
            torch.nn.ReLU()
            )

        self.mean = torch.nn.Sequential(
            torch.nn.Linear(encoder_size, self.input_size),
            MeanAct()
            )

        self.disp = torch.nn.Sequential(
            torch.nn.Linear(encoder_size, self.input_size),
            DispAct()
            )
            
        self.drop = torch.nn.Sequential(
            torch.nn.Linear(encoder_size, self.input_size),
            torch.nn.Sigmoid()
            )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x)

        mean = self.mean(x)
        disp = self.disp(x)
        drop = self.drop(x)

        return mean, disp, drop
