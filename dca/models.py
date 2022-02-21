import torch

def MeanAct(x):
    return torch.clamp(torch.exp(x), 1e-5, 1e6)

def DispAct(x):
    return torch.clamp(torch.nn.Softplus(x), 1e-4, 1e4)

class AutoEncoder(torch.nn.Module):
    def __init__(self, input_size, encoder_size, bottleneck_size):
        super().__init__()

        self.input_size = input_size
        self.encoder_size = encoder_size
        self.bottleneck_size = bottleneck_size

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(self.input_size, self.encoder_size),
            torch.nn.ReLU()
            )
        
        self.bottleneck = torch.nn.Sequential(
            torch.nn.Linear(self.encoder_size, self.bottleneck_size),
            torch.nn.ReLU()
            )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(self.bottleneck_size, self.encoder_size),
            torch.nn.ReLU()
            )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x)

        return x


class NBAutoEncoder(AutoEncoder):
    def __init__(self, **kwds):
        super().__init__(**kwds)

        self.mean = torch.nn.Sequential(
            torch.nn.Linear(self.encoder_size, self.input_size),
            MeanAct()
            )

        self.disp = torch.nn.Sequential(
            torch.nn.Linear(self.encoder_size, self.input_size),
            DispAct()
            )
    
    def forward(self, x):
        x = super().forward(x)

        mean = self.mean(x)
        disp = self.disp(x)

        return mean, disp


        

class ZINBAutoEncoder(torch.nn.Module):
    def __init__(self,**kwds):
        super().__init__(**kwds)

        self.mean = torch.nn.Sequential(
            torch.nn.Linear(self.encoder_size, self.input_size),
            MeanAct()
            )

        self.disp = torch.nn.Sequential(
            torch.nn.Linear(self.encoder_size, self.input_size),
            DispAct()
            )
            
        self.drop = torch.nn.Sequential(
            torch.nn.Linear(self.encoder_size, self.input_size),
            torch.nn.Sigmoid()
            )
    
    def forward(self, x):
        x = super().forward(x)

        mean = self.mean(x)
        disp = self.disp(x)
        drop = self.drop(x)

        return mean, disp, drop
