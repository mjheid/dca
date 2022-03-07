import torch

class MeanAct(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x):
        return torch.clamp(torch.exp(x), 1e-5, 1e6)

class DispAct(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.softplus = torch.nn.Softplus()
    
    def forward(self, x):
        return torch.clamp(self.softplus(x), 1e-4, 1e4)

def init_weights(layer):
    if isinstance(layer, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(layer.weight)

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
        
        self.encoder.apply(init_weights)
        self.decoder.apply(init_weights)
        self.bottleneck.apply(init_weights)
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x)

        return x


class NBAutoEncoder(AutoEncoder):
    def __init__(self, input_size, encoder_size, bottleneck_size):
        super().__init__(input_size, encoder_size, bottleneck_size)

        self.mean = torch.nn.Sequential(
            torch.nn.Linear(self.encoder_size, self.input_size),
            MeanAct()
            )

        self.disp = torch.nn.Sequential(
            torch.nn.Linear(self.encoder_size, self.input_size),
            DispAct()
            )
        
        self.mean.apply(init_weights)
        self.disp.apply(init_weights)
    
    def forward(self, x):
        x = super().forward(x)

        mean = self.mean(x)
        disp = self.disp(x)

        return mean, disp


        

class ZINBAutoEncoder(AutoEncoder):
    def __init__(self, input_size, encoder_size, bottleneck_size):
        super().__init__(input_size, encoder_size, bottleneck_size)

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
        
        self.mean.apply(init_weights)
        self.disp.apply(init_weights)
        self.drop.apply(init_weights)
    
    def forward(self, x):
        x = super().forward(x)

        mean = self.mean(x)
        disp = self.disp(x)
        drop = self.drop(x)

        return mean, disp, drop
