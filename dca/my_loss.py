from turtle import forward
import torch

class NBLoss(torch.nn.Module):
    """
    point mass at zero representing excess zero values in data
    """
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x, mean, theta):
        theta = torch.minimum(theta, 1e6)

        mult1 = torch.lgamma(x + theta) / torch.lgamma(theta)
        mult2 = torch.pow(theta / (theta + mean), theta)
        mult3 = torch.pow(mean / (theta + mean), x)

        loss = mult1 * mult2 + mult3

        return loss


class ZINBLoss(torch.nn.Module):
    """
    point mass at zero representing excess zero values in data
    """
    def __init__(self, delta) -> None:
        super().__init__()
        
        self.delta = delta
        self.nb = NBLoss()
    
    def forward(self, x, pi, mean, theta, scale_factor):
        mean = mean * scale_factor

        nb = self.nb(x, mean, theta)
        # TODO: delta
        zinb = pi * self.delta * x + (1 - pi) * nb
        # TODO: how loss calculated?
        loss = torch.mean(zinb)
        idx = torch.argmax(zinb)
        pi_max, mean_max, theta_max = pi[idx], mean[idx], theta[idx]

        return loss