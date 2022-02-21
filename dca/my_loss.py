import torch

def _nan2zero(x):
    return torch.where(torch.isnan(x), torch.zeros_like(x), x)

def _nan2inf(x):
    return torch.where(torch.isnan(x), torch.zeros_like(x)+torch.tensor(['inf']), x)

def _nelem(x):
    nelem = torch.sum(torch.nan_to_num(x))
    return torch.where(torch.equal(nelem, 0.), 1., nelem).type(x.dtype)

def _reduce_mean(x):
    nelem = _nelem(x)
    x = _nan2zero(x)
    return torch.div(torch.sum(x), nelem)

class NBLoss(torch.nn.Module):
    """
    point mass at zero representing excess zero values in data
    """
    def __init__(self, masking=False, scale_factor=1.0, debug=False) -> None:
        super().__init__()

        self.eps = 1e-10
        self.scale_factor = 1.0
        self.debug = debug
        self.masking = masking

    def forward(self, x, mean, theta, red_mean=True):
        
        mean = mean * self.scale_factor

        if self.masking:
            nelem = _nelem(x)
            x = _nan2zero(x)
        
        theta = torch.minimum(theta, 1e6)

        t1 = torch.lgamma(theta+self.eps) + torch.lgamma(x+1.0) - torch.lgamma(x+theta+self.eps)
        t2 = (theta+x) * torch.log(1.0 + (mean/(theta+self.eps))) + (x * (torch.log(theta+self.eps) - torch.log(mean+self.eps)))

        if self.debug:
            print('No debugging implemented as of yet')
            final = t1 + t2
        else:
            final = t1 + t2
        
        final = _nan2inf(final)

        if red_mean:
            if self.masking:
                final = _reduce_mean(final)
            else:
                final = torch.mean(final)
        
        return final


class ZINBLoss(NBLoss):
    """
    point mass at zero representing excess zero values in data
    """
    def __init__(self, ridge_lambda=0.0, **kwargs) -> None:
        super().__init__(self, **kwargs)
        self.ridge_lambda = ridge_lambda
    
    def forward(self, x, mean, theta, pi, red_mean=True):
        nb_case = super().forward(x, mean, theta, red_mean=False) - torch.log(1.0-pi+self.eps)

        mean = mean = self.scale_factor
        theta = torch.minimum(theta, 1e6)

        zero_nb = torch.pow(theta/(theta+mean+self.eps), theta)
        zero_case = -torch.log(pi + ((1.0-pi)*zero_nb)+self.eps)
        result = torch.where(torch.less(mean, 1e-8), zero_case, nb_case)
        ridge = self.ridge_lambda * torch.square(pi)
        result += ridge

        if red_mean:
            if self.masking:
                result = _reduce_mean(result)
            else:
                result = torch.mean(result)
        
        result = _nan2inf(result)

        if self.debug:
            print('Implement debug stuff!')
        
        return result
