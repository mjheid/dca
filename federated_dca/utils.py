import torch
import numpy as np
import os.path
import random
from federated_dca.models import ZINBAutoEncoder, NBAutoEncoder
from federated_dca.loss import ZINBLoss, NBLoss
import torch
from torch.utils.data import DataLoader
from federated_dca.datasets import GeneCountData

def save_and_load_init_model(model, mname, base='data/checkpoints/'):
    if os.path.exists(os.path.abspath('.') + base + '/init_' + mname + '.npy'):
        saved_params = np.load('init_' + mname + '.npy', allow_pickle=True)
        with torch.no_grad():
            for name, params in model.named_parameters():
                sh = params.shape
                if name == 'encoder.0.weight':
                    params.data = torch.from_numpy(np.transpose(saved_params[0]))
                elif name == 'encoder.0.bias':
                    params.data = torch.from_numpy(saved_params[1])
                elif name == 'bottleneck.0.weight':
                    params.data = torch.from_numpy(np.transpose(saved_params[2]))
                elif name == 'bottleneck.0.bias':
                    params.data = torch.from_numpy(saved_params[3])
                elif name == 'decoder.0.weight':
                    params.data = torch.from_numpy(np.transpose(saved_params[4]))
                elif name == 'decoder.0.bias':
                    params.data = torch.from_numpy(saved_params[5])
                elif name == 'mean.0.weight':
                    params.data = torch.from_numpy(np.transpose(saved_params[6]))
                elif name == 'mean.0.bias':
                    params.data = torch.from_numpy(saved_params[7])
                elif name == 'disp.0.weight':
                    params.data = torch.from_numpy(np.transpose(saved_params[8]))
                elif name == 'disp.0.bias':
                    params.data = torch.from_numpy(saved_params[9])
                elif name == 'drop.0.weight':
                    params.data = torch.from_numpy(np.transpose(saved_params[10]))
                elif name == 'drop.0.bias':
                    params.data = torch.from_numpy(saved_params[11])

        return model
    else:
        print('No original dca params to load!!!')
        params_to_save = []
        for name, params in model.named_parameters():
            sh = params.shape
            if name == 'encoder.0.weight':
                params_to_save.append(np.transpose(params.detach().numpy()))
            elif name == 'encoder.0.bias':
                params_to_save.append(params.detach().numpy())
            elif name == 'bottleneck.0.weight':
                params_to_save.append(np.transpose(params.detach().numpy()))
            elif name == 'bottleneck.0.bias':
               params_to_save.append(params.detach().numpy())
            elif name == 'decoder.0.weight':
                params_to_save.append(np.transpose(params.detach().numpy()))
            elif name == 'decoder.0.bias':
                params_to_save.append(params.detach().numpy())
            elif name == 'mean.0.weight':
                params_to_save.append(np.transpose(params.detach().numpy()))
            elif name == 'mean.0.bias':
                params_to_save.append(params.detach().numpy())
            elif name == 'disp.0.weight':
                params_to_save.append(np.transpose(params.detach().numpy()))
            elif name == 'disp.0.bias':
                params_to_save.append(params.detach().numpy())
            elif name == 'drop.0.weight':
                params_to_save.append(np.transpose(params.detach().numpy()))
            elif name == 'drop.0.bias':
                params_to_save.append(params.detach().numpy())
        np.save(base + '1e_' + mname, params_to_save)
        return model

def load_params(path):
    pass

class trainInstince():
    def __init__(self, params) -> None:
        self.config = params
        self.epoch = params['model_parameters']['epoch']
        self.lr = params['model_parameters']['lr']
        self.batch = params['batch']
        self.encoder_size = params['model_parameters']['encoder_size']
        self.bottleneck_size = params['model_parameters']['bottleneck_size']
        self.ridge = params['model_parameters']['ridge']
        self.reduce_lr = params['reduce_lr']
        self.early_stopping = params['model_parameters']['early_stopping']
        self.name = params['model_parameters']['name']
        self.dataset_path = params['local_dataset']['data']
        self.loginput = params['local_dataset']['loginput']
        self.norminput = params['local_dataset']['norminput']
        self.transpose = params['local_dataset']['transpose']
        self.seed = params['model_parameters']['seed']
        self.param_factor = params['model_parameters']['param_factor']
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dataset = GeneCountData(self.dataset_path, self.device, transpose=self.transpose,
                        loginput=self.loginput, norminput=self.norminput)
        self.input_size = self.dataset.gene_num
        self.dataset.set_mode('train')
        self.trainDataLoader = DataLoader(self.dataset, batch_size=self.batchsize, shuffle=True)
        self.dataset.set_mode('val')
        self.valDataLoader = DataLoader(self.dataset, batch_size=self.batchsize)
        self.model_type = params['model_parameters']['model']
        if params['model'] == 'zinb':
            self.model = ZINBAutoEncoder(input_size=self.input_size, encoder_size=self.encoder_size,
                        bottleneck_size=self.bottleneck_size).to(self.device)
            self.loss = ZINBLoss(ridge_lambda=self.ridge)
        else:
            self.model = NBAutoEncoder(input_size=self.input_size, encoder_size=self.encoder_size,
                        bottleneck_size=self.bottleneck_size).to(self.device)
            self.loss = NBLoss()
        self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', patience=self.reduce_lr)
        self.finished_training = False
        self.best_val_loss = float('inf')

    def train(self, update, log, id):
        epoch = self.epoch
        seed = self.seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        os.environ['PYTHONHASHSEED'] = '0'

        self.model.train()
        self.dataset.set_mode('train')

        train_loss = 0

        for data, target, size_factor in self.trainDataLoader:
            if self.model_type == 'zinb':
                mean, disp, drop = self.model(data, size_factor)
                loss = self.loss(target, mean, disp, drop)
            else:
                mean, disp = self.model(data, size_factor)
                loss = self.loss(target, mean, disp)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
        
        avg_loss = train_loss / len(self.trainDataLoader)
        update(f'{id}: Epoch: {epoch}, Avg train loss: {avg_loss}')

        val_loss = 0

        with torch.no_grad():
            self.model.eval()
            self.dataset.set_mode('test')
            for data, target, size_factor in self.valData_loader:
                if self.model_type == 'zinb':
                    mean, disp, drop = self.model(data, size_factor)
                    loss = self.loss(target, mean, disp, drop)
                else:
                    mean, disp = self.model(data, size_factor)
                    loss = self.loss(target, mean, disp)
                val_loss += loss.item()
            avg_loss = val_loss / len(self.valDataLoader)
            update(f'{id}: Epoch: {epoch}, Avg val loss: {avg_loss}')
            self.scheduler.step(avg_loss)

            if avg_loss < self.best_val_loss:
                torch.save(self.model.state_dict(), self.name + '.pt')
        
        self.epoch -= 1
        if epoch <= 0:
            self.finished_training = True
    
    def get_weights(self):
        model = self.model
        weights_list = []
        for name, params in model.named_parameters():
            weights_list.append(params)
        return weights_list
    
    def set_weights(self, weights):
        model = self.model
        index = 0
        with torch.no_grad():
            for name, params in model.named_parameters():
                params.data = params.data + self.param_factor * (weights[index] - params.data)
                index += 1


def average_model_params(model_params):
    params = []
    for i in list(range(len(model_params[0]))):
        weight = 0
        for model in model_params:
            if weight == 0:
                weight = model[i]
            else:
                weight += model[i]
        
        weight = weight / len(model_params)
        params.append(weight)
        
    return params
