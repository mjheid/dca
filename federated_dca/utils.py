from re import X
import torch
import numpy as np
import os.path
import random
from federated_dca.models import ZINBAutoEncoder, NBAutoEncoder
from federated_dca.loss import ZINBLoss, NBLoss
import torch
from torch.utils.data import DataLoader
from federated_dca.datasets import GeneCountData, write_text_matrix

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
        self.train_loss = []
        self.val_loss = []
        self.denoise = params['result']['denoise']
        self.result = params['result']['data']
        self.epoch = params['model_parameters']['epoch']
        self.epoch_count = 0
        self.lr = params['model_parameters']['lr']
        self.batch = params['model_parameters']['batch']
        self.encoder_size = params['model_parameters']['encoder_size']
        self.bottleneck_size = params['model_parameters']['bottleneck_size']
        self.ridge = params['model_parameters']['ridge']
        self.reduce_lr = params['model_parameters']['reduce_lr']
        self.early_stopping = params['model_parameters']['early_stopping']
        self.es_count = 0
        self.name = params['model_parameters']['name']
        self.dataset_path = params['local_dataset']['data']
        self.loginput = params['local_dataset']['loginput']
        self.norminput = params['local_dataset']['norminput']
        self.transpose = params['local_dataset']['transpose']
        self.test_split = params['local_dataset']['test_split']
        self.seed = params['model_parameters']['seed']
        self.param_factor = params['model_parameters']['param_factor']
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dataset = GeneCountData('/mnt/input/'+self.dataset_path, self.device, transpose=self.transpose,
                        loginput=self.loginput, norminput=self.norminput, test_split=self.test_split)
        self.input_size = self.dataset.gene_num
        self.dataset.set_mode('train')
        self.trainDataLoader = DataLoader(self.dataset, batch_size=self.batch, shuffle=True)
        self.dataset.set_mode('val')
        self.valDataLoader = DataLoader(self.dataset, batch_size=self.batch)
        self.model_type = params['model_parameters']['model_type']
        if self.model_type == 'zinb':
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
        seed = self.seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        os.environ['PYTHONHASHSEED'] = f'{seed}'

    def train(self, update, log, id):
        epoch = self.epoch_count
        train_loss = 0
        self.model.train()
        self.dataset.set_mode(self.dataset.train)
        for data, target, size_factor in self.trainDataLoader:

            if self.model_type == 'zinb':
                mean, disp, drop = self.model(data, size_factor)
                loss = self.loss(target, mean, disp, drop)
            else:
                mean, disp = self.model(data)
                loss = self.loss(data, mean, disp)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()

        avg_loss = train_loss / len(self.trainDataLoader)
        log(f'Epoch: {epoch}, Avg train loss: '+"%.6f"%avg_loss)

        val_loss = 0
        with torch.no_grad():
            self.model.eval()
            self.dataset.set_mode(self.dataset.val)
            for data, target, size_factor in self.valDataLoader:
                if self.model_type == 'zinb':
                    mean, disp, drop = self.model(data, size_factor)
                    loss = self.loss(target, mean, disp, drop)
                else:
                    mean, disp = self.model(data)
                    loss = self.loss(data, mean, disp)

                val_loss += loss.item()
            
            avg_loss = val_loss / len(self.valDataLoader)
            self.scheduler.step(avg_loss)
            log(f'Epoch: {epoch}, Avg val loss: '+"%.6f"%avg_loss)
            if avg_loss < self.best_val_loss:
                self.best_val_loss = avg_loss
                torch.save(self.model.state_dict(), '/mnt/output/'+self.name+'.pt')
                self.es_count = 0
            else:
                self.es_count += 1

        self.epoch_count += 1
        if self.epoch - self.epoch_count <= 0:
            self.finished_training = True
    
    def get_weights(self):
        model = self.model
        weights_list = []
        for name, params in model.named_parameters():
            weights_list.append(params.data)
        return weights_list
    
    def set_weights(self, weights):
        model = self.model
        index = 0
        with torch.no_grad():
            for name, params in model.named_parameters():
                params.data = params.data + self.param_factor * (weights[index] - params.data)
                index += 1
    
    def finish(self):
        np.save('/mnt/output/train_loss', self.train_loss)
        np.save('/mnt/output/val_loss', self.val_loss)
        if self.denoise:
            self.model.load_state_dict(torch.load('/mnt/output/'+self.name+'.pt'))
            self.model.eval()
            self.dataset.set_mode(self.dataset.test)
            eval_dataloader = DataLoader(self.dataset, batch_size=self.dataset.__len__())
            for data, target, size_factor in eval_dataloader:
                if self.model_type == 'zinb':
                    mean, disp, drop = self.model(data, size_factor)
                    loss = self.loss(target, mean, disp, drop)
                else:
                    mean, disp = self.model(data, size_factor)
                    loss = self.loss(target, mean, disp)
            adata = self.dataset.adata.copy()
            adata.X = mean.detach().numpy()
            colnames = adata.var_names.values
            rownames = adata.obs_names.values
            write_text_matrix(adata.X,
                              '/mnt/output/'+self.result,
                              rownames=rownames, colnames=colnames, transpose=True)



def average_model_params(model_params):
    params = []
    for i in list(range(len(model_params[0]))):
        weight = 0
        for model in model_params:
            if type(weight) is int:
                weight = model[i]
            else:
                weight += model[i]
        
        weight = weight / len(model_params)
        params.append(weight)
        
    return params
