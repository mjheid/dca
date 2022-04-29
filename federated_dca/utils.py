import torch
import numpy as np
import pandas as pd
import os.path
import random
from federated_dca.models import ZINBAutoEncoder, NBAutoEncoder
from federated_dca.loss import ZINBLoss, NBLoss
import torch
from torch.utils.data import DataLoader
from federated_dca.datasets import GeneCountData, write_text_matrix
import glob

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
        self.filter_min_counts = params['local_dataset']['filter_min_counts']
        self.size_factor = params['local_dataset']['size_factor']
        self.seed = params['model_parameters']['seed']
        self.param_factor = params['model_parameters']['param_factor']
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dataset = GeneCountData('/mnt/input/'+self.dataset_path, self.device, transpose=self.transpose,
                        loginput=self.loginput, norminput=self.norminput, test_split=self.test_split,
                        filter_min_counts=self.filter_min_counts, size_factor=self.size_factor)
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

def aggregate(global_model, client_models, client_lens, param_factor):
    total = sum(client_lens)
    n = len(client_models)
    global_dict = global_model.state_dict()
    for key in global_dict.keys():
        global_dict[key] = torch.stack([client_models[i].state_dict()[key].float()*(n*client_lens[i]/total) for i in range(len(client_models))], 0).mean(0)
    global_model.load_state_dict(global_dict)
    for model in client_models:
        model_dict = model.state_dict()
        for key in global_dict.keys():
            model_dict[key] = model_dict[key] + param_factor * (global_dict[key] - model_dict[key])
        model.load_state_dict(model_dict)

def train_client(model,
                trainDataLoader, loss, optimizer,
                valDataLoader, e, aggregate_flag, es_count,
                earlystopping_own, earlystopping_prev, earlystopping_cond, 
                early_stopping, EPOCH, dataset, name, client, scheduler, modeltype):

    best_val_loss = float('inf')
    for epoch in range(EPOCH):
        if  earlystopping_cond.is_set():
            train_loss = 0
            model.train()
            dataset.set_mode(dataset.train)
            print(f'Epoch: {epoch}')
            for data, target, size_factor in trainDataLoader:
                if modeltype == 'zinb':
                    mean, disp, drop = model(data, size_factor)
                    l = loss(target, mean, disp, drop)
                else:
                    mean, disp = model(data)
                    l = loss(data, mean, disp)

                optimizer.zero_grad()
                l.backward()
                optimizer.step()

                train_loss += l.item()

            avg_loss = train_loss / len(trainDataLoader)
            print(f'Client: {client}, Avg train loss: {avg_loss}')

            val_loss = 0
            with torch.no_grad():
                model.eval()
                dataset.set_mode(dataset.val)
                for data, target, size_factor in valDataLoader:
                    if modeltype == 'zinb':
                        mean, disp, drop = model(data, size_factor)
                        l = loss(target, mean, disp, drop)
                    else:
                        mean, disp = model(data)
                        l = loss(data, mean, disp)

                    val_loss += l.item()
                
                avg_loss = val_loss / len(valDataLoader)
                scheduler.step(avg_loss)
                print(f'Client: {client}, Avg val loss: {avg_loss}')
                if avg_loss < best_val_loss:
                    best_val_loss = avg_loss
                    es_count = 0
                    earlystopping_own.set()
                    torch.save(model.state_dict(), 'data/checkpoints/'+name+f'_{client}.pt')
                else:
                    es_count += 1
                if client == 0 and es_count >= early_stopping:
                    earlystopping_own.clear()
                elif es_count >= early_stopping and not earlystopping_prev.is_set():
                    earlystopping_own.clear()
                e.set()
                aggregate_flag.wait()
                e.clear()
        else:
            e.set()
            earlystopping_own.clear()
    print(f'Client: {client}, Best val loss: {best_val_loss}')


def global_agg(client_models,
                global_model, loss, globalDataLoader,
                name, client_lens, param_factor, aggregate_flag, events,
                es_count, earlystopping_own, earlystopping_prev, early_stopping, EPOCH, modeltype):

    best_val_loss = float('inf')
    avg_loss = float('inf')
    for epoch in range(EPOCH):
        if earlystopping_own.is_set() and earlystopping_prev.is_set():
            for event in events:
                event.wait()
            with torch.no_grad():
                test_loss = 0
                aggregate(global_model, client_models, client_lens, param_factor)
                aggregate_flag.set()
                aggregate_flag.clear()
                for data, target, size_factor in globalDataLoader:
                    if modeltype == 'zinb':
                        mean, disp, drop = global_model(data, size_factor)
                        l = loss(target, mean, disp, drop)
                    else:
                        mean, disp = global_model(data)
                        l = loss(data, mean, disp)

                    test_loss += l.item()
                avg_loss = test_loss / len(globalDataLoader)
                print(f'Global avg test loss: {avg_loss}')
                if avg_loss < best_val_loss:
                    best_val_loss = avg_loss
                    es_count = 0
                    earlystopping_own.set()
                    torch.save(global_model.state_dict(), 'data/checkpoints/'+name+f'_global.pt')
                else:
                    es_count += 1
                if es_count >= early_stopping and not earlystopping_prev.is_set():
                    earlystopping_own.clear()
        else:
            aggregate_flag.set()
            earlystopping_own.clear()
    print(f'Global, Best val loss: {best_val_loss}')


def denoise(model, name, path, dataset, modeltype, result, outputdir):
    model.load_state_dict(torch.load(path+name+'.pt'))
    model.eval()
    dataset.set_mode(dataset.test)
    eval_dataloader = DataLoader(dataset, batch_size=dataset.__len__())
    for data, target, size_factor in eval_dataloader:
        if modeltype == 'zinb':
            mean, disp, drop = model(data, size_factor)
        else:
            mean, disp = model(data, size_factor)
    adata = dataset.adata.copy()
    adata.X = mean.detach().numpy()
    colnames = adata.var_names.values
    rownames = adata.obs_names.values
    write_text_matrix(adata.X,
                        outputdir+ result,
                        rownames=rownames, colnames=colnames, transpose=True)


def sort_paths(paths, client=True, ptrn_data='data', ptrn_norm_data='norm', ptrn_anno='anno'):
    if not client:
        data_path = glob.glob(paths+f'*{ptrn_data}*')
        norm_data_path = glob.glob(paths+f'*{ptrn_norm_data}*')
        anno_path = glob.glob(paths+f'*{ptrn_anno}*')
        return [data_path[0], norm_data_path[0], anno_path[0]]
    else:
        ordered_path_list = []
        for i in range(int(len(glob.glob(paths+'*.csv'))/3)):
            i +=1
            data_path = glob.glob(paths+f'*{ptrn_data}*{i}*')
            norm_data_path = glob.glob(paths+f'*{ptrn_norm_data}*{i}*')
            anno_path = glob.glob(paths+f'*{ptrn_anno}*{i}*')
            ordered_path_list.append([data_path[0], norm_data_path[0], anno_path[0]])
        return ordered_path_list


def gen_iid_client_data(path, num_clients, name='', ptrn_data='data', ptrn_norm_data='norm', outputpath='data/input/', idx='Group'):
    data = []
    norm = []
    anno = []
    num_classes = int(len(glob.glob(path+f'/*{ptrn_data}*')))
    for i in range(num_classes):
        i += 1
        data.append(glob.glob(path+f'/*{ptrn_data}*{i}*')[0])
        norm.append(glob.glob(path+f'/*{ptrn_norm_data}*{i}*')[0])
        anno.append(glob.glob(path+f'/*anno*{i}*')[0])
    sorted_data_splits = []
    for i in range(len(data)):
        group_true = pd.read_csv(data[i]).set_index(idx)
        group_norm = pd.read_csv(norm[i]).set_index(idx)
        group_anno = pd.read_csv(anno[i])
        split_len = int(group_norm.shape[0] / num_clients)
        index = 0
        clients_norm = []
        clients_true = []
        clients_anno = []
        for i in range(num_clients):
            if i < num_clients-1:
                split_true = pd.DataFrame(group_true[index:(i+1)*split_len])
                split_norm = pd.DataFrame(group_norm[index:(i+1)*split_len])
                split_anno = pd.DataFrame(group_anno[index:(i+1)*split_len])
            else:
                split_true = pd.DataFrame(group_true[index:])
                split_norm = pd.DataFrame(group_norm[index:])
                split_anno = pd.DataFrame(group_anno[index:])
            clients_norm.append(split_true)
            clients_true.append(split_norm)
            clients_anno.append(split_anno)
            index = (i+1)*split_len
        sorted_data_splits.append([clients_true, clients_norm, clients_anno])
    for i in range(num_clients):
        client_norm_df = pd.DataFrame()
        client_true_df = pd.DataFrame()
        client_anno_df = pd.DataFrame()
        for j in range(num_classes):
            client_true_df = pd.concat([client_true_df, sorted_data_splits[j][1][i]]) #TODO: y 1? error somewhere
            client_norm_df = pd.concat([client_norm_df, sorted_data_splits[j][0][i]])
            client_anno_df = pd.concat([client_anno_df, sorted_data_splits[j][2][i]])
        client_norm_df.to_csv(outputpath+'norm'+name+'_'+str(i+1)+'.csv')
        client_true_df.to_csv(outputpath+'data'+name+'_'+str(i+1)+'.csv')
        client_anno_df.to_csv(outputpath+'anno'+name+'_'+str(i+1)+'.csv')
        