import torch
import numpy as np
import os.path

def save_and_load_init_model(model, mname):
    if os.path.exists(os.path.abspath('.') + '/init_' + mname + '.npy'):
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
        np.save('1e_' + mname, params_to_save)
        return model