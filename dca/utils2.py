import torch
import numpy as np
import os.path

def save_and_load_init_model(model, name):
    if os.path.exists(os.path.abspath('.') + '/init_' + name + '.npy'):
        saved_params = np.load('init_' + name + '.npy', allow_pickle=True)
        with torch.no_grad():
            for name, params in model.named_parameters():
                sh = params.shape
                if name == 'encoder.0.weight':
                    params.data = torch.from_numpy(saved_params[0].reshape((sh[0], sh[1])))
                elif name == 'encoder.0.bias':
                    params.data = torch.from_numpy(saved_params[1])
                elif name == 'bottleneck.0.weight':
                    params.data = torch.from_numpy(saved_params[2].reshape((sh[0], sh[1])))
                elif name == 'bottleneck.0.bias':
                    params.data = torch.from_numpy(saved_params[3])
                elif name == 'decoder.0.weight':
                    params.data = torch.from_numpy(saved_params[4].reshape((sh[0], sh[1])))
                elif name == 'decoder.0.bias':
                    params.data = torch.from_numpy(saved_params[5])
                elif name == 'mean.0.weight':
                    params.data = torch.from_numpy(saved_params[6].reshape((sh[0], sh[1])))
                elif name == 'mean.0.bias':
                    params.data = torch.from_numpy(saved_params[7])
                elif name == 'disp.0.weight':
                    params.data = torch.from_numpy(saved_params[8].reshape((sh[0], sh[1])))
                elif name == 'disp.0.bias':
                    params.data = torch.from_numpy(saved_params[9])
                elif name == 'drop.0.weight':
                    params.data = torch.from_numpy(saved_params[10].reshape((sh[0], sh[1])))
                elif name == 'drop.0.bias':
                    params.data = torch.from_numpy(saved_params[11])

        return model
    else:
        print('No original dca params to load!!!')
        params_to_save = []
        for name, params in model.named_parameters():
            sh = params.shape
            if name == 'encoder.0.weight':
                params_to_save.append(params.detach().numpy().reshape((sh[1], sh[0])))
            elif name == 'encoder.0.bias':
                params_to_save.append(params)
            elif name == 'bottleneck.0.weight':
                params_to_save.append(params.detach().numpy().reshape((sh[1], sh[0])))
            elif name == 'bottleneck.0.bias':
               params_to_save.append(params)
            elif name == 'decoder.0.weight':
                params_to_save.append(params.detach().numpy().reshape((sh[1], sh[0])))
            elif name == 'decoder.0.bias':
                params_to_save.append(params)
            elif name == 'mean.0.weight':
                params_to_save.append(params.detach().numpy().reshape((sh[1], sh[0])))
            elif name == 'mean.0.bias':
                params_to_save.append(params)
            elif name == 'disp.0.weight':
                params_to_save.append(params.detach().numpy().reshape((sh[1], sh[0])))
            elif name == 'disp.0.bias':
                params_to_save.append(params)
            elif name == 'drop.0.weight':
                params_to_save.append(params.detach().numpy().reshape((sh[1], sh[0])))
            elif name == 'drop.0.bias':
                params_to_save.append(params)
        np.save('1eo_' + name, params_to_save)
        return model