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
                    params.data = torch.from_numpy(saved_params[0].reshape((sh[0], sh[1])))
                elif name == 'encoder.0.bias':
                    params.data = torch.from_numpy(saved_params[1])
                elif name == 'bottleneck.0.weight':
                    params.data = torch.from_numpy(saved_params[6].reshape((sh[0], sh[1])))
                elif name == 'bottleneck.0.bias':
                    params.data = torch.from_numpy(saved_params[7])
                elif name == 'decoder.0.weight':
                    params.data = torch.from_numpy(saved_params[12].reshape((sh[0], sh[1])))
                elif name == 'decoder.0.bias':
                    params.data = torch.from_numpy(saved_params[13])
                elif name == 'mean.0.weight':
                    params.data = torch.from_numpy(saved_params[18].reshape((sh[0], sh[1])))
                elif name == 'mean.0.bias':
                    params.data = torch.from_numpy(saved_params[19])
                elif name == 'disp.0.weight':
                    params.data = torch.from_numpy(saved_params[20].reshape((sh[0], sh[1])))
                elif name == 'disp.0.bias':
                    params.data = torch.from_numpy(saved_params[21])
                elif name == 'drop.0.weight':
                    params.data = torch.from_numpy(saved_params[22].reshape((sh[0], sh[1])))
                elif name == 'drop.0.bias':
                    params.data = torch.from_numpy(saved_params[23])

        return model
    else:
        print('No original dca params to load!!!')
        params_to_save = []
        for name, params in model.named_parameters():
            sh = params.shape
            if name == 'encoder.0.weight':
                params_to_save.append(params.detach().numpy().reshape((sh[1], sh[0])))
            elif name == 'encoder.0.bias':
                params_to_save.append(params.detach().numpy())
            elif name == 'bottleneck.0.weight':
                params_to_save.append(params.detach().numpy().reshape((sh[1], sh[0])))
            elif name == 'bottleneck.0.bias':
               params_to_save.append(params.detach().numpy())
            elif name == 'decoder.0.weight':
                params_to_save.append(params.detach().numpy().reshape((sh[1], sh[0])))
            elif name == 'decoder.0.bias':
                params_to_save.append(params.detach().numpy())
            elif name == 'mean.0.weight':
                params_to_save.append(params.detach().numpy().reshape((sh[1], sh[0])))
            elif name == 'mean.0.bias':
                params_to_save.append(params.detach().numpy())
            elif name == 'disp.0.weight':
                params_to_save.append(params.detach().numpy().reshape((sh[1], sh[0])))
            elif name == 'disp.0.bias':
                params_to_save.append(params.detach().numpy())
            elif name == 'drop.0.weight':
                params_to_save.append(params.detach().numpy().reshape((sh[1], sh[0])))
            elif name == 'drop.0.bias':
                params_to_save.append(params.detach().numpy())
        np.save('1e_' + mname, params_to_save)
        return model