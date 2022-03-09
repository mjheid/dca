import torch
import numpy as np
import os.path

def save_and_load_init_model(model, name):
    if os.path.exists(os.path.abspath('.') + '/init_' + name + '.npy'):
        saved_params = np.load('init_' + name + '.npy', allow_pickle=True)
        with torch.no_grad():
            for name, params in model.named_parameters():
                print(name)
                print(params)

        return model
    else:
        print('No original dca params to load!!!')
        return model