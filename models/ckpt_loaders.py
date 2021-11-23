import os
import json
import torch
from torch._C import device

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

class CKPTManagerPyTorch:
    def __init__(self, ckpt_base_dir, model_name, config_file):
        self.model_name = model_name
        self.config_file = config_file
        self.path_template = ckpt_base_dir + '/' + self.model_name + '{}.pt'
        self.ckpt_base_dir = ckpt_base_dir

    def load_ckpt(self, model):
        with open(self.ckpt_base_dir + '/' + self.config_file, 'r') as file:
            config = json.load(file)

        if not config['pickled']:
            return model
        else:
            path = self.path_template.format(config['latest'])
            state = torch.load(path, map_location=device)
            print('Model trained for {} epochs loaded, train loss: {}'.format(state['epochs'], state['train-loss']))

            model.load_state_dict(state['model-state-dict'])

            return model

    def save_ckpt(self, state):
        with open(self.ckpt_base_dir + '/' + self.config_file, 'r') as file:
            config = json.load(file)
        latest_save = config['latest']

        torch.save(state, self.path_template.format(latest_save+1))

        with open(self.ckpt_base_dir + '/' + self.config_file, 'w') as file:
            meta_data = {'pickled': True, 'latest': latest_save+1}
            json.dump(meta_data, file)