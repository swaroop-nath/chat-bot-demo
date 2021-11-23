from torch.nn.functional import leaky_relu

class ConfigManager:
    def __init__(self):
        self.configs = {
            'transfomer-all-data': {'d_model': 384, 'num_heads': 8, 'num_enc': 4, 'num_dec': 4, 'd_ff': 512, 
                'dropout_rate': 0.1, 'activation': leaky_relu, 'embedding': 'common'}, 
            'transformer-conv-ai': None, 
            'gru-pytorch': None, 
            'gpt-2': None, 
            'model-dummy': {"d_model": 384, "num_heads": 8, "num_enc": 4, "num_dec": 4, "d_ff": 512, "dropout_rate": 0.1,
                'activation': leaky_relu, 'embedding': 'diff'}
        }

    def get_config(self, model_invoked):
        return self.configs[model_invoked]