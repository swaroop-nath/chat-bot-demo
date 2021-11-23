from typing import List
from models.models_pytorch import ConvModelT
from models.config_manager import ConfigManager
from models.ckpt_loaders import CKPTManagerPyTorch
import tensorflow as tf

from models.models_tensorflow import Transformer

def generate_response(enc_inp, dec_inp, model_invoked, vocab_mapper) -> str:
    model = load_model(model_invoked, vocab_size=len(vocab_mapper))
    response = predict_response(enc_inp, dec_inp, model, vocab_mapper)
    return post_process(response)

def load_model(model_invoked, vocab_size):
    config_mgr = ConfigManager()
    model_config = config_mgr.get_config(model_invoked)
    ckpt_path = './checkpoints/{}'
    
    if model_invoked == 'transformer-all-data':
        use_common_enc = model_config['embedding'] == 'common'
        model_name = 'transformer-no-ctxt'
        config_file = 'ckpt.json'

        ckpt_mgr = CKPTManagerPyTorch(ckpt_base_dir=ckpt_path.format(model_invoked), model_name=model_name, config_file=config_file)

        model = ConvModelT(
            d_model=model_config['d_model'],
            num_heads=model_config['num_heads'],
            num_enc=model_config['num_enc'],
            num_dec=model_config['num_dec'],
            d_ff=model_config['d_ff'],
            dropout_rate=model_config['dropout_rate'],
            activation=model_config['activation'],
            vocab_size=vocab_size,
            use_common_enc=use_common_enc
        )
        model = ckpt_mgr.load_ckpt(model)
        model.eval()

    elif model_invoked == 'transformer-conv-ai':
        model = Transformer(num_enc_layers=model_config['num_enc'], num_dec_layers=model_config['num_dec'], 
                                   d_model=model_config['d_model'], d_ff=model_config['d_ff'], num_heads=model_config['num_heads'], 
                                   input_vocab_size=vocab_size, output_vocab_size=vocab_size)
        ckpt = tf.train.Checkpoint(transformer=model)

        ckpt_manager = tf.train.CheckpointManager(ckpt, ckpt_path.format(model_invoked), max_to_keep=10)

        # if a checkpoint exists, restore the latest checkpoint.
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print('Latest checkpoint restored!!')

    assert model is not None
    return model

def predict_response(enc_inp, dec_inp, model, vocab_mapper) -> List[str]:
    # TODO: predict the response
    inverse_vocab_mapper = {idx: word for word, idx in vocab_mapper.items()}
    response = model.predict(enc_inp, dec_inp, stop_id=vocab_mapper['//END//'], max_len=15) # A list of token ids
    response = [inverse_vocab_mapper[idx] for idx in response]
    
    return response

def post_process(response) -> str:
    # TODO: post process the output
    return ' '.join(response)