from nltk.tokenize import word_tokenize
import json
from typing import List, Dict

START_TOKEN = '//START//'
END_TOKEN = '//END//'

def pre_process_input(input_sent, model_invoked) -> List[List[int]]:
    vocab_mapper = load_vocab(model_invoked)
    enc_inp = tokenize_sent_and_map(input_sent, vocab_mapper)
    dec_inp = get_decoder_input(vocab_mapper)

    return enc_inp, dec_inp

def load_vocab(model_invoked) -> Dict[str: int]:
    # TODO: load vocab, as per the model invoked - keep a cache so as to make the app faster
    pass

def tokenize_sent_and_map(input_sent, vocab_mapper) -> List[int]:
    # TODO: lower-case the sent, tokenize into words and map to token ids
    # Return type - a 1D list of token ids
    pass

def get_decoder_input(vocab_mapper) -> List[int]:
    dec_prompt = [vocab_mapper['//END//']]
    return dec_prompt