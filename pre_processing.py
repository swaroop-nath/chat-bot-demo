from nltk.tokenize import word_tokenize
import json
from typing import List, Dict, Tuple

START_TOKEN = '//START//'
END_TOKEN = '//END//'

def pre_process_input(input_sent, model_invoked) -> Tuple[List[int], List[int]]:
    vocab_mapper = load_vocab(model_invoked)
    enc_inp = tokenize_sent_and_map(input_sent, vocab_mapper)
    dec_inp = get_decoder_input(vocab_mapper)

    return enc_inp, dec_inp, vocab_mapper

def load_vocab(model_invoked) -> Dict:
    # TODO: load vocab, as per the model invoked - keep a cache so as to make the app faster
    vocab_path = './models/vocabs/vocab_{}.json'.format(model_invoked)
    with open(vocab_path, 'r') as file:
        vocab_mapper = json.load(file)
    return vocab_mapper

def tokenize_sent_and_map(input_sent, vocab_mapper, should_lower=True) -> List[int]:
    # TODO: lower-case the sent, tokenize into words and map to token ids, don't lower in case of convai2 no_ctxt model
    # Return type - a 1D list of token ids

    if should_lower: input_sent = input_sent.lower()

    input_sent = [vocab_mapper[word] for word in word_tokenize(input_sent)]
    return input_sent

def get_decoder_input(vocab_mapper) -> List[int]:
    dec_prompt = [vocab_mapper['//END//']]
    return dec_prompt