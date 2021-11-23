from nltk.tokenize import word_tokenize
import json
from typing import List, Dict, Tuple
import re

START_TOKEN = '//START//'
END_TOKEN = '//END//'

word_polymorphs = {'wanna': 'want to', 'favourite': 'favorite', 'colour': 'color', 'flavour': 'flavor', 'humour': 'humor',
                  'labour': 'labor', 'neighbour': 'neighbor', 'apologise': 'apologize', 'organise': 'organize', 'recognise': 'recognize',
                  'analyse': 'analyze', 'paralyse': 'paralyze', 'gonna': 'going to', 'gotta': 'got to',
                  'imma': 'i am going to', 'hii': 'hi', 'hiii': 'hi', 'heyy': 'hey', 'heyyy': 'hey', 'yess': 'yes',
                  'yesss': 'yes', 'nooo': 'no', 'noo': 'no', 'noooo': 'no', 'youuuuu': 'you', 'youuuu': 'you',
                  'youuu': 'you', 'youu': 'you', 'tooo': 'too', 'toooo': 'too'}
smile_polymorphs = {'ha-ha': 'haha'}

for multiplier in range(3, 10):
    smile_polymorphs['ha'*multiplier] = 'haha'

words_to_remove = [('<num>', ''), ('/[\w]+', ''), ('\[\w]+', ''), ('~', ''), ('[[\w]+]', ''), ('[^\w.!?\-:,;<>()\"\'\s]+', ''), ('\s{2,}', ' ')]
contractions = {"'cause": "because", "it's": "it is", "'em'": "them", "i'll": "i will", "im": "i am", "he's": "he is", "\'ll": " will", "\'ve": " have", "\'re": " are", 'whazzup': 'what\'s up',
               'wassup': 'what\'s up'}
ext_contractions = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is", "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are", "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have", "you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have"}
contractions.update(ext_contractions)

def pre_process_input(input_sent, model_invoked) -> Tuple[List[int], List[int]]:
    vocab_mapper = load_vocab(model_invoked)
    should_lower = model_invoked != 'transformer-conv-ai'
    enc_inp = tokenize_sent_and_map(input_sent, vocab_mapper, should_lower=should_lower)
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

    input_sent = drop_unnecessary_items(input_sent, words_to_remove)
    input_sent = unroll_contractions(input_sent, contractions)
    input_sent = substitute_polymorphs(input_sent, word_polymorphs)
    input_sent = substitute_polymorphs(input_sent, smile_polymorphs)

    input_sent = [vocab_mapper[word] for word in word_tokenize(input_sent)]
    return input_sent

def drop_unnecessary_items(sent, words_to_remove):
    for word_to_remove in words_to_remove:
        sent = re.sub(word_to_remove[0], word_to_remove[1], sent)
        
    return sent

def unroll_contractions(sent, contractions):
    for candidate, substitute in contractions.items():
        sent = re.sub(candidate, substitute, sent)
        
    return sent

def substitute_polymorphs(sent, polymorph_mapper):
    for _ in range(3):
        for candidate, substitute in polymorph_mapper.items():
            sent = re.sub(candidate, substitute, sent)
        
    return sent

def get_decoder_input(vocab_mapper) -> List[int]:
    dec_prompt = [vocab_mapper['//START//']]
    return dec_prompt