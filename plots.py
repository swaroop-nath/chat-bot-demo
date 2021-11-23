import plotly.figure_factory as ff
import plotly.graph_objects as go
from nltk.tokenize import word_tokenize

def plot_attn_heatmap(attn_weights, type='masked_attn', **kwargs):
    if type == 'masked_attn':
        # attn_weights.shape = [dec_inp_len, dec_inp_len]
        sent = get_words(kwargs['sent'])
        fig = px.imshow(attn_weights, labels=dict(x='Dec Input', y='Dec Input'), x=sent, y=sent)
        return fig
    elif type == 'cross_attn':
        # attn_weights.shape = [enc_inp, dec_output]
        enc_inp_words = get_words(kwargs['prompt'], is_response=False)
        dec_output_words = get_words(kwargs['response'])
        fig = ff.create_annotated_heatmap(attn_weights, colorscale='Viridis', x=enc_inp_words, y=dec_output_words)
        return fig

def get_words(sent, is_response=True):
    if is_response: return word_tokenize(sent) + ['//END//']
    return word_tokenize(sent)