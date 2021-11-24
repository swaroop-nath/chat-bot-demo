import streamlit as webapp
from pre_processing import pre_process_input
from generate_response import generate_response, generate_response_gpt2
from plots import plot_attn_heatmap

def get_text():
    input_text = webapp.text_input("You: ","Hi there")
    return input_text 

def get_model():
    option = webapp.selectbox('Which model?', ('transformer-conv-ai', 'transformer-all-data', 'gru-pytorch', 'gpt-2'))
    return option

def write_response(response):
    webapp.header('Response')
    webapp.text(response)

webapp.sidebar.title("CS626 Bot")
webapp.title("""
CS626 Bot 
It is a simple conversational bot
""")

user_input = get_text()
model_invoked = get_model()
should_generate = webapp.button('Generate Response')
temp = webapp.slider('Temperature for sampling: ', 0.01, 1.25, 0.5)
use_greedy = webapp.checkbox('Use Greedy Decoding')

if should_generate:
    if model_invoked == 'gpt-2':
        response = generate_response_gpt2(user_input, temp)
        cross_attn_weights = None
    else:
        enc_inp, dec_inp, vocab_mapper = pre_process_input(user_input, model_invoked)
        response, masked_attn_weights, cross_attn_weights = generate_response(enc_inp, dec_inp, model_invoked, vocab_mapper, temp, use_greedy)

    write_response(response)

    if cross_attn_weights is not None:
        cross_attn_chart = plot_attn_heatmap(cross_attn_weights, type='cross_attn', **{'prompt': user_input, 'response': response})
        webapp.plotly_chart(cross_attn_chart)