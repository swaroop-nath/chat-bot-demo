import streamlit as webapp
from pre_processing import pre_process_input
from generate_response import generate_response
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

enc_inp, dec_inp, vocab_mapper = pre_process_input(user_input, model_invoked)
response, masked_attn_weights, cross_attn_weights = generate_response(enc_inp, dec_inp, model_invoked, vocab_mapper)

write_response(response)

if cross_attn_weights is not None:
    cross_attn_chart = plot_attn_heatmap(cross_attn_weights, type='cross_attn', **{'prompt': user_input, 'response': response})
    webapp.plotly_chart(cross_attn_chart)