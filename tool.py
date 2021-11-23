import streamlit as webapp
from pre_processing import pre_process_input
from generate_response import generate_response
from numpy.random import randint

print('Start')

def get_text():
    input_text = webapp.text_input("You: ","Hi there")
    return input_text 

def get_model():
    option = webapp.selectbox('Which model?', ('transfomer-all-data', 'transformer-conv-ai', 'gru-pytorch', 'gpt-2', 'model-dummy'))
    return option

def write_response(response):
    webapp.header('Response')
    webapp.text(response)

webapp.sidebar.title("NLP Bot")
webapp.title("""
NLP Bot  
NLP Bot is an NLP conversational chatterbot. 
""")

user_input = get_text()
model_invoked = get_model()

enc_inp, dec_inp, vocab_mapper = pre_process_input(user_input, model_invoked)
response = generate_response(enc_inp, dec_inp, model_invoked, vocab_mapper)

# dummy_response = ['Hello there', 'demo response', 'why this response?', 'hey back'][randint(0, 4)]  

write_response(response)

print('End')