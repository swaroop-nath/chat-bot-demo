import streamlit as webapp

def get_text():
    input_text = webapp.text_input("You: ","Hi there")
    return input_text 

webapp.sidebar.title("NLP Bot")
webapp.title("""
NLP Bot  
NLP Bot is an NLP conversational chatterbot. 
""")

user_input = get_text()

print(user_input)