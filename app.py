import os
import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain, SequentialChain
from dotenv import load_dotenv
load_dotenv()

apiKey = os.getenv("OPENAI_API_KEY")

#app framework ----- python -m streamlit run app.py
st.title('Youtube GPT Creator')
prompt = st.text_input("plug in your prompt here")

#llms
llm = OpenAI(temperature=0.9)

#prompt template
titleTemplate = PromptTemplate(
    input_variables=['topic'],
    template='write me a youtube video title about {topic}'
)

scriptTemplate = PromptTemplate(
    input_variables=['title'],
    template='write me a youtube video script based on this title: {title}'
)

titleChain = LLMChain(llm=llm, prompt=titleTemplate, verbose=True, output_key='title')
scriptChain = LLMChain(llm=llm, prompt=scriptTemplate, verbose=True, output_key='script')
sequentialChain = SequentialChain(chains=[titleChain, scriptChain], input_variables=['topic'], output_variables=['title', 'script'], verbose=True)

#show stuff to screen
if prompt:
    response = sequentialChain({'topic':prompt})
    st.write(response['title'])
    st.write(response['script'])
