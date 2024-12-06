import os
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain.schema import AIMessage, HumanMessage, SystemMessage
import gradio as gr
import numpy as np
from dotenv import load_dotenv

load_dotenv()


choices = ['OPENAI ChatGPT', 'Google GenAI', 'Anthropic Claude']

def get_openai_llm() -> ChatOpenAI:
    #Initializing chat models
    print("Using OPENAI llm")
    return ChatOpenAI(temperature=0.7, model="gpt-4o-mini", streaming=True)

def get_google_llm() -> ChatGoogleGenerativeAI:
    #Initializing chat models
    print("Using Google GEN-AI llm")
    return ChatGoogleGenerativeAI(model="gemini-1.5-flash-002", streaming=True)

def get_anthropic_llm()-> ChatAnthropic:
    #Initializing chat models
    print("Using ANTHROPIC llm")
    return ChatAnthropic(model="claude-3-haiku-20240307", streaming=True)

def get_llm(llm_choice: str) -> [ChatOpenAI | ChatGoogleGenerativeAI | ChatAnthropic]:
    llm_retrievers = [get_openai_llm, get_google_llm, get_anthropic_llm]
    llm = llm_retrievers[choices.index(llm_choice)]()
    return llm



system_message = """You are an intelligent chatbot who talks like a pirate in an archaic English.
"""

def stream_response(message, history):
    print("Input:-> {0}. \nHistory: {1}".format(message,history))
    history_langchain_format = []
    #history_langchain_format.append(SystemMessage(content=system_message))

    for human, ai in history:
        history_langchain_format.append(HumanMessage(content=human))
        history_langchain_format.append(AIMessage(content=ai))

    if message is not None:
        history_langchain_format.append(HumanMessage(content=message))
        part_message = ""
        for response in llm.stream(history_langchain_format):
            part_message += response.content
            yield part_message

interface = gr.ChatInterface(stream_response, textbox=gr.Textbox(placeholder="Write something to the LLM...", container=False, scale=10))

if __name__=="__main__":
    interface.launch(debug=True)
        