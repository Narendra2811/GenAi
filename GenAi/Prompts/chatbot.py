from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
import streamlit as st
from langchain.messages import HumanMessage, AIMessage, SystemMessage

from langchain_core.prompts import load_prompt

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",  # Set the desired task
)

model = ChatHuggingFace(llm=llm)
chat_history = [
    SystemMessage(content="You are a helpful assistant "),
]

while True:
    
    user_input = input("you:")
    if user_input.lower() in ["exit", "quit"]:
        print("Exiting the chat.")
        break
    
    chat_history.append(HumanMessage(content=user_input))

    response = model.invoke(chat_history)
    chat_history.append(AIMessage(content=response.content))
    print("bot:", response.content)

print("Chat history:", chat_history)