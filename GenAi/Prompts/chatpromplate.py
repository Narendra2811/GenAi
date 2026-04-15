from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
import streamlit as st
from langchain.messages import HumanMessage, AIMessage, SystemMessage

from langchain_core.prompts import load_prompt
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()

chat_history = []

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",  # Set the desired task
)

model = ChatHuggingFace(llm=llm)

template = ChatPromptTemplate(
    [
        ("system", "you are a helpful assistant that have experinec in {domain}"),
        ("human", "explain {topic} to me in a {style} way"),
    ]
)

# prompt = template.invoke(
#     {
#         "domain": "machine learning",
#         "topic": "transformer architecture",
#         "style": "beginner-friendly",
#     }
# )


style = "beginner-friendly"
domain = "global markte"


while True:

    user_input = input("you:")
    chat_history.append(("human", user_input))
    if user_input.lower() in ["exit", "quit"]:
        print("Exiting the chat.")
        break

    chain = template | model

    response = chain.invoke(
        {
            "domain": domain,
            "topic": user_input,
            "style": style,
        }
    )
    chat_history.append(("ai", response.content))
    print("bot:", response.content)


print(chat_history)
