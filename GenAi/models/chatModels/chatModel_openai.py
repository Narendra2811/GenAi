from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os


load_dotenv()

llm = ChatOpenAI(
    api_key=os.getenv("api_key"),
    base_url=os.getenv("base_url"),
    model="meta-llama/llama-3-8b-instruct",
)

response = llm.invoke("Explain machine learning in simple words")

print(response.content)
