from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os

load_dotenv()

print(os.getenv("HUGGINGFACEHUB_API_TOKEN"))

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",  # Set the desired task
)

model = ChatHuggingFace(llm=llm)
result = model.invoke("what is your task")

print(result.content)
