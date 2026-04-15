from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

embedding = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=1024)

result = embedding.embed_query("what is capital of INDIA?")

print(str(result))
