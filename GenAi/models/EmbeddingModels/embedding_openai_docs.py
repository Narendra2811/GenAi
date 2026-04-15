from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=1024)

documents = [
    "India is a county of India",
    "kolkata is a city of India",
    "paris is the capital of france",
]
result = embedding.embed_documents(documents)
# gives 2D list ..
print(str(result))
