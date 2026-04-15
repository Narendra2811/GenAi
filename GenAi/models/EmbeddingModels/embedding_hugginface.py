from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = HuggingFaceEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2"
)

documents = [
    "India is a county of India",
    "kolkata is a city of India",
    "paris is the capital of france",
]


result = embedding.embed_documents(documents)

print(str(result))
