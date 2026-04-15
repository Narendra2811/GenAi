from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

embedding = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")

documents = [
    "India is a county of India",
    "paris is the capital of france",
    "kolkata is a city of India",
]

doc = embedding.embed_documents(documents)
query = embedding.embed_query("what is capital of INDIA?")

result = cosine_similarity([query], doc)[0]
print(result)

index, score = list(sorted(enumerate(result), key=lambda x: x[1], reverse=True))[0]

print(f"Most similar document is: {documents[index]} with score: {score}")
