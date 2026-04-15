from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_chroma import Chroma

# Create LangChain documents for IPL players

doc1 = Document(
    page_content="Virat Kohli is one of the most successful and consistent batsmen in IPL history. Known for his aggressive batting style and fitness, he has led the Royal Challengers Bangalore in multiple seasons.",
    metadata={"team": "Royal Challengers Bangalore"},
)
doc2 = Document(
    page_content="Rohit Sharma is the most successful captain in IPL history, leading Mumbai Indians to five titles. He's known for his calm demeanor and ability to play big innings under pressure.",
    metadata={"team": "Mumbai Indians"},
)
doc3 = Document(
    page_content="MS Dhoni, famously known as Captain Cool, has led Chennai Super Kings to multiple IPL titles. His finishing skills, wicketkeeping, and leadership are legendary.",
    metadata={"team": "Chennai Super Kings"},
)
doc4 = Document(
    page_content="Jasprit Bumrah is considered one of the best fast bowlers in T20 cricket. Playing for Mumbai Indians, he is known for his yorkers and death-over expertise.",
    metadata={"team": "Mumbai Indians"},
)
doc5 = Document(
    page_content="Ravindra Jadeja is a dynamic all-rounder who contributes with both bat and ball. Representing Chennai Super Kings, his quick fielding and match-winning performances make him a key player.",
    metadata={"team": "Chennai Super Kings"},
)
docs = [doc1, doc2, doc3, doc4, doc5]


"""_
    --> Running locally (In-Memory)
         You can get a Chroma server running in memory by simply instantiating a Chroma instance with a collection name and your embeddings provider:

from langchain_chroma import Chroma
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
)


If you don’t need data persistence, this is a great option for experimenting while building your AI application with LangChain.
​
Running locally (with data persistence)

You can provide the persist_directory argument to save your data across multiple runs of your program:
from langchain_chroma import Chroma

vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",
)
"""


vector_store = Chroma(
    collection_name="test_collection",
    embedding_function=HuggingFaceEmbeddings(),
)


ids = vector_store.add_documents(docs)

print("--------------------------------------------")

print(vector_store.get(include=["embeddings", "documents"]))

print("--------------------------------------------")

result = vector_store.similarity_search(query="who among these is bowler?", k=2)

# vector_store.similarity_search_with_score(query="who among these is bowler?", k=2)

print(result)

vector_store.similarity_search_with_score(
    query="", filter={"team": "Chennai Super Kings"}
)


updated_doc1 = Document(
    page_content="Virat Kohli no The virat kohli is one of the most successful and consistent batsmen in IPL history. Known for his aggressive batting style and fitness, he has led the Royal Challengers Bangalore in multiple seasons.",
    metadata={"team": "Royal Challengers Bangalore"},
)

vector_store.update_document(document_id=ids[0], document=updated_doc1)

print("==================")
print(vector_store.get(include=["documents"]))
