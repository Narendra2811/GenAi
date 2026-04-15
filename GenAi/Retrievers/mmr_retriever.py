from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document


# Sample documents
docs = [
    Document(page_content="LangChain makes it easy to work with LLMs."),
    Document(page_content="LangChain is used to build LLM based applications."),
    Document(page_content="Chroma is used to store and search document embeddings."),
    Document(page_content="Embeddings are vector representations of text."),
    Document(
        page_content="MMR helps you get diverse results when doing similarity search."
    ),
    Document(page_content="LangChain supports Chroma, FAISS, Pinecone, and more."),
]

embedding = HuggingFaceEmbeddings()

vectorstore = FAISS.from_documents(documents=docs, embedding=embedding)

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 2,
        "lambda_mult": 0.5,  # jetlu ochu atlu vadhare diverse result madse...if 1 rakhisu to same as normal retriever ni jem work karse
    },
)

query = "What is langchain?"
results = retriever.invoke(query)


for i, doc in enumerate(results):
    print(f"\n--- Result {i+1} ---")
    print(doc.page_content)
