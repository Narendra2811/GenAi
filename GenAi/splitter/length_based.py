from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("../Loader/sample.pdf")
docs = loader.load()

splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=15, separator=" ")

# we can use spli_text if we have text instead of document
result = splitter.split_documents(docs)

print(docs[2])

print("------------------------")

print(result[4])
