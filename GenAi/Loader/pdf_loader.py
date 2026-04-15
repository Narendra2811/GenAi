from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("sample.pdf")

docs = loader.load()

# print(docs)
# print(len(docs))  # as many pages in pdf file


print(docs[0].page_content)


# lazy loader (use this when big and many files)
# docs = loader.lazy_load()

# for document in docs:
#     print(document)
