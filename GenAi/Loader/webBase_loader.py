from langchain_community.document_loaders import WebBaseLoader

url = "https://www.website.com/?source=SC&country=IN"

loader = WebBaseLoader(url)
docs = loader.lazy_load()

for document in docs:
    print(document)
