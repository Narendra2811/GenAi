from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("../Loader/sample.pdf")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=55,
    separators=[
        "\n\n",
        "\n",
        ". ",
        "? ",
        "! ",
        ", ",
        " ",
        "",
    ],
)
# here we have Document so we use split_documents
# if we have text then use create_document


result = splitter.split_documents(docs)

for chunk in result:
    print(chunk)
    print("------")
