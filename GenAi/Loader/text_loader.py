from langchain_community.document_loaders import TextLoader
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct", task="text-generation"
)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

prompt = PromptTemplate(
    template="create bullet point summay of this text {text}", input_variables=["text"]
)

loader = TextLoader("test.txt")

docs = loader.load()

# print(type(docs))
# print(len(docs))
# print(type(docs[0]))
# print(docs)


chain = prompt | model | parser

result = chain.invoke({"text": docs[0].page_content})

print(result)
