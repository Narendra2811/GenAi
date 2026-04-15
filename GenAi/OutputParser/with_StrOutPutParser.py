from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",  # Set the desired task
)

model = ChatHuggingFace(llm=llm)

# create detail explanation
template = PromptTemplate(
    template="explained detail in this topic {topic}?", input_variables=["topic"]
)

template2 = PromptTemplate(
    template="create 2 line summary of thi text : {text}", input_variables=["text"]
)

parser = StrOutputParser()


# difference between  with an without StrOUTPUTPARSER
chain = template | model | parser | template2 | model | parser

response = chain.invoke({"topic": "Artificial Intelligence"})

print("Summary:", response)