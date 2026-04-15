from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate


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


prompt1 = template.invoke({"topic": "Artificial Intelligence"})

response1 = model.invoke(prompt1)

prompt2 = template2.invoke({"text": response1.content})

response2 = model.invoke(prompt2)

print("Summary:", response2.content)
