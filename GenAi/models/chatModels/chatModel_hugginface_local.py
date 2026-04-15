from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from dotenv import load_dotenv
import os

load_dotenv()

llm = HuggingFacePipeline.from_model_id(
    model_id="distilgpt2",
    task="text-generation",
    pipeline_kwargs={"max_new_tokens": 100, "temperature": 0.1},
)

# model = ChatHuggingFace(llm=llm)..because this model distilgpt2 is not a chat model
result = llm.invoke("what is national languange of india?")

print(result)
