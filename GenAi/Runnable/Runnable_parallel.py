from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableParallel, RunnableSequence

load_dotenv()

prompt1 = PromptTemplate(
    template="write a good thing  about thi person  {person}",
    input_variables=["person"],
)

prompt2 = PromptTemplate(
    template="write  a bad thing about this person {person}", input_variables=["person"]
)


llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",  # Set the desired task)
)
model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

chain = RunnableParallel(
    {
        "GOOD": RunnableSequence(prompt1, model, parser),
        "BAD": RunnableSequence(prompt2, model, parser),
    }
)
result = chain.invoke({"person": "narendra modi"})

print(chain.get_graph().print_ascii())

print(result["BAD"])
