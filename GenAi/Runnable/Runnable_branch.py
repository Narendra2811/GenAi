from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import (
    RunnableSequence,
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda,
    RunnableBranch,
)


load_dotenv()

prompt1 = PromptTemplate(
    template="write a things for this topis : {topic}", input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="write a summary of this text : {text}", input_variables=["text"]
)

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",  # Set the desired task)
)
model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

chain1 = prompt1 | model | parser

branch_chain = RunnableBranch(
    (lambda x: len(x.split()) > 500, prompt2 | model | parser),
    RunnablePassthrough(),
)

final_chain = RunnableSequence(chain1, branch_chain)

result = final_chain.invoke({"topic": "Narendra Patel"})

print(result)
print(final_chain.get_graph().print_ascii())
