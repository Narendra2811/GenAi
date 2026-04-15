from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import (
    RunnableSequence,
    RunnableParallel,
    RunnablePassthrough,
)


load_dotenv()

prompt1 = PromptTemplate(template="write a joke about {joke}", input_variables=["joke"])

promtp2 = PromptTemplate(
    template="write a explantion of this joke : {joke} ", input_variables=["joke"]
)


llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",  # Set the desired task)
)
model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

joke_chain = RunnableSequence(prompt1, model, parser)

parallel_chain = RunnableParallel(
    {
        "JOKE": RunnablePassthrough(),
        "EXPLANATION": RunnableSequence(promtp2, model, parser),
    }
)

final_chain = RunnableSequence(joke_chain, parallel_chain)

result = final_chain.invoke({"joke": "narendra modi"})

print(result["EXPLANATION"])
