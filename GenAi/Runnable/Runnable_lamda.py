from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import (
    RunnableSequence,
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda,
)


load_dotenv()


# landa function
def words_count(text: str) -> int:
    return len(text.split())


prompt1 = PromptTemplate(template="write a joke about {joke}", input_variables=["joke"])


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
        "word_count": RunnableLambda(words_count),
    }
)

final_chain = RunnableSequence(joke_chain, parallel_chain)

result = final_chain.invoke({"joke": "narendra modi"})

print(result["word_count"])
