from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field


load_dotenv()


class Person(BaseModel):
    name: str = Field(description="The person's name")
    age: int = Field(gt=18, description="The person's age")
    city: str = Field(description="The city where the person lives")


llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",  # Set the desired task
)

model = ChatHuggingFace(llm=llm)

parser = PydanticOutputParser(pydantic_object=Person)


# create detail explanation
template = PromptTemplate(
    template="generate a person's information in json format based on the following topic: {topic}. {format_instructions}",
    partial_variables={"format_instructions": parser.get_format_instructions()},
)


# print("Prompt:", prompt)
# response = model.invoke(prompt)

# parsed_response = parser.parse(response.content)

chain = template | model | parser

parsed_response = chain.invoke(
    {
        "topic": "A software engineer living in San Francisco who is 30 years old and named Alice",
    }
)

print("Response:", parsed_response)
