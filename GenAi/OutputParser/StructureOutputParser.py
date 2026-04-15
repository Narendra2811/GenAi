from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema


load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",  # Set the desired task
)

model = ChatHuggingFace(llm=llm)

ResponseSchema1 = ResponseSchema(
    name="definition", description="Definition of the topic"
)
ResponseSchema2 = ResponseSchema(
    name="applications", description="Applications of the topic"
)
parser = StructureOutputParser(response_schemas=[ResponseSchema1, ResponseSchema2])

response_schema = [
    ResponseSchema(name="definition", description="Defintion of the topic"),
    ResponseSchema(name="applications", description="Applications of the topic"),
]


parser = StructureOutputParser.from_response_schemas(response_schema)

# create detail explanation
template = PromptTemplate(
    template="""Give information about the topic.
       Topic: {topic}
       Return the answer strictly in JSON format  {format_instructions}""",
    input_variables=["topic"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)


prompt = template.invoke({"topic": "Artificial Intelligence"})

response = model.invoke(prompt)

parsed_response = parser.parse(response.content)
print("Response:", parsed_response)
