from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser


load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",  # Set the desired task
)

model = ChatHuggingFace(llm=llm)

parser = JsonOutputParser()

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
