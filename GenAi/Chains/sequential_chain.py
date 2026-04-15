from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")

prompt1 = PromptTemplate(
    template="explain  a bad thing for this country : {country}",
    input_variables=["country"],
)

prompt2 = PromptTemplate(
    template="write good thing for this country {country} from this bad text : {text} ",
    input_variables=["text", "country"],
)

parser = StrOutputParser()

# this type of things we have to do it if in chain there is more than one variable
chain = (
    {
        "text": prompt1 | model | parser,
        "country": lambda x: x['country']
    }
    | prompt2 | model | parser
)

result = chain.invoke({"country": "India"})

print(result)
