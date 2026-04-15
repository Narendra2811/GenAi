from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")

result = llm.invoke("what is you primary work")

print(result.content)
