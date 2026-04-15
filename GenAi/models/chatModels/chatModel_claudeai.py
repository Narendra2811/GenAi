from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatAnthropic(
    model="claude-3-haiku-20240307", api_key=os.getenv("ANTHROPIC_API_KEY")
)

response = llm.invoke("What is the capital of India?")

print(response.content)
