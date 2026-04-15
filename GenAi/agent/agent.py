from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from dotenv import load_dotenv
import requests
from langchain_classic import hub
from langchain_classic.agents import create_react_agent, AgentExecutor


load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",  # Set the desired task
)

model = ChatHuggingFace(llm=llm)


@tool
def get_weather_data(city: str) -> str:
    """
    This function fetches the current weather data for a given city
    """
    url = f"https://api.weatherstack.com/current?access_key=e1265d86bff0bc8d139775bca75b034a&query={city}"

    response = requests.get(url)

    return response.json()


# second tool
search_tool = DuckDuckGoSearchRun()

prompt = hub.pull("hwchase17/react")  # pulls the standard ReAct agent prompt

agent = create_react_agent(
    llm=model, tools=[search_tool, get_weather_data], prompt=prompt
)

agent_executor = AgentExecutor(
    agent=agent, tools=[search_tool, get_weather_data], verbose=True
)

response = agent_executor.invoke(
    {
        "input": "Find the capital of Madhya Pradesh, then find it's current weather condition"
    }
)
print(response)
