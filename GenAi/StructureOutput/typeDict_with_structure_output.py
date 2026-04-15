from typing import TypedDict, Annotated, Optional, Literal
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

from dotenv import load_dotenv

load_dotenv()


class Template(TypedDict):
    topic: Annotated[str, "The product or service being reviewed"]
    pros: Annotated[str, "The advantages or positive aspects of the product or service"]
    cons: Annotated[
        str, "The disadvantages or negative aspects of the product or service"
    ]
    rating: Annotated[
        float, "The rating of the product or service on a scale of 1 to 5"
    ]
    summary: Annotated[str, "A brief summary of the review"]
    sentiment: Annotated[
        Literal["pos", "neg"],
        "The overall sentiment of the review, either 'pos' for positive or 'neg' for negative",
    ]


llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",  # Set the desired task
)


model = ChatHuggingFace(llm=llm)


structure_model = model.with_structured_output(Template)

response = structure_model.invoke(
    """The Apple AirPods Pro (2nd gen) are premium wireless earbuds designed mainly for Apple users. They offer excellent sound quality, strong noise cancellation, and seamless integration with iPhones and other Apple devices.

👍 Pros



👎 Cons

💰 Expensive compared to many earbuds.

🤖 Some features work best only with Apple devices.

🎧 Sound quality is good but not the absolute best for audiophiles.

📊 Key Features

Active Noise Cancellation

Transparency Mode (hear surroundings)

Spatial Audio

Bluetooth wireless connectivity

MagSafe / USB-C charging case

⭐ Rating

 3.5/ 5"""
)


print(response["rating"])
print(response["sentiment"])
