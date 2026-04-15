from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

template = ChatPromptTemplate(
    [
        ("system", "you are a helpful assistant "),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{query}"),
    ]
)

history = []

with open("Prompts/chat_history.txt") as f:
    history.extend(f.readlines())

prompt = template.invoke({"history": history, "query": "my order status"})


print(prompt)
