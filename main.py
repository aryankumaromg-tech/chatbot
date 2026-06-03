from langchain.chat_models import init_chat_model
from langchain_groq import ChatGroq


from langchain_core.messages import AIMessage,SystemMessage,HumanMessage

from dotenv import load_dotenv
load_dotenv()
llm =ChatGroq(model="lama-3.3-70b-versatile",temperature=0.7)
history =[
    SystemMessage(content="you are a helpful AI and you always cry")
]


print("Welcome type'quit' to exit")
while True:
    
    que=input("you :-").strip()
    if que =="quit":
        break
history.append(HumanMessage(que))
response =llm.invoke(history)
history.append(AIMessage(response.content))

print(f"Bot-{response.content}")
