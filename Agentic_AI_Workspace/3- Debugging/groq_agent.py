from typing import Annotated
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage
import os
from dotenv import load_dotenv

load_dotenv()

os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
os.environ['LANGSMITH_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')

class State(BaseModel):
    messages: Annotated[list[BaseMessage], add_messages]

model = ChatGroq(model='llama-3.1-8b-instant', temperature=0)

def make_default_graph():
    graph_workflow = StateGraph(State)

    def call_model(state: State):
        return {"messages": [model.invoke(state.messages)]}
    
    graph_workflow.add_node('agent', call_model)
    graph_workflow.add_edge('START', 'agent')
    graph_workflow.add_edge('agent', END)

    agent = graph_workflow.compile()

    return agent

agent = make_default_graph() 