from util import *

from langchain.agents import tool, load_tools, initialize_agent
from langchain.agents import AgentType
from langchain_openai import ChatOpenAI
from datetime import date

llm = ChatOpenAI(temperature=0, model=llm_model)

def new_agent(tools, llm, verbose=False):
    return initialize_agent(
        tools, 
        llm, 
        agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True,
        verbose = verbose)
