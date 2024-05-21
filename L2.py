from util import *

# Chat Completion
from langchain_openai import ChatOpenAI

# Conversation Chain
from langchain.chains import ConversationChain

# Conversation Buffers
from langchain.memory import ConversationBufferMemory
from langchain.memory import ConversationBufferWindowMemory
from langchain.memory import ConversationTokenBufferMemory
from langchain.memory import ConversationSummaryBufferMemory

def new_chain(llm: ChatOpenAI, memory: ConversationBufferMemory):
    conversation = ConversationChain(
        llm=llm,
        memory=memory,
    )
    def invoke(a, verbose=False):
        conversation.verbose = verbose
        return conversation.predict(input=a)
    return invoke

def add_buffer(memory, messages: list[str]):
    for i in range(len(messages) // 2):
        memory.save_context(
            {"input": messages[2*i]},
            {"output": messages[2*i+1]},
        )

llm = ChatOpenAI(temperature=0.0, model=llm_model)