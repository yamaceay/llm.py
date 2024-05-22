from util import *

from langchain_openai import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.indexes import VectorstoreIndexCreator
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()

def embed(text):
    return embeddings.embed_query(text)

def new_vector_db(file):
    loader = CSVLoader(file_path=file)
    index = VectorstoreIndexCreator(
        vectorstore_cls=DocArrayInMemorySearch,
        embedding=embeddings,
    ).from_loaders([loader])
    return index

llm = ChatOpenAI(temperature=0.0, model=llm_model)