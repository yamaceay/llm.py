from util import *

import langchain
from langchain_openai import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.indexes import VectorstoreIndexCreator
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.evaluation.qa import QAGenerateChain
from langchain.evaluation.qa import QAEvalChain

class DebugContextManager:
    def __init__(self, module):
        self.module = module

    def __enter__(self):
        self.original_debug_state = self.module.debug
        self.module.debug = True

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.module.debug = self.original_debug_state

embeddings = OpenAIEmbeddings()

def embed(text):
    return embeddings.embed_query(text)

def load_csv(file):
    loader = CSVLoader(file_path=file)
    return loader.load()

def new_vector_db(file):
    loader = CSVLoader(file_path=file)
    index = VectorstoreIndexCreator(
        vectorstore_cls=DocArrayInMemorySearch,
        embedding=embeddings,
    ).from_loaders([loader])
    return index

def new_qa(index, sep, verbose=False):
    qa = RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type="stuff", 
        retriever=index.vectorstore.as_retriever(), 
        verbose=verbose,
        chain_type_kwargs = {
            "document_separator": sep
        }
    )
    return qa

def gen_examples(llm, data):
    example_gen_chain = QAGenerateChain.from_llm(llm)
    new_examples = example_gen_chain.apply_and_parse(
        [{"doc": t} for t in data]
    )
    qa_pairs = [example["qa_pairs"] for example in new_examples]
    return qa_pairs

def evaluate(examples, predictions):
    eval_chain = QAEvalChain.from_llm(llm)
    graded_outputs = eval_chain.evaluate(examples, predictions)
    
    accuracy = 0
    for pred, output in zip(predictions, graded_outputs):
        if output["results"] == "CORRECT":
            accuracy += 1
        else:
            print("Question: " + pred['query'])
            print("Real Answer: " + pred['answer'])
            print("Predicted Answer: " + pred['result'])
            print()
    return accuracy / len(graded_outputs)

llm = ChatOpenAI(temperature=0.0, model=llm_model)