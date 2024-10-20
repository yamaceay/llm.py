### SECTION 1: AUTHENTICATE ###
import os
import openai
import datetime
from dotenv import load_dotenv, find_dotenv
from util import *
import langchain
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.chains import ConversationChain, LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable
from operator import itemgetter
from langchain.chains.router import MultiPromptChain
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.prompts import PromptTemplate
from langchain.document_loaders import CSVLoader
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.indexes import VectorstoreIndexCreator
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.evaluation.qa import QAGenerateChain, QAEvalChain

_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']

### SECTION 2: SET GPT VERSION ###

# Note: LLM's do not always produce the same results. When executing the code 
# in your notebook, you may get slightly different answers.

# account for deprecation of LLM model
# Get the current date
current_date = datetime.datetime.now().date()

# Define the date after which the model should be set to "gpt-3.5-turbo"
target_date = datetime.date(2024, 6, 12)

# Set the model variable based on the current date
if current_date > target_date:
    llm_model = "gpt-3.5-turbo"
else:
    llm_model = "gpt-3.5-turbo-0301"

def complete(prompt: str, model: str = llm_model):
    messages = [{"role": "user", "content": prompt}]
    response = openai.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0, 
    )
    message = response.choices[0].message
    return getattr(message, "content")

def new_template(template_string: str):
    prompt_template = ChatPromptTemplate.from_template(template_string)
    return prompt_template

def new_prompt(prompt_template: ChatPromptTemplate, *args, **kwargs):
    messages = prompt_template.format_messages(*args, **kwargs)
    return messages

def new_schema(name: str, description: str, type: str):
    schema = ResponseSchema(name=name, description=description, type=type)
    return schema

def new_output_parser(*schema_args):
    schemas = []
    for schema_arg in schema_args:
        schema = new_schema(**schema_arg)
        schemas += [schema]
    output_parser = StructuredOutputParser.from_response_schemas(schemas)
    return output_parser

def get_parser_prompt(output_parser: StructuredOutputParser):
    return output_parser.get_format_instructions()

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

def new_runnable(prompt, llm, output_key=None):
    chain: Runnable = prompt | llm
    if output_key is not None:
        chain |= {output_key: StrOutputParser()}
    return chain

def invoke(runnable, **kwargs):
    return runnable.invoke(kwargs)

def invoke_sequence(chains, **inputs):
    results = {}
    for chain in chains:
        input_vars = list(chain.input_schema.schema()["properties"].keys())
        required_inputs = {var: inputs[var] if var in inputs else results[var] for var in input_vars}
        output_vars = list(chain.output_schema.schema()["properties"].keys())
        assert len(output_vars) == 1, "Only one output variable is expected."
        results[output_vars[0]] = invoke(chain | itemgetter(output_vars[0]), **required_inputs)
    return results

def new_route(llm, prompt_infos, verbose=False):
    destination_chains = {}
    for p_info in prompt_infos:
        name = p_info["name"]
        prompt_template = p_info["prompt_template"]
        prompt = ChatPromptTemplate.from_template(template=prompt_template)
        chain = LLMChain(llm=llm, prompt=prompt)
        destination_chains[name] = chain
        
    destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
    destinations_str = "\n".join(destinations)

    default_prompt = ChatPromptTemplate.from_template("{input}")
    default_chain = LLMChain(llm=llm, prompt=default_prompt)

    router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(
        destinations=destinations_str
    )
    router_prompt = PromptTemplate(
        template=router_template,
        input_variables=["input"],
        output_parser=RouterOutputParser(),
    )

    router_chain = LLMRouterChain.from_llm(llm, router_prompt)

    chain = MultiPromptChain(
        router_chain=router_chain, 
        destination_chains=destination_chains, 
        default_chain=default_chain, 
        verbose=verbose,
    )

    return chain

MULTI_PROMPT_ROUTER_TEMPLATE = """Given a raw text input to a \
language model select the model prompt best suited for the input. \
You will be given the names of the available prompts and a \
description of what the prompt is best suited for. \
You may also revise the original input if you think that revising \
it will ultimately lead to a better response from the language model.

<< FORMATTING >>
Return a markdown code snippet with a JSON object formatted to look like:
```json
{{{{
    "destination": string \ name of the prompt to use or "DEFAULT"
    "next_inputs": string \ a potentially modified version of the original input
}}}}
```

REMEMBER: "destination" MUST be one of the candidate prompt \
names specified below OR it can be "DEFAULT" if the input is not \
well suited for any of the candidate prompts.
REMEMBER: "next_inputs" can just be the original input \
if you don't think any modifications are needed.

<< CANDIDATE PROMPTS >>
{destinations}

<< INPUT >>
{{input}}

<< OUTPUT (remember to include the ```json)>>"""

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