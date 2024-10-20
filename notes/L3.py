from util import *

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable
from langchain.chains import LLMChain
from operator import itemgetter
from langchain.chains.router import MultiPromptChain
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.prompts import PromptTemplate

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

llm = ChatOpenAI(temperature=0.0, model=llm_model)