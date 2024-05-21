from util import *

# Chat Completion
from langchain_openai import ChatOpenAI

# Prompt Template
from langchain.prompts import ChatPromptTemplate

# Response Schema and Output Parser
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser

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