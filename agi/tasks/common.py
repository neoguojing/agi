from agi.llms.model_factory import ModelFactory
from agi.tasks.prompt import english_traslate_template
from langchain_core.output_parsers import StrOutputParser

def create_translate_chain(llm):
    return english_traslate_template | llm | StrOutputParser()


