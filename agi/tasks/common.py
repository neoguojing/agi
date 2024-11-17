from agi.llms.model_factory import ModelFactory
from agi.tasks.prompt import english_traslate_template
from langchain_core.output_parsers import StrOutputParser
from agi.llms.text2image import Text2Image
from agi.llms.image2image import Image2Image


def create_translate_chain(llm):
    return english_traslate_template | llm | StrOutputParser()


def create_text2image_chain(llm):
    translate = create_translate_chain(llm)
    text2image = Text2Image()
    
    return translate | text2image

def create_image2image_chain(llm):
    translate = create_translate_chain(llm)
    image2image = Image2Image()
    
    return translate | image2image