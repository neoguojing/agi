from agi.llms.model_factory import ModelFactory
from agi.tasks.prompt import english_traslate_template,multimodal_input_template
from langchain_core.output_parsers import StrOutputParser
from agi.llms.text2image import Text2Image
from agi.llms.image2image import Image2Image
from langchain_core.runnables import RunnablePassthrough,RunnableLambda
import json
from langchain_core.messages import HumanMessage

def create_translate_chain(llm):
    return english_traslate_template | llm | StrOutputParser()


def create_text2image_chain(llm):
    translate = create_translate_chain(llm)
    text2image = Text2Image()
    
    return translate | text2image

def create_image2image_chain(llm):
    translate = create_translate_chain(llm)
    image2image = Image2Image()
    
    def parse_input(input:str):
        data = json.loads(input)
        return data
        
    def build_messages(input :dict):
        return HumanMessage(content=[
            {"type": "text", "text": input.get("text")},
            {"type": "media", "media": input.get("media")},
        ])
        
    chain = (multimodal_input_template
        | RunnableLambda(parse_input)
        | RunnablePassthrough.assign(text=translate.with_config(run_name="translate"))
        | RunnableLambda(build_messages)
        | image2image
    )
    
    return chain
