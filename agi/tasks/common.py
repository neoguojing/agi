from agi.llms.model_factory import ModelFactory
from agi.tasks.prompt import english_traslate_template,multimodal_input_template
from langchain_core.output_parsers import StrOutputParser
from agi.llms.text2image import Text2Image
from agi.llms.image2image import Image2Image
from agi.llms.tts import TextToSpeech
from agi.llms.speech2text import Speech2Text
from langchain_core.runnables import RunnablePassthrough,RunnableLambda,RunnableBranch
import json
from langchain_core.messages import HumanMessage
from langchain_core.prompt_values import StringPromptValue

# TODO
def input_branch():
    return RunnableBranch(
        (),
        
    )
def build_messages(input :dict):
   
    media = None
    media_dict = input.get("media") 
    if media_dict.get('data'):  # 首先检查 'data' 字段
        media = media_dict['data']
    elif media_dict.get('path'):  # 其次检查 'path' 字段
        media =  media_dict['path']
    elif media_dict.get('url'):  # 最后检查 'url' 字段
        media =  media_dict['url'] 
    else:
        return HumanMessage(content=input.get("text"))

    return HumanMessage(content=[
        {"type": "text", "text": input.get("text")},
        {"type": "media", "media": media},
    ])
    
def parse_input(input: StringPromptValue) -> dict:
    try:
        
        print(input.to_json())
        data = json.loads(input.to_string())
        print("**********",data)
        return data
    except json.JSONDecodeError as e:
        print(e,input.to_string())
        return {}

def create_translate_chain(llm):
    return english_traslate_template | llm | StrOutputParser()


def create_text2image_chain(llm):
    translate = create_translate_chain(llm)
    text2image = Text2Image()
    
    return translate | text2image

def create_image_gen_chain(llm):
    translate = create_translate_chain(llm)
    image2image = Image2Image()
    text2image = Text2Image()
    
    chain = (multimodal_input_template
        | RunnableLambda(parse_input)
        | RunnablePassthrough.assign(text=translate.with_config(run_name="translate"))
        | RunnableLambda(build_messages)
        | RunnableBranch((
            (lambda x: isinstance(x.content, str),text2image)
        ),image2image)
    )
    
    return chain

def create_text2speech_chain():
    text2speech = TextToSpeech()
    chain = (multimodal_input_template
        | RunnableLambda(parse_input)
        | RunnableLambda(build_messages)
        | text2speech
    )
    return chain

def create_speech2text_chain():
    speech2text = Speech2Text()
    chain = (multimodal_input_template
        | RunnableLambda(parse_input)
        | RunnableLambda(build_messages)
        | speech2text
    )
    return chain