import unittest
from unittest.mock import patch, MagicMock
import threading
from langchain_core.messages import AIMessage, HumanMessage

# Assuming we import the TaskFactory and constants like TASK_LLM, TASK_EMBEDDING, etc.
from agi.tasks.task_factory import TaskFactory, TASK_TRANSLATE, TASK_IMAGE_GEN, TASK_TTS, TASK_SPEECH_TEXT,TASK_LLM_WITH_HISTORY
from agi.tasks.prompt import multimodal_input_template
from agi.tasks.define import State
from agi.tasks.multi_model_app import user_understand

from agi.config import log


class TestTaskMultiModalFactory(unittest.TestCase):
    def test_template(self):
        value = multimodal_input_template.invoke({"text":"123","image":"233","audio":"1223","video":"1111"})
        print(value.to_messages())
        
    def test_translate_chain(self):
        # Test for TASK_LLM
        llm_task = TaskFactory.create_task(TASK_TRANSLATE)
        input = State(
            messages=[HumanMessage(content="我爱北京天安门")],
        )
        resp = llm_task.invoke(input)
        self.assertIsNotNone(resp)
        self.assertIsInstance(resp["messages"][-1],HumanMessage)
        self.assertIsInstance(resp["messages"][-1].content,str)
        
    def test_text2speech_chain(self):
        # Test for TASK_LLM
        llm_task = TaskFactory.create_task(TASK_TTS)
        input = State(
            messages=[HumanMessage(content="These prompt templates are used to format a single string, and generally are used for simpler inputs")],
        )
        resp = llm_task.invoke(input)
        self.assertIsNotNone(resp["messages"][-1],AIMessage)
        self.assertIsInstance(resp["messages"][-1].content,list)
        self.assertIsNotNone(resp["messages"][-1].content[0].get("audio"))
        self.assertEqual(resp["messages"][-1].content[0].get("type"),"audio")
        
    def test_speech2text_chain(self):
        llm_task = TaskFactory.create_task(TASK_SPEECH_TEXT)
        input = State(
            messages=[HumanMessage(content=[{"type":"audio","audio":"tests/zh-cn-sample.wav"}])],
        )
        resp = llm_task.invoke(input)
        self.assertIsNotNone(resp.get("messages"))
        self.assertIsInstance(resp.get("messages")[-1],HumanMessage)
        self.assertIsNotNone(resp.get("messages")[-1].content)
        input["feature"] = "speech"
        resp = llm_task.invoke(input)
        self.assertIsInstance(resp["messages"][-1],AIMessage)
        self.assertIsInstance(resp["messages"][-1].content,str)
        self.assertIsNotNone(resp["messages"][-1].content)
        
    def test_text2image_chain(self):
        llm_task = TaskFactory.create_task(TASK_IMAGE_GEN)
        input = State(
            messages=[HumanMessage(content=[{"type":"text","text":"星辰大海"}])],
        )
        resp = llm_task.invoke(input)
        print(resp)
        self.assertIsInstance(resp["messages"][-1],AIMessage)
        self.assertIsInstance(resp["messages"][-1].content,list)
        self.assertIsNotNone(resp["messages"][-1].content[0].get("image"))
        self.assertEqual(resp["messages"][-1].content[0].get("type"),"image")
        # self.assertIsNotNone(resp.content)
        input = State(
            messages=[HumanMessage(content=[{"type":"text","text":"猫咪在游泳"},{"type":"image","image":"tests/cat.jpg"}])],
        )
        resp = llm_task.invoke(input)
        self.assertIsInstance(resp["messages"][-1],AIMessage)
        self.assertIsInstance(resp["messages"][-1].content,list)
        self.assertIsNotNone(resp["messages"][-1].content[0].get("image"))
        self.assertEqual(resp["messages"][-1].content[0].get("type"),"image")
        
    def test_user_understand(self):
        config={"configurable": {"user_id": "test", "conversation_id": "6"}}
        input = State(
            messages=[HumanMessage(content=[{"type":"text","text":"画一幅水墨画"}])],
        )
        chain = user_understand(TaskFactory.get_llm())
        resp = chain.invoke(input,config=config)
        print(resp)
        self.assertIsInstance(resp["messages"][-1],HumanMessage)
        self.assertIsNotNone(resp["messages"][-1].content[0].get("text"))
        
        input = State(
            messages=[
                HumanMessage(content=[{"type":"text","text":"画一幅水墨画"}]),
                AIMessage(content=[{"type": "image", "image": "http://example.com/aaaa.jpg"}]),
                HumanMessage(content=[{"type": "text", "text": "画的难看死了，重新画"}]),
                ],
        )
        resp = chain.invoke(input,config=config)
        print(resp)
        self.assertIsInstance(resp["messages"][-1],HumanMessage)
        self.assertIsNotNone(resp["messages"][-1].content[0].get("text"))
        
        input = State(
            messages=[
                HumanMessage(content=[{"type":"text","text":"画一幅水墨画"}]),
                AIMessage(content=[{"type": "image", "image": "http://example.com/aaaa.jpg"}]),
                HumanMessage(content=[{"type": "text", "text": "修改以上的画为油画风格"}]),
                ],
        )
        resp = chain.invoke(input,config=config)
        print(resp)
        self.assertIsInstance(resp["messages"][-1],HumanMessage)
        self.assertIsNotNone(resp["messages"][-1].content[0].get("text"))
        self.assertIsNotNone(resp["messages"][-1].content[1].get("text"))
        
if __name__ == '__main__':
    unittest.main()
