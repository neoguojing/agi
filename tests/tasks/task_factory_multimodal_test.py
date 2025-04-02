import unittest
from unittest.mock import patch, MagicMock
import threading
from langchain_core.messages import AIMessage, HumanMessage

# Assuming we import the TaskFactory and constants like TASK_LLM, TASK_EMBEDDING, etc.
from agi.tasks.task_factory import TaskFactory, TASK_TRANSLATE, TASK_IMAGE_GEN, TASK_TTS, TASK_SPEECH_TEXT
from agi.tasks.prompt import multimodal_input_template
from agi.tasks.agent import State
import logging
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

class TestTaskMultiModalFactory(unittest.TestCase):
    # def test_template(self):
    #     value = multimodal_input_template.invoke({"text":"123","image":"233","audio":"1223"})
    #     print(value.to_messages())
        
    # def test_translate_chain(self):
    #     # Test for TASK_LLM
    #     llm_task = TaskFactory.create_task(TASK_TRANSLATE)
    #     input = State(
    #         messages=[HumanMessage(content="我爱北京天安门")],
    #     )
    #     resp = llm_task.invoke(input)
    #     self.assertIsNotNone(resp)
    #     self.assertIsInstance(resp["messages"][-1],HumanMessage)
    #     self.assertIsInstance(resp["messages"][-1].content,str)
        
    # TODO torch 2.6不兼容
    # def test_text2speech_chain(self):
    #     # Test for TASK_LLM
    #     llm_task = TaskFactory.create_task(TASK_TTS)
    #     input = State(
    #         messages=[HumanMessage(content="These prompt templates are used to format a single string, and generally are used for simpler inputs")],
    #     )
    #     resp = llm_task.invoke(input)
    #     self.assertIsInstance(resp,AIMessage)
    #     self.assertIsInstance(resp.content,list)
    #     self.assertIsNotNone(resp.content[0].get("audio"))
    #     self.assertEqual(resp.content[0].get("type"),"audio")
        
    # def test_speech2text_chain(self):
    #     llm_task = TaskFactory.create_task(TASK_SPEECH_TEXT)
    #     input = State(
    #         messages=[HumanMessage(content=[{"type":"audio","audio":"tests/zh-cn-sample.wav"}])],
    #     )
    #     resp = llm_task.invoke(input)
    #     self.assertIsNotNone(resp.get("messages"))
    #     self.assertIsInstance(resp.get("messages")[-1],HumanMessage)
    #     self.assertEqual(len(resp.get("messages")[-1].content),2)
    #     input["feature"] = "speech"
    #     resp = llm_task.invoke(input)
    #     self.assertIsInstance(resp,AIMessage)
    #     self.assertIsInstance(resp.content,str)
    #     self.assertIsNotNone(resp.content)
        
    def test_text2image_chain(self):
        llm_task = TaskFactory.create_task(TASK_IMAGE_GEN)
        input = State(
            messages=[HumanMessage(content=[{"type":"text","text":"星辰大海"}])],
        )
        resp = llm_task.invoke(input)
        print(resp)
        self.assertIsInstance(resp,AIMessage)
        self.assertIsInstance(resp.content,list)
        self.assertIsNotNone(resp.content[0].get("image"))
        self.assertEqual(resp.content[0].get("type"),"image")
        # self.assertIsNotNone(resp.content)
        input = State(
            messages=[HumanMessage(content=[{"type":"text","text":"猫咪在游泳"},{"type":"image","image":"tests/cat.jpg"}])],
        )
        resp = llm_task.invoke(input)
        self.assertIsInstance(resp,AIMessage)
        self.assertIsInstance(resp.content,list)
        self.assertIsNotNone(resp.content[0].get("image"))
        self.assertEqual(resp.content[0].get("type"),"image")

if __name__ == '__main__':
    unittest.main()
