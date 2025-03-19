import unittest
from unittest.mock import patch, MagicMock
import threading
from langchain_core.messages import AIMessage, HumanMessage

# Assuming we import the TaskFactory and constants like TASK_LLM, TASK_EMBEDDING, etc.
from agi.tasks.task_factory import TaskFactory, TASK_TRANSLATE, TASK_IMAGE_GEN, TASK_TTS, TASK_SPEECH_TEXT, TASK_RETRIEVER, TASK_AGENT
import logging
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

class TestTaskMultiModalFactory(unittest.TestCase):

    def test_translate_chain(self):
        # Test for TASK_LLM
        llm_task = TaskFactory.create_task(TASK_TRANSLATE)
        resp = llm_task.invoke({"text":"我爱北京天安门"})
        self.assertIsInstance(resp,str)
        self.assertIsNotNone(resp)
    def test_text2speech_chain(self):
        # Test for TASK_LLM
        llm_task = TaskFactory.create_task(TASK_TTS)
        resp = llm_task.invoke({"text":"These prompt templates are used to format a single string, and generally are used for simpler inputs",
                                "image":"","audio":""})
        self.assertIsInstance(resp,AIMessage)
        self.assertIsInstance(resp.content,list)
        self.assertIsNotNone(resp.content[0].get("audio"))
        self.assertEqual(resp.content[0].get("type"),"audio")
    def test_speech2text_chain(self):
        llm_task = TaskFactory.create_task(TASK_SPEECH_TEXT)
        # resp = llm_task.invoke({"data":"tests/zh-cn-sample.wav","type":"audio"})
        resp = llm_task.invoke({"audio":"tests/zh-cn-sample.wav","text":"","image":""})
        self.assertIsInstance(resp,AIMessage)
        self.assertIsInstance(resp.content,str)
        self.assertIsNotNone(resp.content)
    def test_text2image_chain(self):
        llm_task = TaskFactory.create_task(TASK_IMAGE_GEN)
        resp = llm_task.invoke({"text":"星辰大海","image":"","audio":""})
        log.debug(resp)
        self.assertIsInstance(resp,AIMessage)
        self.assertIsInstance(resp.content,list)
        self.assertIsNotNone(resp.content[0].get("image"))
        self.assertEqual(resp.content[0].get("type"),"image")
        # self.assertIsNotNone(resp.content)
        resp = llm_task.invoke({"text":"猫咪在游泳","image":"tests/cat.jpg","audio":""})
        self.assertIsInstance(resp,AIMessage)
        self.assertIsInstance(resp.content,list)
        self.assertIsNotNone(resp.content[0].get("image"))
        self.assertEqual(resp.content[0].get("type"),"image")

if __name__ == '__main__':
    unittest.main()
