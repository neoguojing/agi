import unittest
from unittest.mock import patch, MagicMock
import threading
from langchain_core.messages import AIMessage, HumanMessage

# Assuming we import the TaskFactory and constants like TASK_LLM, TASK_EMBEDDING, etc.
from agi.tasks.task_factory import TaskFactory, TASK_LLM, TASK_EMBEDDING, TASK_LLM_WITH_HISTORY, TASK_LLM_WITH_RAG, TASK_TRANSLATE, TASK_IMAGE_GEN, TASK_TTS, TASK_SPEECH_TEXT, TASK_RETRIEVER, TASK_AGENT


class TestTaskFactory(unittest.TestCase):

    def test_translate_chain(self):
        # Test for TASK_LLM
        llm_task = TaskFactory.create_task(TASK_TRANSLATE)
        resp = llm_task.invoke({"text":"These prompt templates are used to format a single string, and generally are used for simpler inputs"})
        print("*********",resp)
        # self.assertIsInstance(resp,HumanMessage)
    def test_text2speech_chain(self):
        # Test for TASK_LLM
        llm_task = TaskFactory.create_task(TASK_TTS)
        resp = llm_task.invoke({"text":"These prompt templates are used to format a single string, and generally are used for simpler inputs"})
        print("*********",resp)
        # self.assertIsInstance(resp,HumanMessage)

if __name__ == '__main__':
    unittest.main()
