import unittest
from unittest.mock import patch, MagicMock
import threading
from langchain_core.messages import AIMessage, HumanMessage
from agi.tasks.define import State
from agi.tasks.task_factory import TaskFactory, TASK_LLM, TASK_LLM_WITH_HISTORY

from agi.config import log

class TestLLMFactory(unittest.TestCase):

    def test_llm(self):
        llm_task = TaskFactory.create_task(TASK_LLM)
        input = State(
            messages=[HumanMessage(content="介绍下llm")],
        )
        
        resp = llm_task.invoke(input)
        self.assertIsInstance(resp,AIMessage)
        self.assertIsInstance(resp.content,str)
        self.assertIsInstance(resp.response_metadata,dict)
        self.assertIsNotNone(resp.response_metadata['token_usage'])
        self.assertIsNotNone(resp.response_metadata['model_name'])
    
    def test_embbeding(self):
        llm_task = TaskFactory.get_embedding()
        resp = llm_task.embed_query("我爱北京天安门")
        self.assertIsInstance(resp,list)
        
    def test_llm_with_history(self):
        config={"configurable": {"user_id": "test", "conversation_id": "1"}}
        llm_task = TaskFactory.create_task(TASK_LLM_WITH_HISTORY)
        
        resp = llm_task.invoke([HumanMessage(content="你好，我是neo")],config=config)
        print(resp)
        self.assertIsInstance(resp.content,str)
        self.assertIsInstance(resp.response_metadata,dict)
        self.assertIsNotNone(resp.response_metadata['token_usage'])
        self.assertIsNotNone(resp.response_metadata['model_name'])
        resp = llm_task.invoke([HumanMessage(content="火星上有生命吗？")],config=config)
        print(resp)
        self.assertIsInstance(resp.content,str)
        self.assertIsInstance(resp.response_metadata,dict)
        self.assertIsNotNone(resp.response_metadata['token_usage'])
        self.assertIsNotNone(resp.response_metadata['model_name'])
        resp = llm_task.invoke([HumanMessage(content="我的名字叫什么？")],config=config)
        print(resp)
        self.assertIsInstance(resp.content,str)
        if "Neo" not in resp.content:
            self.assertIsNone(None)
        self.assertIsInstance(resp.response_metadata,dict)
        self.assertIsNotNone(resp.response_metadata['token_usage'])
        self.assertIsNotNone(resp.response_metadata['model_name'])
        
if __name__ == '__main__':
    unittest.main()
