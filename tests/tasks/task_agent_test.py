import unittest
from agi.tasks.task_factory import TaskFactory,TASK_AGENT
from langgraph.errors import GraphRecursionError
from langchain_core.messages import AIMessage
import uuid
import logging
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
class TestAgent(unittest.TestCase):
    def setUp(self):
        self.agent = TaskFactory.create_task(TASK_AGENT)
        self.RECURSION_LIMIT = 5
        self.config={"configurable": {"user_id": "test", "conversation_id": "1","thread_id": str(uuid.uuid4())},
                     "recursion_limit": self.RECURSION_LIMIT }
    def test_agent(self):
        query = "查询tesla股票价格"
        try:
            messages = self.agent.invoke({"messages": [("human", query)]},config=self.config)
            print(messages)
            messages = messages.get("messages",[])
            self.assertIsInstance(messages,list)
            self.assertIsInstance(messages[-1],AIMessage)
            self.assertIsInstance(messages[-1].content,str)
        except GraphRecursionError:
            print({"input": query, "output": "Agent stopped due to max iterations."})
            
    def test_agent_steam(self):
        query = "下周上海的天气如何?"
        try:
            for chunk in self.agent.stream(
                {"messages": [("human", query)]},
                config=self.config,
                stream_mode="values",
            ):
                self.assertIsInstance(chunk["messages"],list)
                self.assertIsInstance(chunk["messages"][-1].content,str)
        except GraphRecursionError:
            print({"input": query, "output": "Agent stopped due to max iterations."})