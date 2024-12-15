import unittest
from agi.tasks.task_factory import TaskFactory,TASK_AGENT
from langgraph.errors import GraphRecursionError
class TestAgent(unittest.TestCase):
    def setUp(self):
        self.agent = TaskFactory.create_task(TASK_AGENT)
        self.RECURSION_LIMIT = 5
        self.config={"configurable": {"user_id": "test", "conversation_id": "1"},"recursion_limit": self.RECURSION_LIMIT }
    def test_agent(self):
        query = "下周上海的天气如何?"
        try:
            messages = self.agent.invoke({"messages": [("human", query)]},config=self.config)
            print(messages)
        except GraphRecursionError:
            print({"input": query, "output": "Agent stopped due to max iterations."})
            
    # def test_agent_steam(self):
    #     query = "下周上海的天气如何?"
    #     try:
    #         for chunk in self.agent.stream(
    #             {"messages": [("human", query)]},
    #             config=self.config,
    #             stream_mode="values",
    #         ):
    #             print(chunk["messages"][-1])
    #     except GraphRecursionError:
    #         print({"input": query, "output": "Agent stopped due to max iterations."})