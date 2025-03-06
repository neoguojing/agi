import unittest
from agi.tasks.task_factory import TaskFactory,TASK_AGENT
from langgraph.errors import GraphRecursionError
from langchain_core.messages import AIMessage,HumanMessage,ToolMessage
from agi.tasks.graph import AgiGraph
class TestAgent(unittest.TestCase):
    def setUp(self):        
        self.graph = AgiGraph()
        self.graph.display()
    # def test_agi(self):
    #     input_example = {
    #         "messages":  [
    #             HumanMessage(content="俄乌局势")
    #         ],
    #         "input_type": "text",
    #         "need_speech": False,
    #         "status": "in_progress",
    #     }
    #     resp = self.graph.invoke(input_example)
    #     print(resp["messages"][-1],AIMessage)
    #     self.assertIsInstance(resp,dict)
    #     self.assertIsInstance(resp["messages"],list)
    #     self.assertEqual(resp["input_type"],"text")
    #     self.assertIsInstance(resp["messages"][-1],AIMessage)
    #     # 使用agent，由agent决策是否调用图片生成工具
    #     input_example = {
    #         "messages":  [
    #             HumanMessage(
    #                 content="生成一张超人拯救了太阳",
    #             )
    #         ],
    #         "input_type": "text",
    #         "need_speech": False,
    #         "status": "in_progress",
    #     }
    #     resp = self.graph.invoke(input_example)
    #     print(resp["messages"][-1])
    #     self.assertIsInstance(resp,dict)
    #     self.assertIsInstance(resp["messages"],list)
    #     self.assertEqual(resp["input_type"],"text")
    #     self.assertIsInstance(resp["messages"][-1],ToolMessage)
    #     input_example = {
    #         "messages":  [
    #             HumanMessage(
    #                 content=[
    #                     {"type":"text","text":"猫咪是黑猫警长"},
    #                     {"type":"image","image":"tests/cat.jpg"},
    #                 ],
    #             )
    #         ],
    #         "input_type": "image",
    #         "need_speech": False,
    #         "status": "in_progress",
    #     }
    #     resp = self.graph.invoke(input_example)
    #     self.assertIsInstance(resp,dict)
    #     self.assertIsInstance(resp["messages"],list)
    #     self.assertEqual(resp["input_type"],"image")
    #     self.assertIsInstance(resp["messages"][-1],AIMessage)
        
    #     input_example = {
    #         "messages":  [
    #             HumanMessage(
    #                 content=[
    #                     {"type":"audio","audio":"tests/1730604079.wav"},
    #                 ],
    #             )
    #         ],
    #         "input_type": "audio",
    #         "need_speech": True,
    #         "status": "in_progress",
    #     }
    #     resp = self.graph.invoke(input_example)
    #     self.assertIsInstance(resp,dict)
    #     self.assertIsInstance(resp["messages"],list)
    #     self.assertEqual(resp["need_speech"],True)
    #     self.assertEqual(resp["input_type"],"audio")
    #     self.assertIsInstance(resp["messages"][-1],AIMessage)
        
    def test_web(self):
        input_example = {
            "messages":  [
                HumanMessage(
                    content="今天上海天气如何？"
                )
            ],
            "input_type": "text",
            "need_speech": False,
            "feature": "web",
        }
        resp = self.graph.invoke(input_example)
        print(resp)
        # self.assertIsInstance(resp,dict)
        # self.assertIsInstance(resp["messages"],list)
        # self.assertEqual(resp["need_speech"],True)
        # self.assertEqual(resp["input_type"],"audio")
        # self.assertIsInstance(resp["messages"][-1],AIMessage)

    def test_custom_rag(self):
        input_example = {
            "messages":  [
                HumanMessage(
                    content="今天上海天气如何？"
                )
            ],
            "input_type": "text",
            "need_speech": False,
            "feature": "rag",
        }
        resp = self.graph.invoke(input_example)
        print(resp)
        # self.assertIsInstance(resp,dict)
        # self.assertIsInstance(resp["messages"],list)
        # self.assertEqual(resp["need_speech"],True)
        # self.assertEqual(resp["input_type"],"audio")
        # self.assertIsInstance(resp["messages"][-1],AIMessage)