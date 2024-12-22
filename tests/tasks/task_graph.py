import unittest
from agi.tasks.task_factory import TaskFactory,TASK_AGENT
from langgraph.errors import GraphRecursionError
from langchain_core.messages import AIMessage,HumanMessage
from agi.tasks.graph import AgiGraph
class TestAgent(unittest.TestCase):
    def setUp(self):        
        self.graph = AgiGraph()
    def test_agi(self):
        self.graph.display()
        input_example = {
            "messages":  [
                HumanMessage(
                    content="画一幅太阳",
                )
            ],
            "input_type": "image",
            "need_speech": False,
            "status": "in_progress",
        }
        resp = self.graph.invoke(input_example)
        print(resp)
        # input_example = {
        #     "messages":  [
        #         HumanMessage(
        #             content="超人拯救了太阳",
        #             additional_kwargs={"image":"/win/text-generation-webui/apps/pics/output/2024_09_16/1726452758.png"}
        #         )
        #     ],
        #     "input_type": "image",
        #     "need_speech": False,
        #     "status": "in_progress",
        # }

        # input_example = {
        #     "messages":  [
        #         HumanMessage(
        #             content="俄乌战争进展",
        #         )
        #     ],
        #     "input_type": "text",
        #     "need_speech": False,
        #     "status": "in_progress",
        # }
        