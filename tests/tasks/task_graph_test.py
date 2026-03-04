import unittest
from langchain_core.messages import AIMessage,HumanMessage,ToolMessage,AIMessageChunk
from agi.tasks.graph import AgiGraph
import asyncio
from agi.config import log


class TestGraph(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):       
        self.graph = AgiGraph()
        self.graph.display()
    
    async def test_agent(self):
        input_example = {
            "messages":  [
                HumanMessage(content="俄乌局势")
            ],
            "input_type": "text",
            "need_speech": False,
            "status": "in_progress",
            "feature": "agent"
        }
        resp = await self.graph.invoke(input_example)
        print(resp)
        self.assertIsInstance(resp,dict)
        self.assertIsInstance(resp["messages"],list)
        self.assertEqual(resp["input_type"],"text")
        self.assertIsInstance(resp["messages"][-1],AIMessage)
        self.assertIsNotNone(resp["messages"][-1].content)
        
        async for event in self.graph.stream(input_example):
            print(f"******event******{event,type(event)}")
            self.assertIsInstance(event,tuple)
            self.assertIsInstance(event[1][0],AIMessageChunk)
            self.assertIsInstance(event[1][1],dict)

    async def test_human_feedback(self):
        input_example = {
            "messages":  [
                HumanMessage(
                    content="11111111111",
                )
            ],
            "input_type": "text",
            "need_speech": False,
            "feature": "human",
            "user_id": "human_feedback"
        }
        
        for event in self.graph.stream(input_example):
            print(event)
            self.assertEqual(event[0],"updates")
            self.assertIsNotNone(event[1].get("__interrupt__"))
            self.assertEqual(event[1].get("__interrupt__")[0].value,"breaked")

        input_example = {
            "messages":  [
                HumanMessage(
                    content="22222222222222222",
                )
            ],
            "input_type": "text",
            "need_speech": False,
            "user_id": "human_feedback"
        }
        # TODO 此处没有返回
        for event in self.graph.stream(input_example):
            print(f"******event******{event,type(event)}")

        input_example = {
            "messages":  [
                HumanMessage(
                    content="3333333333333333333333",
                )
            ],
            "input_type": "text",
            "need_speech": False,
            "feature": "human",
            "user_id": "human_feedback1"
        }
        
        resp = await self.graph.invoke(input_example)
        print(resp)
        self.assertIsInstance(resp["messages"][-1],HumanMessage)
        self.assertEqual(resp["messages"][-1].content,"3333333333333333333333")        
        input_example = {
            "messages":  [
                HumanMessage(
                    content="444444444444444444",
                )
            ],
            "input_type": "text",
            "need_speech": False,
            "user_id": "human_feedback1"
        }
        resp = await self.graph.invoke(input_example)
        print(resp)
        self.assertIsInstance(resp["messages"][-1],AIMessage)
        self.assertEqual(resp["messages"][-1].content,"444444444444444444")
        
        