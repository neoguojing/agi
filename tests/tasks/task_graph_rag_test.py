import unittest
from langchain_core.messages import AIMessage,HumanMessage,ToolMessage,AIMessageChunk
from agi.tasks.graph import AgiGraph
import asyncio
from agi.config import log


class TestGraph(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):       
        self.graph = AgiGraph()
        self.graph.display()
   
    async def test_web(self):
        # TODO 引用无法返回给用户
        input_example = {
            "messages":  [
                HumanMessage(
                    content="俄乌战争消息"
                )
            ],
            "input_type": "text",
            "need_speech": False,
            "feature": "web",
        }
        resp = await self.graph.invoke(input_example)
        print(resp)
        self.assertIsInstance(resp,dict)
        self.assertIsInstance(resp["messages"],list)
        self.assertEqual(resp["feature"],"web")
        self.assertEqual(resp["input_type"],"text")
        self.assertIsInstance(resp["messages"][-1],AIMessage)
        self.assertIsNotNone(resp["messages"][-1].content)
        self.assertIsInstance(resp['citations'],list)
        
        async for event in self.graph.stream(input_example):
            print(f"******event******{event,type(event)}")
            if event[0]  == "custom":
                self.assertIsInstance(event[1],dict)
                self.assertIsNotNone(event[1].get("citations"))
            else:
                self.assertIsInstance(event,tuple)
                self.assertIsInstance(event[1][0],AIMessage)
                self.assertIsInstance(event[1][1],dict)
               
    async def test_custom_rag(self):
        import json
        input_example = {
            "messages":  [
                HumanMessage(
                    content="NTP3000Plus",
                )
            ],
            "collection_names": ["test"],
            "input_type": "text",
            "need_speech": False,
            "feature": "rag",
        }
        resp = await self.graph.invoke(input_example)
        print(resp)
        self.assertIsInstance(resp,dict)
        self.assertIsInstance(resp["messages"],list)
        self.assertEqual(resp["feature"],"rag")
        self.assertEqual(resp["input_type"],"text")
        self.assertIsInstance(resp["messages"][-1],AIMessage)
        self.assertIsNotNone(resp["messages"][-1].content)
        self.assertIsNotNone(resp['citations'])
        
        async for event in self.graph.stream(input_example):
            print(f"******event******{event,type(event)}")
            if event[0]  == "custom":
                self.assertIsInstance(event[1],dict)
                self.assertIsNotNone(event[1].get("citations"))
            else:
                self.assertIsInstance(event,tuple)
                self.assertIsInstance(event[1][0],AIMessage)
                self.assertIsInstance(event[1][1],dict)
                