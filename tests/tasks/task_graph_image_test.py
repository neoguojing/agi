import unittest
from langchain_core.messages import AIMessage,HumanMessage,ToolMessage,AIMessageChunk
from agi.tasks.graph import AgiGraph
import asyncio
from agi.config import log


class TestGraph(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):       
        self.graph = AgiGraph()
        self.graph.display()
    
    async def test_text_image_gene(self):
        # 使用agent，由agent决策是否调用图片生成工具
        input_example = {
            "messages":  [
                HumanMessage(
                    content="生成一张超人拯救了太阳",
                )
            ],
            "input_type": "text",
            "need_speech": False,
            "status": "in_progress",
        }
        resp = await self.graph.invoke(input_example)
        print(resp)
        self.assertIsInstance(resp,dict)
        self.assertIsInstance(resp["messages"],list)
        self.assertEqual(resp["input_type"],"text")
        self.assertIsInstance(resp["messages"][-1],AIMessage)
        self.assertIsInstance(resp["messages"][-1].content,list)
        self.assertIsInstance(resp["messages"][-1].content[0],dict)
        self.assertEqual(resp["messages"][-1].content[0].get("type"),"image")
        self.assertIsNotNone(resp["messages"][-1].content[0].get("image"))
        
        async for event in self.graph.stream(input_example):
            print(f"******event******{event,type(event)}")
            self.assertIsInstance(event,tuple)
            self.assertIsInstance(event[1][0],AIMessage)
            self.assertIsInstance(event[1][1],dict)

    
    async def test_image_image_gene(self):
        input_example = {
            "messages":  [
                HumanMessage(
                    content=[
                        {"type":"text","text":"猫咪是黑猫警长"},
                        {"type":"image","image":"tests/cat.jpg"},
                    ],
                )
            ],
            "input_type": "image",
            "feature": "image2image",
            "need_speech": False,
            "status": "in_progress",
        }
        resp = await self.graph.invoke(input_example)
        print(resp)
        self.assertIsInstance(resp,dict)
        self.assertIsInstance(resp["messages"],list)
        self.assertEqual(resp["input_type"],"image")
        self.assertIsInstance(resp["messages"][-1],AIMessage)
        self.assertIsInstance(resp["messages"][-1].content,list)
        self.assertIsInstance(resp["messages"][-1].content[0],dict)
        self.assertEqual(resp["messages"][-1].content[0].get("type"),"image")
        self.assertIsNotNone(resp["messages"][-1].content[0].get("image"))

        async for event in self.graph.stream(input_example):
            print(f"******event******{event,type(event)}")
            self.assertIsInstance(event,tuple)
            self.assertIsInstance(event[1][0],AIMessage)
            self.assertIsInstance(event[1][1],dict)

            