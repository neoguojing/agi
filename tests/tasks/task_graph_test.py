import unittest
from langchain_core.messages import AIMessage,HumanMessage,ToolMessage,AIMessageChunk
from agi.tasks.graph import AgiGraph
import asyncio
from agi.config import log


class TestGraph(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):       
        self.graph = AgiGraph()
        await self.graph.display()
    
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
        
        events = await self.graph.stream(input_example)
        async for event in events:
            print(f"******event******{event,type(event)}")
            self.assertIsInstance(event,tuple)
            self.assertIsInstance(event[1][0],AIMessageChunk)
            self.assertIsInstance(event[1][1],dict)

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
        

        events = await self.graph.stream(input_example)
        async for event in events:
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

        events = await self.graph.stream(input_example)
        async for event in events:
            print(f"******event******{event,type(event)}")
            self.assertIsInstance(event,tuple)
            self.assertIsInstance(event[1][0],AIMessage)
            self.assertIsInstance(event[1][1],dict)
    
   
    async def test_web(self):
        # TODO 引用无法返回给用户
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
        resp = await self.graph.invoke(input_example)
        print(resp)
        self.assertIsInstance(resp,dict)
        self.assertIsInstance(resp["messages"],list)
        self.assertEqual(resp["feature"],"web")
        self.assertEqual(resp["input_type"],"text")
        self.assertIsInstance(resp["messages"][-1],AIMessage)
        self.assertIsInstance(resp['docs'],list)
        self.assertIsNotNone(resp["messages"][-1].content)
        self.assertIsInstance(resp['citations'],list)
        
        
        events = await self.graph.stream(input_example)
        async for event in events:
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
        self.assertIsNotNone(resp['docs'])
        self.assertIsNotNone(resp["messages"][-1].content)
        self.assertIsNotNone(resp['citations'])
        
        events = await self.graph.stream(input_example)
        async for event in events:
            print(f"******event******{event,type(event)}")
            if event[0]  == "custom":
                self.assertIsInstance(event[1],dict)
                self.assertIsNotNone(event[1].get("citations"))
            else:
                self.assertIsInstance(event,tuple)
                self.assertIsInstance(event[1][0],AIMessage)
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
        
        resp = await self.graph.stream(input_example)
        for event in resp:
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
        resp = await self.graph.stream(input_example)
        # TODO 此处没有返回
        for event in resp:
            print(f"******event******{event,type(event)}")

        # input_example = {
        #     "messages":  [
        #         HumanMessage(
        #             content="3333333333333333333333",
        #         )
        #     ],
        #     "input_type": "text",
        #     "need_speech": False,
        #     "feature": "human",
        #     "user_id": "human_feedback1"
        # }
        
        # resp = await self.graph.invoke(input_example)
        # print(resp)
        # self.assertIsInstance(resp["messages"][-1],HumanMessage)
        # self.assertEqual(resp["messages"][-1].content,"3333333333333333333333")        
        # input_example = {
        #     "messages":  [
        #         HumanMessage(
        #             content="444444444444444444",
        #         )
        #     ],
        #     "input_type": "text",
        #     "need_speech": False,
        #     "user_id": "human_feedback1"
        # }
        # resp = await self.graph.invoke(input_example)
        # print(resp)
        # self.assertIsInstance(resp["messages"][-1],AIMessage)
        # self.assertEqual(resp["messages"][-1].content,"444444444444444444")
        