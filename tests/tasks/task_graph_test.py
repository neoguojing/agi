import unittest
from langchain_core.messages import AIMessage,HumanMessage,ToolMessage,AIMessageChunk
from agi.tasks.graph import AgiGraph

from agi.config import log


class TestGraph(unittest.TestCase):
    def setUp(self):        
        self.graph = AgiGraph()
        self.graph.display()
        
    def test_agent(self):
        input_example = {
            "messages":  [
                HumanMessage(content="俄乌局势")
            ],
            "input_type": "text",
            "need_speech": False,
            "status": "in_progress",
            "feature": "agent"
        }
        resp = self.graph.invoke(input_example)
        print(resp)
        self.assertIsInstance(resp,dict)
        self.assertIsInstance(resp["messages"],list)
        self.assertEqual(resp["input_type"],"text")
        self.assertIsInstance(resp["messages"][-1],AIMessage)
        self.assertIsNotNone(resp["messages"][-1].content)
        
        events = self.graph.stream(input_example)
        for event in events:
            print(f"******event******{event,type(event)}")
            self.assertIsInstance(event,tuple)
            self.assertIsInstance(event[1][0],AIMessageChunk)
            self.assertIsInstance(event[1][1],dict)

        
    def test_text_image_gene(self):
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
        resp = self.graph.invoke(input_example)
        print(resp)
        self.assertIsInstance(resp,dict)
        self.assertIsInstance(resp["messages"],list)
        self.assertEqual(resp["input_type"],"text")
        self.assertIsInstance(resp["messages"][-1],AIMessage)
        self.assertIsInstance(resp["messages"][-1].content,list)
        self.assertIsInstance(resp["messages"][-1].content[0],dict)
        self.assertEqual(resp["messages"][-1].content[0].get("type"),"image")
        self.assertIsNotNone(resp["messages"][-1].content[0].get("image"))
        

        events = self.graph.stream(input_example)
        for event in events:
            print(f"******event******{event,type(event)}")
            self.assertIsInstance(event,tuple)
            self.assertIsInstance(event[1][0],AIMessage)
            self.assertIsInstance(event[1][1],dict)

    
    def test_image_image_gene(self):
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
        resp = self.graph.invoke(input_example)
        print(resp)
        self.assertIsInstance(resp,dict)
        self.assertIsInstance(resp["messages"],list)
        self.assertEqual(resp["input_type"],"image")
        self.assertIsInstance(resp["messages"][-1],AIMessage)
        self.assertIsInstance(resp["messages"][-1].content,list)
        self.assertIsInstance(resp["messages"][-1].content[0],dict)
        self.assertEqual(resp["messages"][-1].content[0].get("type"),"image")
        self.assertIsNotNone(resp["messages"][-1].content[0].get("image"))

        events = self.graph.stream(input_example)
        for event in events:
            print(f"******event******{event,type(event)}")
            self.assertIsInstance(event,tuple)
            self.assertIsInstance(event[1][0],AIMessage)
            self.assertIsInstance(event[1][1],dict)
    
    def test_audio_input(self):
        # TODO 语音输出暂时不支持
        # 语音输入，语音输出
        input_example = {
            "messages":  [
                HumanMessage(
                    content=[
                        {"type":"audio","audio":"tests/zh-cn-sample.wav"},
                    ],
                )
            ],
            "input_type": "audio",
            "need_speech": True,
            "status": "in_progress",
        }
        resp = self.graph.invoke(input_example)
        print(resp)
        self.assertIsInstance(resp,dict)
        self.assertIsInstance(resp["messages"],list)
        self.assertEqual(resp["need_speech"],True)
        self.assertEqual(resp["input_type"],"audio")
        self.assertIsInstance(resp["messages"][-1],AIMessage)
        self.assertIsInstance(resp["messages"][-1].content,list)
        self.assertIsInstance(resp["messages"][-1].content[0],dict)
        self.assertEqual(resp["messages"][-1].content[0].get("type"),"audio")
        self.assertIsNotNone(resp["messages"][-1].content[0].get("audio"))
        self.assertIsNotNone(resp["messages"][-1].content[0].get("file_path"))
        self.assertIsNotNone(resp["messages"][-1].content[0].get("text"))

        events = self.graph.stream(input_example)
        for event in events:
            print(f"******event******{event,type(event)}")
            self.assertIsInstance(event,tuple)
            self.assertIsInstance(event[1][0],AIMessage)
            self.assertIsInstance(event[1][1],dict)
        
        input_example["need_speech"] = False
        resp = self.graph.invoke(input_example)
        print(resp)
        self.assertIsInstance(resp,dict)
        self.assertIsInstance(resp["messages"],list)
        self.assertEqual(resp["need_speech"],False)
        self.assertEqual(resp["input_type"],"audio")
        self.assertIsInstance(resp["messages"][-1],AIMessage)
        self.assertIsInstance(resp["messages"][-1].content,str)
        self.assertIsNotNone(resp["messages"][-1].content)
        
        events = self.graph.stream(input_example)
        for event in events:
            print(f"******event******{event,type(event)}")
            self.assertIsInstance(event,tuple)
            self.assertIsInstance(event[1][0],AIMessage)
            self.assertIsInstance(event[1][1],dict)
            
        input_example["feature"] = "speech"
        resp = self.graph.invoke(input_example)
        print(resp)
        self.assertIsInstance(resp,dict)
        self.assertIsInstance(resp["messages"],list)
        self.assertEqual(resp["need_speech"],False)
        self.assertEqual(resp["input_type"],"audio")
        self.assertIsInstance(resp["messages"][-1],AIMessage)
        self.assertIsInstance(resp["messages"][-1].content,str)
        self.assertEqual(resp["messages"][-1].content,"当我还只有六岁的时候,看到了一幅精彩的插画。")
        
        events = self.graph.stream(input_example)
        for event in events:
            print(f"******event******{event,type(event)}")
            self.assertIsInstance(event,tuple)
            self.assertIsInstance(event[1][0],AIMessage)
            self.assertIsInstance(event[1][1],dict)
        
    def test_web(self):
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
        resp = self.graph.invoke(input_example)
        print(resp)
        self.assertIsInstance(resp,dict)
        self.assertIsInstance(resp["messages"],list)
        self.assertEqual(resp["feature"],"web")
        self.assertEqual(resp["input_type"],"text")
        self.assertIsInstance(resp["messages"][-1],AIMessage)
        self.assertIsInstance(resp['docs'],list)
        self.assertIsNotNone(resp["messages"][-1].content)
        self.assertIsInstance(resp['citations'],list)
        
        
        events = self.graph.stream(input_example)
        for event in events:
            print(f"******event******{event,type(event)}")
            if event[0]  == "custom":
                self.assertIsInstance(event[1],dict)
                self.assertIsNotNone(event[1].get("citations"))
            else:
                self.assertIsInstance(event,tuple)
                self.assertIsInstance(event[1][0],AIMessage)
                self.assertIsInstance(event[1][1],dict)
                
    def test_custom_rag(self):
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
        resp = self.graph.invoke(input_example)
        print(resp)
        self.assertIsInstance(resp,dict)
        self.assertIsInstance(resp["messages"],list)
        self.assertEqual(resp["feature"],"rag")
        self.assertEqual(resp["input_type"],"text")
        self.assertIsInstance(resp["messages"][-1],AIMessage)
        self.assertIsNotNone(resp['docs'])
        self.assertIsNotNone(resp["messages"][-1].content)
        self.assertIsNotNone(resp['citations'])
        
        events = self.graph.stream(input_example)
        for event in events:
            print(f"******event******{event,type(event)}")
            if event[0]  == "custom":
                self.assertIsInstance(event[1],dict)
                self.assertIsNotNone(event[1].get("citations"))
            else:
                self.assertIsInstance(event,tuple)
                self.assertIsInstance(event[1][0],AIMessage)
                self.assertIsInstance(event[1][1],dict)