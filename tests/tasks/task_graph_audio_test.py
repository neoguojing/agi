import unittest
from langchain_core.messages import AIMessage,HumanMessage,ToolMessage,AIMessageChunk
from agi.tasks.graph import AgiGraph

from agi.config import log


class TestGraph(unittest.TestCase):
    def setUp(self):        
        self.graph = AgiGraph()
        self.graph.display()
    
    def test_audio_input(self):
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
    