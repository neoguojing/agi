import unittest
from agi.tasks.task_factory import TaskFactory,TASK_AGENT
from langgraph.errors import GraphRecursionError
from langchain_core.messages import AIMessage,HumanMessage,ToolMessage
from agi.tasks.graph import AgiGraph
class TestAgent(unittest.TestCase):
    def setUp(self):        
        self.graph = AgiGraph()
        self.graph.display()
        
    # def test_text(self):
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
        
    # def test_text_image_gene(self):
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
    
    # def test_image_image_gene(self):
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
        # print(resp)
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
        human_message_count = 0
        tool_message_count = 0
        ai_message_count = 0
        events = self.graph.stream(input_example)
        for event in events:
            print("******event******",event,type(event))
            if isinstance(event,HumanMessage):
                human_message_count += 1
            elif isinstance(event,ToolMessage):
                tool_message_count += 1
            elif isinstance(event,AIMessage):
                ai_message_count += 1
                if isinstance(event.content,list):
                    self.assertIsInstance(event.content[0],dict)
                    self.assertEqual(event.content[0].get("type"),"audio")
                    self.assertIsNotNone(event.content[0].get("audio"))
                    self.assertIsNotNone(event.content[0].get("file_path"))
                    self.assertIsNotNone(event.content[0].get("text"))
                else:
                    self.assertIsInstance(event.content,str)
        # TODO tool_message_count 为什么会有两个
        print("human_message_count:",human_message_count,"tool_message_count:",tool_message_count,"ai_message_count:",ai_message_count)
    
        
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
        
        human_message_count = 0
        tool_message_count = 0
        ai_message_count = 0
        events = self.graph.stream(input_example)
        for event in events:
            print("******event******",event,type(event))
            if isinstance(event,HumanMessage):
                human_message_count += 1
            elif isinstance(event,ToolMessage):
                tool_message_count += 1
            elif isinstance(event,AIMessage):
                ai_message_count += 1
                self.assertIsInstance(event.content,str)
        # TODO tool_message_count 为什么会有两个
        print("human_message_count:",human_message_count,"tool_message_count:",tool_message_count,"ai_message_count:",ai_message_count)
    
        
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
        
        human_message_count = 0
        tool_message_count = 0
        ai_message_count = 0
        events = self.graph.stream(input_example)
        for event in events:
            print("******event******",event,type(event))
            if isinstance(event,HumanMessage):
                human_message_count += 1
            elif isinstance(event,ToolMessage):
                tool_message_count += 1
            elif isinstance(event,AIMessage):
                ai_message_count += 1
                self.assertIsInstance(event.content,str)
                self.assertEqual(event.content,"当我还只有六岁的时候,看到了一幅精彩的插画。")
        # TODO tool_message_count 为什么会有两个
        print("human_message_count:",human_message_count,"tool_message_count:",tool_message_count,"ai_message_count:",ai_message_count)

        
    # def test_web(self):
    #     input_example = {
    #         "messages":  [
    #             HumanMessage(
    #                 content="今天上海天气如何？"
    #             )
    #         ],
    #         "input_type": "text",
    #         "need_speech": False,
    #         "feature": "web",
    #     }
    #     resp = self.graph.invoke(input_example)
    #     print(resp)
    #     self.assertIsInstance(resp,dict)
    #     self.assertIsInstance(resp["messages"],list)
    #     self.assertEqual(resp["feature"],"web")
    #     self.assertEqual(resp["input_type"],"text")
    #     self.assertIsInstance(resp["messages"][-1],AIMessage)
    #     self.assertIsInstance(resp["messages"][-1].additional_kwargs['context'],list)
    #     self.assertIsNotNone(resp["messages"][-1].content)
    #     self.assertIsInstance(resp["messages"][-1].additional_kwargs['citations'],list)
        
    #     human_message_count = 0
    #     tool_message_count = 0
    #     ai_message_count = 0
    #     events = self.graph.stream(input_example)
    #     for event in events:
    #         print("******event******",event,type(event))
    #         if isinstance(event,HumanMessage):
    #             human_message_count += 1
    #         elif isinstance(event,ToolMessage):
    #             tool_message_count += 1
    #             self.assertIsInstance(event.additional_kwargs['context'],list)
    #             self.assertIsInstance(event.additional_kwargs['citations'],list)
    #             self.assertEqual(event.content,"working...")
    #             self.assertEqual(event.tool_call_id,"web or rag")
    #         elif isinstance(event,AIMessage):
    #             ai_message_count += 1
    #             self.assertIsInstance(event.additional_kwargs['context'],list)
    #             self.assertIsInstance(event.additional_kwargs['citations'],list)
    #             self.assertIsNotNone(event.content)
    #     # TODO tool_message_count 为什么会有两个
    #     print("human_message_count:",human_message_count,"tool_message_count:",tool_message_count,"ai_message_count:",ai_message_count)

    # def test_custom_rag(self):
    #     import json
    #     input_example = {
    #         "messages":  [
    #             HumanMessage(
    #                 content="NTP3000Plus",
    #                 additional_kwargs={"collection_names":json.dumps(["test"])}
    #             )
    #         ],
    #         "input_type": "text",
    #         "need_speech": False,
    #         "feature": "rag",
    #     }
    #     resp = self.graph.invoke(input_example)
    #     print("resp:\n",resp)
    #     self.assertIsInstance(resp,dict)
    #     self.assertIsInstance(resp["messages"],list)
    #     self.assertEqual(resp["feature"],"rag")
    #     self.assertEqual(resp["input_type"],"text")
    #     self.assertIsInstance(resp["messages"][-1],AIMessage)
    #     self.assertIsInstance(resp["messages"][-1].additional_kwargs['context'],list)
    #     self.assertIsNotNone(resp["messages"][-1].content)
    #     self.assertIsInstance(resp["messages"][-1].additional_kwargs['citations'],list)
        
    #     human_message_count = 0
    #     tool_message_count = 0
    #     ai_message_count = 0
    #     events = self.graph.stream(input_example)
    #     for event in events:
    #         print("******event******",event,type(event))
    #         if isinstance(event,HumanMessage):
    #             human_message_count += 1
    #         elif isinstance(event,ToolMessage):
    #             tool_message_count += 1
    #             self.assertIsInstance(event.additional_kwargs['context'],list)
    #             self.assertIsInstance(event.additional_kwargs['citations'],list)
    #             self.assertEqual(event.content,"working...")
    #             self.assertEqual(event.tool_call_id,"web or rag")
    #         elif isinstance(event,AIMessage):
    #             ai_message_count += 1
    #             self.assertIsInstance(event.additional_kwargs['context'],list)
    #             self.assertIsInstance(event.additional_kwargs['citations'],list)
    #             self.assertIsNotNone(event.content)
    #     # TODO tool_message_count 为什么会有两个
    #     print("human_message_count:",human_message_count,"tool_message_count:",tool_message_count,"ai_message_count:",ai_message_count)