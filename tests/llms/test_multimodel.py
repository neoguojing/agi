import unittest
from langchain_core.messages import AIMessage, HumanMessage

from agi.config import log

class TestMultiModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from agi.llms.multimodel import MultiModel
        cls.instance = MultiModel()


    def test_text_input(self):
        input = HumanMessage(content="介绍下多模态模型")
        output = self.__class__.instance.invoke(input)
        self.assertIsNotNone(output)
        self.assertIsNotNone(output.content)
        self.assertIsInstance(output.content,list)
        self.assertEqual(output.content[0].get("type"),"text")
        self.assertIsNotNone(output.content[0].get("text"))
        print(output.content)

    def test_audio_input(self):
        input = HumanMessage(content=[{"type":"audio","audio":"tests/zh-cn-sample.wav"}])
        output = self.__class__.instance.invoke(input)
        self.assertIsNotNone(output)
        self.assertIsNotNone(output.content)
        self.assertIsInstance(output.content,list)
        self.assertEqual(output.content[0].get("type"),"text")
        self.assertIsNotNone(output.content[0].get("text"))
        print(output.content)

    def test_image_input(self):
        input = HumanMessage(content=[{"type":"image","image":"tests/cat.jpg"}])
        output = self.__class__.instance.invoke(input)
        self.assertIsNotNone(output)
        self.assertIsNotNone(output.content)
        self.assertIsInstance(output.content,list)
        self.assertEqual(output.content[0].get("type"),"text")
        self.assertIsNotNone(output.content[0].get("text"))
        print(output.content)

    def test_audio_output(self):
        config={"configurable": {"need_speech": True}}
        input = HumanMessage(content=[{"type":"image","image":"tests/cat.jpg"}])
        output = self.__class__.instance.invoke(input,config=config)
        self.assertIsNotNone(output)
        self.assertIsNotNone(output.content)
        self.assertIsInstance(output.content,list)
        self.assertEqual(output.content[0].get("type"),"audio")
        self.assertIsNotNone(output.content[0].get("audio"))
        print(output.content)

    # def test_video_input(self):
    #     input = HumanMessage(content=[{"type":"video","video":"../cat.jpg"}])
    #     output = self.__class__.instance.invoke(input)
    #     self.assertIsNotNone(output)
    #     self.assertIsNotNone(output.content)
    #     print(output.content)

        
if __name__ == "__main__":
    unittest.main()
