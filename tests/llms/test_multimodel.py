import unittest
from langchain_core.messages import AIMessage, HumanMessage
from agi.utils.common import file_to_data_uri
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
        audio_data = file_to_data_uri("tests/zh-cn-sample.wav")
        input = HumanMessage(content=[{"type":"audio","audio":audio_data}])
        output = self.__class__.instance.invoke(input)
        self.assertIsNotNone(output)
        self.assertIsNotNone(output.content)
        self.assertIsInstance(output.content,list)
        self.assertEqual(output.content[0].get("type"),"text")
        self.assertIsNotNone(output.content[0].get("text"))
        print(output.content)

    def test_image_input(self):
        image_data = file_to_data_uri("tests/cat.jpg")
        input = HumanMessage(content=[{"type":"image","image":image_data}])
        output = self.__class__.instance.invoke(input)
        self.assertIsNotNone(output)
        self.assertIsNotNone(output.content)
        self.assertIsInstance(output.content,list)
        self.assertEqual(output.content[0].get("type"),"text")
        self.assertIsNotNone(output.content[0].get("text"))
        print(output.content)

    def test_audio_output(self):
        config={"configurable": {"need_speech": True}}
        image_data = file_to_data_uri("tests/cat.jpg")
        input = HumanMessage(content=[{"type":"image","image":image_data}])
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
