import unittest
from langchain_core.messages import AIMessage, HumanMessage
import logging
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
class TestMultiModel(unittest.TestCase):

    def setUp(self):
        from agi.llms.multimodel import MultiModel
        self.instance = MultiModel()

    def test_text_input(self):
        input = HumanMessage(content="介绍下多模态模型")
        output = self.instance.invoke(input)
        self.assertIsNotNone(output)
        self.assertIsNotNone(output.content)
        print(output.content)

    def test_audio_input(self):
        input = HumanMessage(content=[{"type":"audio","audio":"../zh-cn-sample.wav"}])
        output = self.instance.invoke(input)
        self.assertIsNotNone(output)
        self.assertIsNotNone(output.content)
        print(output.content)

    def test_image_input(self):
        input = HumanMessage(content=[{"type":"image","image":"../cat.jpg"}])
        output = self.instance.invoke(input)
        self.assertIsNotNone(output)
        self.assertIsNotNone(output.content)
        print(output.content)

    def test_audio_output(self):
        config={"configurable": {"need_speech": True}}
        input = HumanMessage(content=[{"type":"image","image":"../cat.jpg"}])
        output = self.instance.invoke(input,config=config)
        self.assertIsNotNone(output)
        self.assertIsNotNone(output.content)
        print(output.content)

    # def test_video_input(self):
    #     input = HumanMessage(content=[{"type":"video","video":"../cat.jpg"}])
    #     output = self.instance.invoke(input)
    #     self.assertIsNotNone(output)
    #     self.assertIsNotNone(output.content)
    #     print(output.content)

        
if __name__ == "__main__":
    unittest.main()
