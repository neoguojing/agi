import unittest
from agi.llms.speech2text import Speech2Text
import logging
from langchain_core.messages import AIMessage,HumanMessage
class TestSpeech2Text(unittest.TestCase):

    def setUp(self):
        self.input = HumanMessage(content=[
                {"type":"audio","audio":"tests/zh-cn-sample.wav"}
            ]
        )

    def test_speach2text(self):
        self.instance = Speech2Text()
        output = self.instance.invoke(self.input)
        self.assertIsNotNone(output)
        self.assertIsNotNone(output.content)
        print("test_speach2text:",output.content)
    
    def test_speach2text_cpu(self):  
        self.instance = Speech2Text(device="cpu")
        output = self.instance.invoke(self.input)
        self.assertIsNotNone(output)
        self.assertIsNotNone(output.content)
        print("test_speach2text_cpu:",output.content)

        
if __name__ == "__main__":
    unittest.main()
