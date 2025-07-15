import unittest
from agi.llms.speech2text import Speech2Text

from langchain_core.messages import AIMessage,HumanMessage
from pathlib import Path
from agi.config import log
from agi.utils.common import file_to_data_uri

class TestSpeech2Text(unittest.TestCase):

    def setUp(self):
        audio_data = Path("tests/zh-cn-sample.wav")
        self.input = HumanMessage(content=[
                {"type":"audio","audio":audio_data}
            ]
        )

    def test_speach2text(self):
        self.instance = Speech2Text()
        output = self.instance.invoke(self.input)
        self.assertIsNotNone(output)
        self.assertIsNotNone(output.content)
        print(f"test_speach2text:{output.content}")
    
    def test_speach2text_cpu(self):  
        self.instance = Speech2Text(device="cpu")
        output = self.instance.invoke(self.input)
        self.assertIsNotNone(output)
        self.assertIsNotNone(output.content)
        print(f"test_speach2text_cpu:{output.content}")

        
if __name__ == "__main__":
    unittest.main()
