import unittest
from agi.llms.speech2text import Speech2Text
import logging

class TestSpeech2Text(unittest.TestCase):

    def setUp(self):
        
        from agi.llms.base import MultiModalMessage,Audio
        
        self.input = MultiModalMessage(audio=Audio.from_local("tests/1730604079.wav"))

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
