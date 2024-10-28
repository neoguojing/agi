import unittest
from agi.llms.speech2text import Speech2Text

class TestSpeech2Text(unittest.TestCase):

    def setUp(self):
        
        from agi.llms.base import MultiModalMessage,Audio
        
        self.input = MultiModalMessage(audio=Audio.from_local("cache/2024_10_28/1730087840.wav"))

    def test_speach2text(self):
        self.instance = Speech2Text()
        output = self.instance.invoke(self.input)
        self.assertIsNotNone(output)
        self.assertIsNotNone(output.content)
        print(output.content)
        
        self.instance = Speech2Text(device="cpu")
        output = self.instance.invoke(self.input)
        self.assertIsNotNone(output)
        self.assertIsNotNone(output.content)
        print(output.content)

        
if __name__ == "__main__":
    unittest.main()
