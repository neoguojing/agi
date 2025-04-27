import unittest
from langchain_core.messages import AIMessage, HumanMessage

from agi.config import log

class TestText2Image(unittest.TestCase):

    def setUp(self):
        from agi.llms.text2image import Text2Image
        self.instance = Text2Image()
        self.input = HumanMessage(content="a midlife crisis man")

    def test_image2image(self):
        output = self.instance.invoke(self.input)
        self.assertIsNotNone(output)
        self.assertIsNotNone(output.content)
        print(output.content)

        
if __name__ == "__main__":
    unittest.main()
