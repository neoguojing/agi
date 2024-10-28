import unittest


class TestText2Image(unittest.TestCase):

    def setUp(self):
        from agi.llms.speech2text import Speech2Text
        from agi.llms.base import MultiModalMessage,Audio
        import torch
        self.instance = Speech2Text()
        self.input = MultiModalMessage(content="a midlife crisis man")

    def test_image2image(self):
        output = self.instance.invoke(self.input)
        self.assertIsNotNone(output)
        self.assertIsNotNone(output.image)
        self.assertIsNotNone(output.image.pil_image)
        self.assertIsNotNone(output.content)
        print(output.content)

        
if __name__ == "__main__":
    unittest.main()
