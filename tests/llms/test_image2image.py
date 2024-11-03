import unittest


class TestImage2Image(unittest.TestCase):

    def setUp(self):
        from agi.llms.image2image import Image2Image
        from agi.llms.base import MultiModalMessage,Image
        import torch
        self.instance = Image2Image()
        img = Image.new("tests/cat.jpg")
        
        self.assertIsNotNone(img)
        
        self.input = MultiModalMessage(content="as a tiger",image=img)

    def test_image2image(self):
        output = self.instance.invoke(self.input)
        self.assertIsNotNone(output)
        self.assertIsNotNone(output.image)
        self.assertIsNotNone(output.image.pil_image)
        self.assertIsNotNone(output.content)
        print(output.content)

        
if __name__ == "__main__":
    unittest.main()
