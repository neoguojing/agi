import unittest
from agi.llms.base import Image,build_multi_modal_message,ImageType

class TestImage2Image(unittest.TestCase):

    def setUp(self):
        from agi.llms.image2image import Image2Image
        
        import torch
        self.instance = Image2Image()
        img = Image.new("tests/cat.jpg",ImageType.FILE_PATH)
        
        self.assertIsNotNone(img)
        
        self.input = build_multi_modal_message("as a tiger","tests/cat.jpg",ImageType.FILE_PATH) 

    def test_image2image(self):
        output = self.instance.invoke(self.input)
        self.assertIsNotNone(output)
        self.assertIsNotNone(output.content)
        for item in output.content:
            context_type = item.get("type") 
            if context_type != "text":
                self.assertIsNotNone(context_type,ImageType.PIL_IMAGE)
        print(output.content)

        
if __name__ == "__main__":
    unittest.main()
