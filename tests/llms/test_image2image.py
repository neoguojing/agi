import unittest
from agi.llms.base import Media
from agi.llms.image2image import Image2Image
from langchain_core.messages import AIMessage,HumanMessage
from PIL import Image as PIL_Image
import logging
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
class TestImage2Image(unittest.TestCase):

    def setUp(self):
        """Set up the test environment for Image2Image."""
        self.instance = Image2Image()  # Initialize the Image2Image instance
        
        # Prepare input image
        self.image_path = "tests/cat.jpg"  # Path to the test image
        self.img = Media.from_data(self.image_path, media_type="image")
        self.assertIsNotNone(self.img, "Failed to load image.")

        # Prepare the multi-modal message with text and image
        self.input_message = HumanMessage(content=[
            {"type":"text","text":"as a tiger"},
            {"type":"image","image":self.image_path}
        ])

    def test_image2image_invoke(self):
        """Test the invocation of the Image2Image model."""
        output = self.instance.invoke(self.input_message)
        
        # Assert the output is not None
        self.assertIsNotNone(output, "Output should not be None.")
        
        # Assert that output has content
        self.assertIsNotNone(output.content, "Output content should not be None.")
        
        # Check the content types
        for item in output.content:
            context_type = item.get("type")
            if context_type == "text":
                self.assertIsInstance(item.get("text"), str, "Text content should be a string.")
            elif context_type == "image":
                self.assertIsInstance(item.get("image"), str, "PIL image content should be an instance of Image.")
            else:
                self.fail(f"Unexpected content type: {context_type}")

        print(f"Test Output: {output.content}")

    def tearDown(self):
        """Clean up after tests."""
        # You can include any clean-up logic here if necessary.
        pass


if __name__ == "__main__":
    unittest.main()
