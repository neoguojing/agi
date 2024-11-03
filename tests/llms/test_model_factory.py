import unittest
from agi.llms.model_factory import ModelFactory

class TestModelFactory(unittest.TestCase):

    def test_get_model(self):
        ollama_model = ModelFactory.get_model("ollama")
        resp = ollama_model.invoke("介绍下美国")
        print(type(resp))
        
    # def test_release_model(self):
    #     self.assertIsNotNone(output)
    #     self.assertIsNotNone(output.image)
    #     self.assertIsNotNone(output.image.pil_image)
    #     self.assertIsNotNone(output.content)
    #     print(output.content)

        
if __name__ == "__main__":
    unittest.main()
