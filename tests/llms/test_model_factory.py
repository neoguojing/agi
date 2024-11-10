import unittest
from agi.llms.model_factory import ModelFactory
from agi.llms.base import build_multi_modal_message,ImageType,AudioType
from langchain_core.messages import AIMessage, HumanMessage


class TestModelFactory(unittest.TestCase):

    def test_get_model(self):
        ollama_model = ModelFactory.get_model("ollama")
        resp = ollama_model.invoke("介绍下美国")
        self.assertIsNotNone(resp.content)
        self.assertEqual(len(ModelFactory._instances),1)
        ModelFactory.destroy("ollama")
        self.assertEqual(len(ModelFactory._instances),0)
        
        instance = ModelFactory.get_model("text2image")
        input = HumanMessage(content="a chinese leader") 
        resp = instance.invoke(input)
        self.assertIsNotNone(resp.content)
        self.assertEqual(len(ModelFactory._instances),1)
        ModelFactory.destroy("text2image")
        self.assertEqual(len(ModelFactory._instances),0)
        
        instance = ModelFactory.get_model("image2image")
        input = build_multi_modal_message("as a cat woman","tests/cat.jpg",ImageType.FILE_PATH) 
        resp = instance.invoke(input)
        self.assertIsNotNone(resp.content)
        self.assertEqual(len(ModelFactory._instances),1)
        ModelFactory.destroy("image2image")
        self.assertEqual(len(ModelFactory._instances),0)
        
        instance = ModelFactory.get_model("speech2text")
        input = self.input = build_multi_modal_message("","tests/1730604079.wav",AudioType.FILE_PATH)
        resp = instance.invoke(input)
        self.assertIsNotNone(resp.content)
        print(resp.content)
        self.assertEqual(len(ModelFactory._instances),1)
        ModelFactory.destroy("speech2text")
        self.assertEqual(len(ModelFactory._instances),0)
        
        instance = ModelFactory.get_model("text2speech")
        input = HumanMessage(content="岁的思考的加快速度为空军党委科技")
        resp = instance.invoke(input)
        self.assertIsNotNone(resp.content)
        print(resp.content)
        self.assertEqual(len(ModelFactory._instances),1)
        ModelFactory.destroy("text2speech")
        self.assertEqual(len(ModelFactory._instances),0)
        
        instance = ModelFactory.get_model("embedding")
        resp = instance.embed_query("岁的思考的加快速度为空军党委科技")
        self.assertIsNotNone(resp)
        print(resp)
        self.assertEqual(len(ModelFactory._instances),1)
        ModelFactory.destroy("embedding")
        self.assertEqual(len(ModelFactory._instances),0)
        
        
if __name__ == "__main__":
    unittest.main()
