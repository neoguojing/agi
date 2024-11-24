import unittest
from agi.llms.model_factory import ModelFactory
from agi.llms.base import build_multi_modal_message,ImageType,AudioType
from langchain_core.messages import AIMessage, HumanMessage


class TestModelFactory(unittest.TestCase):

    def test_get_model(self):
     
        instance = ModelFactory.get_model("text2image")
        input = HumanMessage(content="a chinese leader") 
        resp = instance.invoke(input)
        self.assertIsNotNone(resp.content)
        self.assertEqual(len(ModelFactory._instances),1)
        
        instance = ModelFactory.get_model("image2image")
        input = build_multi_modal_message("as a cat woman","tests/cat.jpg") 
        resp = instance.invoke(input)
        self.assertIsNotNone(resp.content)
        self.assertEqual(len(ModelFactory._instances),2)

        instance = ModelFactory.get_model("speech2text")
        input = self.input = build_multi_modal_message("","tests/1730604079.wav")
        resp = instance.invoke(input)
        self.assertIsNotNone(resp.content)
        print(resp.content)
        self.assertEqual(len(ModelFactory._instances),2)

        
        instance = ModelFactory.get_model("text2speech")
        input = HumanMessage(content="岁的思考的加快速度为空军党委科技")
        resp = instance.invoke(input)
        self.assertIsNotNone(resp.content)
        print(resp.content)
        self.assertEqual(len(ModelFactory._instances),2)
        
        
if __name__ == "__main__":
    unittest.main()
