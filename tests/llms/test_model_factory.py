import unittest
from agi.llms.model_factory import ModelFactory
from langchain_core.messages import AIMessage, HumanMessage


class TestModelFactory(unittest.TestCase):

    def test_get_model(self):
     
        instance = ModelFactory.get_model("text2image")
        input = HumanMessage(content="a chinese leader") 
        resp = instance.invoke(input)
        self.assertIsNotNone(resp.content)
        self.assertEqual(len(ModelFactory._instances),1)
        
        instance = ModelFactory.get_model("image2image")
        input = HumanMessage(content=[
                {"type":"text","text":"as a cat woman"},
                {"type":"image","image":"tests/cat.jpg"}
            ],
        ) 
        resp = instance.invoke(input)
        self.assertIsNotNone(resp.content)
        self.assertEqual(len(ModelFactory._instances),2)

        instance = ModelFactory.get_model("speech2text")
        input = self.input = HumanMessage(content=[
                {"type":"audio","audio":"tests/zh-cn-sample.wav"}
            ]
        ) 
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
