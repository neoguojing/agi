import unittest
from langchain_core.messages import AIMessage, HumanMessage

# Assuming we import the TaskFactory and constants like TASK_LLM, TASK_EMBEDDING, etc.
from agi.tasks.task_factory import TaskFactory, TASK_LLM_WITH_RAG,TASK_RETRIEVER,TASK_DOC_DB
from agi.tasks.retriever import KnowledgeManager,SourceType
from agi.tasks.llm_app import create_chat_with_rag
from agi.config import CACHE_DIR
class TestTaskRagFactory(unittest.TestCase):
    
    def setUp(self):
        self.kmanager = TaskFactory.create_task(TASK_DOC_DB)
        self.retreiver = TaskFactory.create_task(TASK_RETRIEVER)
        self.rag = TaskFactory.create_task(TASK_LLM_WITH_RAG)
    # def test_add_doc(self):
    #     param = {"filename" : "test.pdf"}
    #     collect_name,know_type,raw_docs = self.kmanager.store("test","./tests/test.pdf",**param)
    #     self.assertEqual(collect_name,"test")
    #     docs = self.kmanager.list_documets("test")
    #     self.assertEqual(len(docs),2)
    
    # def test_web_loader(self):
    #     loader = self.kmanager.get_web_loader(["https://news.qq.com/rain/a/20241207A071BP00","https://www.thepaper.cn/newsDetail_forward_29557801"])
    #     docs = loader.load()
    #     self.assertEqual(len(docs),2)
    
    # def test_web_search(self):
    #     collect_name,know_type,raw_results,raw_docs = self.kmanager.web_search("上海未来一周天气如何？")
    #     self.assertNotEqual(len(raw_results),0)
    #     self.assertNotEqual(len(raw_docs),0)
    #     docs = self.kmanager.query_doc("web","上海未来一周天气如何？")
    #     self.assertNotEqual(len(docs),0)

    # def test_doc_search(self):
    #     docs = self.kmanager.query_doc("test","完善动平衡计算模块")
    #     self.assertEqual(docs[0].metadata["page"],0)
    #     self.assertEqual(docs[0].metadata["source"],"./tests/test.pdf")
    #     self.assertIn("平衡计算模块",docs[0].page_content)
    
    # def test_retreiver(self):
    #     ret = self.retreiver.invoke("上海未来一周天气如何？")
    #     self.assertIsNotNone(ret)
    #     self.assertNotEqual(len(ret),0)
    
    def test_rag(self):
        config={"configurable": {"user_id": "test", "conversation_id": "1"}}
        ret = self.rag.invoke({"text":"上海未来一周天气如何？","language":"chinese"},config=config)
        self.assertIsNotNone(ret)
        self.assertIsInstance(ret['chat_history'],list)
        self.assertIsInstance(ret['context'],list)
        self.assertIsNotNone(ret['answer'])
        print(ret['citations'])

if __name__ == '__main__':
    unittest.main()
