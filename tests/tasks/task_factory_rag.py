import unittest
from langchain_core.messages import AIMessage, HumanMessage

# Assuming we import the TaskFactory and constants like TASK_LLM, TASK_EMBEDDING, etc.
from agi.tasks.task_factory import TaskFactory, TASK_LLM_WITH_RAG,TASK_RETRIEVER,TASK_DOC_DB
from agi.tasks.retriever import KnowledgeManager
from agi.config import CACHE_DIR
class TestTaskRagFactory(unittest.TestCase):
    
    def setUp(self):
        self.kmanager = TaskFactory.create_task(TASK_DOC_DB)
    # def test_add_doc(self):
    #     param = {"filename" : "test.pdf"}
    #     collect_name,know_type = self.kmanager.store("test","./tests/test.pdf",**param)
    #     self.assertEqual(collect_name,"test")
    #     docs = self.kmanager.list_documets("test")
    #     self.assertEqual(len(docs),2)
        
    def test_web_search(self):
        pass
    
    def test_doc_search(self):
        docs = self.kmanager.query_doc("test","完善动平衡计算模块")
        self.assertEqual(docs[0].metadata["page"],0)
        self.assertEqual(docs[0].metadata["source"],"./tests/test.pdf")

if __name__ == '__main__':
    unittest.main()
