import unittest
from langchain_core.messages import AIMessage, HumanMessage

# Assuming we import the TaskFactory and constants like TASK_LLM, TASK_EMBEDDING, etc.
from agi.tasks.task_factory import TaskFactory, TASK_LLM_WITH_RAG,TASK_RETRIEVER,TASK_DOC_DB
from agi.tasks.retriever import KnowledgeManager
from agi.config import CACHE_DIR
class TestTaskRagFactory(unittest.TestCase):
    
    def setUp(self):
        self.kmanager = TaskFactory.create_task(TASK_DOC_DB)
    def test_add_doc(self):
        collect_name,file_type = self.kmanager.store("test","./test/test.pdf")
        print(collect_name,file_type)
    
    def test_web_search(self):
        pass
    
    def test_doc_search(self):
        pass

if __name__ == '__main__':
    unittest.main()
