import unittest
from langchain_core.messages import AIMessage, HumanMessage,ToolMessage 
# Assuming we import the TaskFactory and constants like TASK_LLM, TASK_EMBEDDING, etc.
from agi.tasks.task_factory import TaskFactory,TASK_RAG,TASK_WEB_SEARCH,TASK_DOC_CHAT
from agi.tasks.define import State
from agi.config import CACHE_DIR
import asyncio
from agi.config import log

class TestTaskRagFactory(unittest.TestCase):
    
    def setUp(self):
        self.kmanager = TaskFactory.get_knowledge_manager()
        self.rag = TaskFactory.create_task(TASK_RAG)
        self.web = TaskFactory.create_task(TASK_WEB_SEARCH)
        self.doc_stuff = TaskFactory.create_task(TASK_DOC_CHAT)
        print(self.kmanager.list_collections())
    def test_add_doc(self):
        param = {"filename" : "test.pdf"}
        collect_name,know_type,raw_docs = asyncio.run(self.kmanager.store("test","./tests/test.pdf",**param))
        self.assertEqual(collect_name,"test")
        docs = self.kmanager.list_documets("test")
        self.assertEqual(len(docs),2)
    
    def test_web_search(self):
        urls,raw_results,raw_docs = asyncio.run(self.kmanager.web_search("上海未来一周天气如何？"))
        self.assertNotEqual(len(raw_results),0)
        self.assertNotEqual(len(raw_docs),0)
        self.assertNotEqual(len(urls),0)

    def test_doc_search(self):
        docs = asyncio.run(self.kmanager.query_doc("test","完善动平衡计算模块"))
        self.assertEqual(docs[0].metadata["page"],0)
        self.assertEqual(docs[0].metadata["source"],"./tests/test.pdf")
        self.assertIn("平衡计算模块",docs[0].page_content)

    def test_rag(self):
        config={"configurable": {"user_id": "default_tenant", "conversation_id": "2"}}
        input = State(
            messages=[HumanMessage(content="NTP3000Plus")],
            collection_names = ["test"]
            
        )
        ret = self.rag.invoke(input,config=config)
        print(ret)
        self.assertIsNotNone(ret)
        self.assertIsInstance(ret["messages"],list)
        self.assertIsInstance(ret['docs'],list)
        self.assertIsNotNone(ret['docs'],list)
        self.assertIsNotNone(ret["messages"][-1].content)
        self.assertIsInstance(ret["citations"],list)
        self.assertIsNotNone(ret['citations'])
        
        ret = self.doc_stuff.invoke(ret,config=config)
        self.assertIsInstance(ret,AIMessage)
        self.assertIsNotNone(ret.content)

    def test_web_search_chat(self):
        config={"configurable": {"user_id": "default_tenant", "conversation_id": "3"}}
        input = State(
            messages=[HumanMessage(content="今天的科技新闻")],
        )
        ret = self.web.invoke(input,config=config)
        print(ret)
        self.assertIsInstance(ret["messages"],list)
        self.assertIsInstance(ret['docs'],list)
        self.assertIsNotNone(ret['docs'],list)
        self.assertIsNotNone(ret["messages"][-1].content)
        self.assertIsInstance(ret["citations"],list)
        self.assertIsNotNone(ret['citations'])
        
        ret = self.doc_stuff.invoke(ret,config=config)
        self.assertIsInstance(ret,AIMessage)
        self.assertIsNotNone(ret.content)
        
    def test_web_loader(self):
        loader = self.kmanager.get_web_loader(["https://www.thepaper.cn/newsDetail_forward_29557801"])
        docs = loader.load()
        print(docs)
        self.assertNotEqual(len(docs),0)
        self.assertNotEqual(docs[0].page_content,"")
        loader = self.kmanager.get_web_loader(["https://news.qq.com/rain/a/20241207A071BP00"])
        docs = loader.load()
        print(docs)
        self.assertNotEqual(len(docs),0)
        self.assertNotEqual(docs[0].page_content,"")
        loader = self.kmanager.get_web_loader(["https://world.huanqiu.com/article/4MSVE1DOKy3"])
        docs = loader.load()
        print(docs)
        self.assertNotEqual(len(docs),0)
        self.assertNotEqual(docs[0].page_content,"")
        loader = self.kmanager.get_web_loader(["https://www.toutiao.com/article/7497472910076953139/?log_from=5372de4ab4e0b_1745828583840"])
        docs = loader.load()
        print(docs)
        self.assertNotEqual(len(docs),0)
        self.assertNotEqual(docs[0].page_content,"")
        loader = self.kmanager.get_web_loader(["https://mp.weixin.qq.com/s?__biz=MzI3MTA0MTk1MA==&mid=2652590095&idx=1&sn=a4d45748db6b4012139a4e7f9a158ea5&chksm=f0b9a5be731611e394f14ee44ad674aff0734ee0d0bdd85cb8072059246780b7894c1adeff4c&mpshare=1&scene=1&srcid=0428fR1goe3aXVjXnH0jEu00&sharer_shareinfo=cb902956c8f31588deefb84b5bfe26b7&sharer_shareinfo_first=cb902956c8f31588deefb84b5bfe26b7#rd"])
        docs = loader.load()
        print(docs)
        self.assertNotEqual(len(docs),0)
        self.assertNotEqual(docs[0].page_content,"")
if __name__ == '__main__':
    unittest.main()
