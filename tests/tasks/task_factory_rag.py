import unittest
from langchain_core.messages import AIMessage, HumanMessage

# Assuming we import the TaskFactory and constants like TASK_LLM, TASK_EMBEDDING, etc.
from agi.tasks.task_factory import TaskFactory, TASK_LLM_WITH_RAG,TASK_RETRIEVER, TASK_AGENT


class TestTaskRagFactory(unittest.TestCase):
    pass

if __name__ == '__main__':
    unittest.main()
