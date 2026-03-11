import unittest
from typing import List
from langchain_core.documents import Document
# 假设你的代码保存在 splitter_module.py 中
from agi.rag.spliter import CustomDocumentSplitter

class TestCustomDocumentSplitter(unittest.TestCase):
    def setUp(self):
        """初始化切分器，设置较小的 chunk_size 以便触发切分"""
        self.splitter = CustomDocumentSplitter(chunk_size=100, chunk_overlap=20)

    def test_markdown_logic(self):
        """测试 Markdown 标题识别与元数据注入"""
        md_text = """# 核心架构
这是第一章的内容。
## 数据流转
这是第二章的详细细节，内容比较长，旨在测试是否能够正确保留 H1 和 H2 的元数据。"""
        
        docs = self.splitter.split_text(md_text, file_type="md", file_name="arch.md")
        
        # 验证是否生成了多个文档
        self.assertGreater(len(docs), 1)
        # 验证元数据中是否包含标题路径
        self.assertEqual(docs[0].metadata["H1"], "核心架构")
        self.assertIn("breadcrumb", docs[-1].metadata)
        self.assertTrue("核心架构 > 数据流转" in docs[-1].metadata["breadcrumb"])
        # 验证序号
        self.assertEqual(docs[0].metadata["chunk_index"], 0)

    def test_code_splitting(self):
        """测试 Python 代码切分是否保留了 def 结构的完整性趋势"""
        py_code = """
import os

def function_one():
    print("This is a long function body to trigger splitting...")
    return 1 + 1

def function_two():
    print("Another function starts here.")
"""
        docs = self.splitter.split_text(py_code, file_type="py")
        
        # 验证代码切分器是否工作
        self.assertTrue(any("function_one" in d.page_content for d in docs))
        self.assertEqual(docs[0].metadata["doc_id"], docs[1].metadata["doc_id"])

    def test_auto_detection(self):
        """测试自动类型识别功能"""
        # 测试 HTML 自动识别
        html_content = "<html><body><h1>网页标题</h1><p>正文内容...</p></body></html>"
        docs = self.splitter.split_text(html_content)
        self.assertEqual(docs[0].metadata["H1"], "网页标题")

        # 测试 JSON 自动识别（虽然目前 JSON 走默认递归，但应能识别）
        json_content = '{"key": "value", "description": "some long text"}'
        docs = self.splitter.split_text(json_content)
        self.assertIn("doc_id", docs[0].metadata)

    def test_content_integrity(self):
        """测试内容指纹与唯一性 ID"""
        text = "这是一段完全相同的测试文本。"
        docs1 = self.splitter.split_text(text)
        docs2 = self.splitter.split_text(text)
        
        # 内容相同，hash 应该相同
        self.assertEqual(docs1[0].metadata["content_hash"], docs2[0].metadata["content_hash"])
        # 但 doc_id 应该不同（因为每次 split_text 被视为新文档录入）
        self.assertNotEqual(docs1[0].metadata["doc_id"], docs2[0].metadata["doc_id"])

if __name__ == "__main__":
    unittest.main()