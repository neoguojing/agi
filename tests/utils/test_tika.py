import os
import pytest
from pathlib import Path
from agi.utils.tika import TikaExtractor  # 引用上面写的类

# 准备测试文件
TEST_FILE = "tests/tot.pdf"

@pytest.mark.parametrize("file_path", [TEST_FILE])
def test_extract_metadata(file_path):
    """
    测试从真实 Tika Server 获取 Metadata
    """

    extractor = TikaExtractor("http://localhost:9998")
    meta = extractor.extract_metadata(str(file_path))

    print(meta)
    # 确认返回是 dict
    assert isinstance(meta, dict)

    # 至少应该包含 Content-Type
    assert "Content-Type" in meta
    assert meta["Content-Type"] == "application/pdf"

    # 确认 Page-Count 是数字字符串
    if meta.get("Page-Count") is not None:
        assert meta["Page-Count"].isdigit()

    # 确认 title/Author 等字段可取（可能为 None）
    assert "title" in meta
    assert "Author" in meta

@pytest.mark.parametrize("output,accept", [
    ("text", None),            # 默认纯文本
    ("main", None),            # 主体文本
    ("html", None),            # HTML 格式
    ("text", "text/plain"),    # 显式 text/plain
])
def test_extract_text(output, accept):
    """
    真实测试 Tika Server 的 extract_text 功能
    """
    extractor = TikaExtractor("http://localhost:9998")
    text = extractor.extract_text(TEST_FILE, output=output, accept=accept)

    print(text)
    # 返回必须是 str
    assert isinstance(text, str)
    assert len(text) > 0

    # 如果是 text/main 模式，结果应该比纯 text 短（一般去掉 header/footer）
    if output == "main":
        full_text = extractor.extract_text(TEST_FILE, output="text")
        assert len(text) <= len(full_text)
