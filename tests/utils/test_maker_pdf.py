import os
import pytest
from marker.models import create_model_dict
from marker.config.parser import ConfigParser
from marker.converters.pdf import PdfConverter
from marker.output import text_from_rendered

SAMPLE_PDF = "tests/2025CHIANSAE.pdf"  # 包含标题、段落、表格、图片等结构

@pytest.fixture(scope="module")
def model_dict():
    import pdb;pdb.set_trace()
    md = create_model_dict()
    assert isinstance(md, dict)
    # 检查至少包含关键模型 artifact key
    assert any(k in md for k in ["layout", "text_recognition", "table", "texify"]), \
        "Expected layout, OCR, table, texify models loaded"
    return md

@pytest.fixture(scope="module")
def converter(model_dict):
    import pdb;pdb.set_trace()

    cfg = ConfigParser({"output_format": "markdown"})
    return PdfConverter(
        config=cfg.generate_config_dict(),
        artifact_dict=model_dict,
        processor_list=cfg.get_processors(),
        renderer=cfg.get_renderer(),
        llm_service=cfg.get_llm_service()
    )

def test_create_model_dict_loads_models(model_dict):
    # 检查模型加载响应性能合理
    assert model_dict, "Model dict should not be empty"

def test_pdf_conversion_and_output(converter, tmp_path):
    import pdb;pdb.set_trace()

    rendered = converter(SAMPLE_PDF)
    text, images, metadata = text_from_rendered(rendered)
    # 确保获取到非空文本
    assert text.strip(), "Extracted markdown text should be non-empty"
    # 若 PDF 包含图片
    assert isinstance(images, dict)
    # 检查文本中含标题标记或表格标示
    assert "#" in text or "##" in text, "Markdown output should contain headings"
    assert "|" in text or "-" in text, "Markdown output should contain table or list structures"

    md_file = tmp_path / "out.md"
    md_file.write_text(text, encoding="utf-8")
    assert md_file.exists(), "Should write a markdown file"

def test_conversion_performance(model_dict):
    import time
    start = time.time()
    _ = create_model_dict()  # 测试加载时间
    duration = time.time() - start
    assert duration < 180, f"Model loading took too long: {duration:.2f}s"
