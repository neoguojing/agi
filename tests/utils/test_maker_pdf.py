import os
import tempfile
import pytest
from marker.convert import convert_single_pdf
from marker.output import save_markdown
from marker.models import load_all_models

PDF_SAMPLE = "tests/sample.pdf"  # 你需要准备一个包含段落、列表、表格、图片等结构的测试 PDF

@pytest.fixture(scope="module")
def models():
    return load_all_models()

def test_marker_pdf_to_markdown(models, tmp_path):
    import pdb;pdb.set_trace()
    # 执行转换
    full_text, images, metadata = convert_single_pdf(PDF_SAMPLE, models)
    assert full_text, "Expected non-empty Markdown text"
    
    # 保存输出
    out_dir = tmp_path / "marker_out"
    out = save_markdown(str(out_dir), os.path.basename(PDF_SAMPLE), full_text, images, metadata)
    md_file = out / f"{os.path.splitext(os.path.basename(PDF_SAMPLE))[0]}.md"
    assert md_file.exists(), "Markdown file should be written"

    md_content = md_file.read_text(encoding="utf-8")
    # 核心检查点：
    assert "#" in md_content or "##" in md_content, "Expect at least a heading"
    assert "|" in md_content or "-" in md_content, "Expect table or list structure"
    # 若 sample PDF 中含图像，这里应包含如下标记
    assert "![image]" in md_content or "images/" in os.listdir(out_dir), "Expect image references"

    # 检查是否清除了页眉页脚噪音
    assert "Page" not in md_content, "Should not include page numbers or headers"

def test_performance_speed(models):
    # 可选：检测转换速度不超过阈值
    import time
    start = time.time()
    convert_single_pdf(PDF_SAMPLE, models)
    duration = time.time() - start
    assert duration < 60, f"Conversion took too long: {duration:.2f}s"
