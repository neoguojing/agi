import pytest
import asyncio
from typing import List
from agi.utils.nlp import TextProcessor

sample_text = "小明毕业于清华大学，后来在谷歌公司担任高级工程师。"

@pytest.fixture
def processor():
    return TextProcessor(top_k=3)

def test_tokenize(processor):
    tokens = processor.tokenize(sample_text)
    print(tokens)
    assert isinstance(tokens, list)
    assert any(word in tokens for word in ["清华大学", "谷歌", "工程师"])

def test_tokenize_with_pos(processor):
    tokens = processor.tokenize_with_pos(sample_text)
    print(tokens)
    assert isinstance(tokens, list)
    assert all(isinstance(t, tuple) and len(t) == 2 for t in tokens)
    assert any(word == "清华大学" and flag.startswith("n") for word, flag in tokens)

def test_extract_keywords_textrank(processor):
    keywords = processor.extract_keywords(sample_text, method="textrank")
    print(f"test_extract_keywords_textrank:{keywords}")
    assert isinstance(keywords, list)
    assert all(isinstance(k, tuple) and isinstance(k[0], str) and isinstance(k[1], float) for k in keywords)
    assert len(keywords) <= processor.top_k

def test_extract_keywords_tfidf(processor):
    keywords = processor.extract_keywords(sample_text, method="tfidf")
    print(f"test_extract_keywords_tfidf:{keywords}")
    assert isinstance(keywords, list)
    assert len(keywords) <= processor.top_k


@pytest.mark.asyncio
async def test_batch_process(processor):
    texts = [sample_text, "他在北京大学获得硕士学位。"]
    results = await processor.abatch_process(texts, method="textrank")
    print(results)
    assert isinstance(results, list)
    assert len(results) == len(texts)
    assert all(isinstance(r, list) for r in results)

    processed_results = await processor.abatch_process(["清华大学投档线 /no_think"], method="textrank")
    print(processed_results)
    processed_results = await processor.abatch_process(["Ablation Studies for Multi-Token Prediction"], method="textrank")
    print(processed_results)
    processed_results = await processor.abatch_process(["Ablation Studies for Multi-Token Prediction 讲了什么？"], method="textrank")
    print(processed_results)
    processed_results = await processor.abatch_process(["这段文字讲了什么？撒大声地撒大大所多撒大所大叔大婶多撒多撒多"], method="textrank")
    print(processed_results)
    
