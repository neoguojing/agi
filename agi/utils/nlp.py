import jieba
import jieba.analyse
import jieba.posseg as pseg
import asyncio
from typing import List, Literal, Tuple, Optional


class TextProcessor:
    def __init__(
        self,
        stop_words_path: Optional[str] = None,
        user_dict_path: Optional[str] = None,
        top_k: int = 5,
        allowed_flags: Optional[List[str]] = ['n', 'v', 'a', 'vn', 'nr', 'ns', 'nt', 'nz']  # 限定关键词词性，如 ["n", "v"]
    ):
        self.top_k = top_k
        self.allowed_flags = allowed_flags
        if stop_words_path:
            jieba.analyse.set_stop_words(stop_words_path)
        if user_dict_path:
            jieba.load_userdict(user_dict_path)

    def tokenize(self, text: str) -> List[str]:
        """仅分词（不带词性）"""
        return list(jieba.cut(text))

    def tokenize_with_pos(self, text: str) -> List[Tuple[str, str]]:
        """分词 + 词性标注"""
        return [(word.word, word.flag) for word in pseg.cut(text)]

    def extract_keywords(
        self,
        text: str,
        method: Literal["tfidf", "textrank"] = "textrank"
    ) -> List[Tuple[str, float]]:
        """提取关键词，并根据词性过滤（可选）"""
        if method == "tfidf":
            keywords = jieba.analyse.extract_tags(text, topK=self.top_k, withWeight=True)
        elif method == "textrank":
            keywords = jieba.analyse.textrank(text, topK=self.top_k, withWeight=True, withFlag=True)
        else:
            raise ValueError("method must be 'tfidf' or 'textrank'")

        # 如果设置了词性过滤（仅 textrank 支持 withFlag）
        if self.allowed_flags and method == "textrank":
            return [(w.word, w.weight) for w in keywords if w.flag in self.allowed_flags]
        else:
            return keywords

    async def process_text(
        self,
        text: str,
        method: Literal["tfidf", "textrank"] = "textrank"
    ) -> dict:
        keywords = self.extract_keywords(text, method)
        return keywords

    async def batch_process(
        self,
        texts: List[str],
        method: Literal["tfidf", "textrank"] = "textrank"
    ) -> List[dict]:
        tasks = [self.process_text(t, method) for t in texts]
        return await asyncio.gather(*tasks)


async def test():
    texts = [
        "苹果公司推出了新款iPhone，引发了市场热议。",
        "人工智能的发展正在改变世界。"
    ]
    processor = TextProcessor(top_k=5)
    results = await processor.batch_process(texts, method="textrank")

    for item in results:
        print(f"\n原文: {item['text']}")
        print("关键词:")
        for kw, weight in item["keywords"]:
            print(f"  - {kw}: {weight:.4f}")

asyncio.run(test())