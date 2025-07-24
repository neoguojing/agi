import jieba
import jieba.analyse
import jieba.posseg as pseg
import asyncio
from typing import List, Literal, Tuple, Optional
from agi.config import STOP_WORDS_PATH
from concurrent.futures import ThreadPoolExecutor
import unicodedata
from bs4 import BeautifulSoup
import re
class TextProcessor:
    def __init__(
        self,
        stop_words_path: Optional[str] = STOP_WORDS_PATH,
        user_dict_path: Optional[str] = None,
        top_k: int = 5,
        allowed_flags: Optional[List[str]] = ['n', 'v', 'a', 'vn', 'nr', 'ns', 'nt', 'nz']  # 限定关键词词性，如 ["n", "v"]
    ):
        self.top_k = top_k
        self.allowed_flags = allowed_flags
        self.stopwords = None
        if stop_words_path:
            jieba.analyse.set_stop_words(stop_words_path)
            self.stopwords = self.load_stopwords(stop_words_path)
        if user_dict_path:
            jieba.load_userdict(user_dict_path)

    def tokenize(self, text: str) -> List[str]:
        """仅分词（不带词性）"""
        return list(jieba.cut(text))

    def tokenize_with_pos(self, text: str) -> List[Tuple[str, str]]:
        """分词 + 词性标注"""
        return [(word.word, word.flag) for word in pseg.cut(text)]

    def load_stopwords(self, stop_words_path: Optional[str] = None) -> set:
        stopwords = set()
        if stop_words_path:
            with open(stop_words_path, 'r', encoding='utf-8') as f:
                stopwords.update(line.strip() for line in f if line.strip())
        return stopwords


    def _remove_stopwords(self, text: str) -> List[str]:
        """从分词结果中移除停用词"""
        words = self.tokenize(text)
        return [w for w in words if w not in self.stopwords and w.strip()]

    def remove_stopwords_batch(self, texts: List[str]) -> List[str]:
        """批量清洗文本"""
        with ThreadPoolExecutor() as executor:
            return list(executor.map(self._remove_stopwords, texts))
        
    def _clean_text(self, text: str) -> str:
        """文本清洗主流程：
        1. 去除 HTML 标签
        2. unicode 标准化
        3. 全角转半角
        4. 去除特殊符号
        5. 合并多空白字符
        6. 去除首尾空格
        """

        # 去除 HTML 标签（如 <p>, <div>）
        text = BeautifulSoup(text, "html.parser").get_text()

        # Unicode 标准化（兼容表情、异体字等）
        text = unicodedata.normalize("NFKC", text)

        # 全角转半角（如：中文输入法下的符号）
        def fullwidth_to_halfwidth(char):
            code = ord(char)
            if code == 0x3000:
                return ' '
            elif 0xFF01 <= code <= 0xFF5E:
                return chr(code - 0xFEE0)
            return char

        text = ''.join(fullwidth_to_halfwidth(c) for c in text)

        # 去除特殊字符（保留中英文、数字和常用标点）
        text = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9\s.,!?;:，。！？；：]", '', text)

        # 合并多空格为一个空格
        text = re.sub(r'\s+', ' ', text)

        # 去除首尾空格
        return text.strip()


    def clean_batch(self, texts: List[str]) -> List[str]:
        """批量清洗文本"""
        with ThreadPoolExecutor() as executor:
            return list(executor.map(self._clean_text, texts))

    def extract_keywords(
        self,
        text: str,
        method: Literal["tfidf", "textrank"] = "textrank"
    ) -> List[Tuple[str, float]]:
        """提取关键词，并根据词性过滤（可选）"""
        if method == "tfidf":
            keywords = jieba.analyse.extract_tags(text, topK=self.top_k, withWeight=True)
        elif method == "textrank":
            keywords = jieba.analyse.textrank(text, topK=self.top_k, withWeight=True)
        else:
            raise ValueError("method must be 'tfidf' or 'textrank'")

        # 如果设置了词性过滤（仅 textrank 支持 withFlag）
        if self.allowed_flags:
            # 构造词性字典
            pos_dict = {word.word: word.flag for word in pseg.cut(text)}
            # 筛选
            keywords = [(word, weight) for word, weight in keywords if pos_dict.get(word, '') in self.allowed_flags]
        
        return keywords

    def batch_process(
        self,
        texts: List[str],
        method: Literal["tfidf", "textrank"] = "textrank"
    ):
        results = []
        with ThreadPoolExecutor() as pool:
            futures = [pool.submit(self.extract_keywords, text, method) for text in texts]
            for future in futures:
                results.append(future.result())
        return results

    async def abatch_process(
        self,
        texts: List[str],
        method: Literal["tfidf", "textrank"] = "textrank"
    ) -> List[dict]:
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor() as pool:
            tasks = [
                loop.run_in_executor(pool, self.extract_keywords, text, method)
                for text in texts
            ]
            results = await asyncio.gather(*tasks)
        return results
