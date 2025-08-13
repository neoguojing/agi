import jieba
import jieba.analyse
import jieba.posseg as pseg
import nltk
import spacy
from nltk.corpus import stopwords as nltk_stopwords
from nltk.tokenize import word_tokenize
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
        filtered_flags: Optional[List[str]] = ['x', 'uj', 'ul', 'e', 'y', 'r', 'd', 'p', 'm', 'q', 'u']  # 限定关键词词性，如 ["n", "v"]
    ):
        self.top_k = top_k
        self.filtered_flags = filtered_flags
        self.stopwords_en = set(nltk_stopwords.words('english'))
        self.stopwords_zh = set()
        if stop_words_path:
            jieba.analyse.set_stop_words(stop_words_path)
            self.stopwords_zh = self.load_stopwords(stop_words_path)
        if user_dict_path:
            jieba.load_userdict(user_dict_path)

        self.nlp_en = spacy.load("en_core_web_sm")
        # nltk.download('punkt_tab')

    def detect_language(self, text: str) -> str:
        """简单中文/英文判断"""
        zh_part = re.findall(r'[\u4e00-\u9fa5]+', text)
        en_part = re.findall(r'[a-zA-Z]+', text)
        
        return ''.join(zh_part),' '.join(en_part)
    
    def tokenize(self, text: str) -> List[str]:
        """仅分词（不带词性）"""
        zh, en = self.detect_language(text)
        zh_part = list(jieba.cut(zh)) if zh else []
        en_part = word_tokenize(en) if en else []
        return zh_part + en_part

    def tokenize_with_pos(self, text: str) -> List[Tuple[str, str]]:
        """分词 + 词性标注"""
        zh, en = self.detect_language(text)
        zh_part = [(word.word, word.flag) for word in pseg.cut(zh)] if zh else []
        en_part = [(token.text, token.pos_) for token in self.nlp_en(en)] if en else []
        
        return zh_part + en_part

    def load_stopwords(self, stop_words_path: Optional[str] = None) -> set:
        stopwords = set()
        if stop_words_path:
            with open(stop_words_path, 'r', encoding='utf-8') as f:
                stopwords.update(line.strip() for line in f if line.strip())
        return stopwords

    def remove_stopwords(self, text: str):
        """从分词结果中移除停用词"""
        zh, en = self.detect_language(text)
        zh_part = list(jieba.cut(zh)) if zh else []
        en_part = word_tokenize(en) if en else []
        tokens = []
        if zh_part:
            tokens.extend([w for w in zh_part if w not in self.stopwords_zh and w.strip()])        
        if en_part:
            tokens.extend([w for w in en_part if w.lower() not in self.stopwords_en and w.isalpha()])

        return ' '.join(tokens)

    def remove_stopwords_batch(self, texts: List[str]):
        """批量清洗文本"""
        with ThreadPoolExecutor() as executor:
            return list(executor.map(self.remove_stopwords, texts))
        
    def _clean_text(self, text: str) -> str:
        """文本清洗主流程（修复版）：
        1. 去除 HTML 标签
        2. unicode 标准化 (NFKC)
        3. 全角转半角
        4. 温和地去除无用字符，保留重要标点和符号
        5. 合并多余空白字符
        6. 去除首尾空格
        """

        # 1. 去除 HTML 标签
        text = BeautifulSoup(text, "html.parser").get_text(separator=' ')

        # 2. Unicode 标准化（兼容表情、异体字等）
        text = unicodedata.normalize("NFKC", text)

        # 3. 全角转半角
        def fullwidth_to_halfwidth(char: str) -> str:
            code = ord(char)
            if code == 0x3000:  # 全角空格
                return ' '
            elif 0xFF01 <= code <= 0xFF5E:  # 全角字符（除空格）
                return chr(code - 0xFEE0)
            return char
        text = ''.join(fullwidth_to_halfwidth(c) for c in text)

        # 4. 温和地去除特殊字符（修复核心）
        # 我们扩展了保留字符的范围，加入了各种括号、引号、以及常见的数学和特殊符号
        # 注意：这里将中英文标点符号统一在半角状态下处理
        # 保留：中文、英文、数字、空格
        # 保留：常用标点 .,!?;:
        # 保留：各种括号 ()[]{}
        # 保留：各种引号 "'`
        # 保留：常见数学与特殊符号 +-*/=<>@#$%&_`
        # 如果还需要保留其他字符（如日文、韩文），可以在此基础上继续添加
        # \u4e00-\u9fa5  (中文字符)
        # a-zA-Z0-9   (英文字母和数字)
        # \s           (空白字符)
        # .,!?;:'"`()\[\]{} (英文标点、引号、括号)
        # +-*/=<>@#$%&_  (数学及特殊符号)
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s.,!?;:\'"`()\[\]{}<>+\-*/=@#$%&_]', '', text, flags=re.UNICODE)


        # 5. 合并多余空白字符（包括空格、换行、制表符）
        text = re.sub(r'\s+', ' ', text)
        
        # 6. 去除首尾空格
        text = text.strip()
        return text


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
        zh, en = self.detect_language(text)  # 返回中文文本和英文文本
        print(zh)
        print(en)
        keywords = []
        if zh:
            if method == "tfidf":
                kw_zh = jieba.analyse.extract_tags(zh, topK=self.top_k, withWeight=True)
            else:
                kw_zh = jieba.analyse.textrank(zh, topK=self.top_k, withWeight=True)
                # 若 textrank 返回为空，fallback 到 tfidf
                if not kw_zh:
                    kw_zh = jieba.analyse.extract_tags(zh, topK=self.top_k, withWeight=True)

            if self.filtered_flags:
                pos_dict = {word.word: word.flag for word in pseg.cut(zh)}
                kw_zh = [(word, weight) for word, weight in kw_zh if pos_dict.get(word, '') not in self.filtered_flags]

            keywords.extend(kw_zh)

        if en:
            tokens_en = word_tokenize(en,language="english")
            # 过滤停用词和非字母词
            filtered_en = [w.lower() for w in tokens_en if w.isalpha() and w.lower() not in getattr(self, 'stopwords_en', set())]
            freq = {}
            for w in filtered_en:
                freq[w] = freq.get(w, 0) + 1
            sorted_en = sorted(freq.items(), key=lambda x: x[1], reverse=True)
            # 取 top_k，频率转 float
            kw_en = [(word, float(count)) for word, count in sorted_en[:self.top_k]]

            keywords.extend(kw_en)

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
