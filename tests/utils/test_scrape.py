import unittest
from agi.utils.scrape import WebScraper

class TestScrape(unittest.TestCase):
    def setUp(self):
        self.play_scrape = WebScraper()
        
    def test_scrape(self):
        # 头条
        ret = self.play_scrape.invoke("https://www.toutiao.com/w/1831910043724936/?log_from=593d5b4b2a1cb_1747288992329")
        print(f"头条:{ret}")
        ret = self.play_scrape.invoke("https://mp.weixin.qq.com/s/9vle6WnTiNxmu6s4hz2TVQ?poc_token=HI-NKGij_o0ij2iSTR3S9n0ASxGeR7nKpUsgzOMV")
        print(f"weixin:{ret}")
        ret = self.play_scrape.invoke("https://baijiahao.baidu.com/s?id=1832195164029857172")
        print(f"baidu:{ret}")
        ret = self.play_scrape.invoke("https://zhuanlan.zhihu.com/p/19244164610")
        print(f"知乎:{ret}")
        

if __name__ == "__main__":
    unittest.main()
