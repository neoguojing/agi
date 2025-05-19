import unittest
from agi.utils.scrape import WebScraper
from agi.tasks.task_factory import TaskFactory

class TestScrape(unittest.TestCase):
    def setUp(self):
        self.llm_scrape = WebScraper(llm=TaskFactory.get_llm_with_output_format(debug=True))
        self.scrape = WebScraper()

    def test_scrape(self):
        # 头条
        
        # ret = self.llm_scrape.invoke("https://www.toutiao.com/w/1831910043724936/?log_from=593d5b4b2a1cb_1747288992329")
        # print(ret)
        # 微信
        # ret = self.scrape.invoke("https://mp.weixin.qq.com/s/9vle6WnTiNxmu6s4hz2TVQ?poc_token=HI-NKGij_o0ij2iSTR3S9n0ASxGeR7nKpUsgzOMV")
        # print(ret)
        # ret = self.llm_scrape.invoke("https://mp.weixin.qq.com/s/9vle6WnTiNxmu6s4hz2TVQ?poc_token=HI-NKGij_o0ij2iSTR3S9n0ASxGeR7nKpUsgzOMV")
        # print(ret)
        # 百度
        ret = self.scrape.invoke("https://baijiahao.baidu.com/s?id=1832336275989774930")
        print(ret)
        # ret = self.llm_scrape.invoke("https://baijiahao.baidu.com/s?id=1832195164029857172")
        # print(ret)
        # 知乎
        # ret = self.llm_scrape.invoke("https://zhuanlan.zhihu.com/p/19244164610")
        # print(ret)
        
        # ret = self.scrape.invoke("https://www.toutiao.com/w/1831910043724936/?log_from=593d5b4b2a1cb_1747288992329")
        # print(ret)
        # ret = self.scrape.invoke("https://zhuanlan.zhihu.com/p/19244164610")
        # print("************",ret)
        

if __name__ == "__main__":
    unittest.main()
