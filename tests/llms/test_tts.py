import unittest

from langchain_core.messages import AIMessage, HumanMessage

from agi.config import log

class TestTextToSpeech(unittest.TestCase):

    def setUp(self):
        from agi.llms.tts import TextToSpeech
        self.instance = TextToSpeech()
        self.instance_gpu = TextToSpeech(is_gpu=True)
        print(self.instance.list_available_models())
        content = '''
        以下是每个缩写的简要解释：

hag: Hanga — 指的是一种语言，主要在巴布亚新几内亚的Hanga地区使用。

hnn: Hanunoo — 指的是菲律宾的一种语言，主要由Hanunoo人使用，属于马来-波利尼西亚语系。

bgc: Haryanvi — 指的是印度哈里亚纳邦的一种方言，属于印地语的一种变体。

had: Hatam — 指的是巴布亚新几内亚的一种语言，主要在Hatam地区使用。

hau: Hausa — 指的是西非的一种语言，广泛用于尼日利亚和尼日尔，是主要的交易语言之一。

hwc: Hawaii Pidgin — 指的是夏威夷的一种克里奥尔语，受英语和夏威夷土著语言影响，常用于当地的日常交流。

hvn: Hawu — 指的是印度尼西亚的一种语言，主要在西努沙登加拉省的Hawu地区使用。

hay: Haya — 指的是坦桑尼亚的一种语言，由Haya人使用，属于尼日尔-刚果语系。
    '''
        self.input = HumanMessage(content=content)

    def test_text2speech(self):
        output = self.instance_gpu.invoke(self.input)
        self.assertIsNotNone(output)
        self.assertIsNotNone(output.content)
        logging.info(output.content)
        for item in output.content:
            context_type = item.get("type") 
            if context_type != "audio":
                self.assertIsNotNone(context_type,"audio")

    def test_text2speech_cpu(self):
        output = self.instance.invoke(self.input)
        self.assertIsNotNone(output)
        self.assertIsNotNone(output.content)
        logging.info(output.content)
        for item in output.content:
            context_type = item.get("type") 
            if context_type != "audio":
                self.assertIsNotNone(context_type,"audio")
        
if __name__ == "__main__":
    unittest.main()
