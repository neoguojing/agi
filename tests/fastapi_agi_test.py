import unittest
from openai import OpenAI
import base64

# 图片转换为Base64
def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_image

# 音频转换为Base64
def audio_to_base64(audio_path):
    with open(audio_path, "rb") as audio_file:
        encoded_audio = base64.b64encode(audio_file.read()).decode('utf-8')
    return encoded_audio

class TestFastApiAgi(unittest.TestCase):
    def setUp(self):        
        self.client = OpenAI(
            api_key="123", # This is the default and can be omitted
            base_url="http://localhost:8000/v1",
        )
    def test_text(self):
        response = self.client.chat.completions.create(
            model="agi-model",
            # stream=True,
            messages=[
                {
                    "role": "user",
                    "content": "俄乌战争",
                }
            ],
        )
        
        self.assertIsNotNone(response.choices)
        self.assertGreater(len(response.choices),0)
        self.assertIsNotNone(response.choices[0].message)
        self.assertIsNotNone(response.choices[0].message.content)
        self.assertIsNotNone(response.usage)
        print(response)
        stream = self.client.chat.completions.create(
            model="agi-model",
            stream=True,
            messages=[
                {
                    "role": "user",
                    "content": "美国建国多少年了？",
                }
            ],
        )
        for chunk in stream:
            print(chunk)
            self.assertIsNotNone(chunk.choices)
            self.assertGreater(len(chunk.choices),0)
            self.assertIsNotNone(chunk.choices[0].delta)
            # self.assertIsNotNone(chunk.choices[0].delta.content)
            
    def test_text_image(self):
        response = self.client.chat.completions.create(
            model="agi-model",
            messages=[
                {
                    "role": "user",
                    "content": "中国神话哪吒的图片",
                }
            ],
        )
        # print(response)
        self.assertIsNotNone(response.choices)
        self.assertGreater(len(response.choices),0)
        self.assertIsNotNone(response.choices[0].message)
        self.assertIsNotNone(response.choices[0].message.content)
        self.assertGreater(len(response.choices[0].message.content),0)
        self.assertEqual(response.choices[0].message.content[0]['type'],"image")
        self.assertIsNotNone(response.choices[0].message.content[0]['image'])
        # 去掉头部信息，只保留 Base64 编码的部分
        # base64_image = response.choices[0].message.content[0]['image'].split(',')[1]
        # image_data = base64.b64decode(base64_image)
        # # 保存为 JPEG 文件
        # with open("output_image.jpeg", "wb") as f:
        #     f.write(image_data)
        
        
    def test_image_image(self):
        response = self.client.chat.completions.create(
            model="agi-model",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "变成人首蛇身"},
                        {
                            "type": "image",
                            "image": image_to_base64("tests/cat.jpeg")
                        }
                    ]
                }
            ]
        )
        self.assertIsNotNone(response.choices)
        self.assertGreater(len(response.choices),0)
        self.assertIsNotNone(response.choices[0].message)
        self.assertIsNotNone(response.choices[0].message.content)
        self.assertGreater(len(response.choices[0].message.content),0)
        self.assertEqual(response.choices[0].message.content[0]['type'],"image")
        self.assertIsNotNone(response.choices[0].message.content[0]['image'])
        # print(response)
             # 去掉头部信息，只保留 Base64 编码的部分
        base64_image = response.choices[0].message.content[0]['image'].split(',')[1]
        image_data = base64.b64decode(base64_image)
        # 保存为 JPEG 文件
        with open("output_image.jpeg", "wb") as f:
            f.write(image_data)
        print(response)
        
    def test_speech_text(self):
        response = self.client.chat.completions.create(
            model="agi-model",
            messages=[
                {
                    "role": "user",
                    "content": [{"type": "audio", "audio": audio_to_base64("tests/zh-cn-sample.wav")}]
                }
            ],
        )
        
        self.assertIsNotNone(response.choices)
        self.assertGreater(len(response.choices),0)
        self.assertIsNotNone(response.choices[0].message)
        self.assertIsNotNone(response.choices[0].message.content)
        # print(response)
        
    def test_tts(self):
        response = self.client.chat.completions.create(
            model="agi-model",
            extra_query={"need_speech": True},
            messages=[
                {
                    "role": "user",
                    "content": "介绍下中国"
                }
            ],
        )
        
        self.assertIsNotNone(response.choices)
        self.assertGreater(len(response.choices),0)
        self.assertIsNotNone(response.choices[0].message)
        self.assertIsNotNone(response.choices[0].message.content)
        self.assertEqual(response.choices[0].message.content[0]['type'],"audio")
        self.assertIsNotNone(response.choices[0].message.content[0]['audio'])
        # print(response)

    def test_web_search(self):
        response = self.client.chat.completions.create(
            model="agi-model",
            extra_query={"need_speech": False,"feature": "web"},
            messages=[
                {
                    "role": "user",
                    "content": "今天的科技新闻"
                }
            ],
        )
        
        print(response)
        self.assertIsNotNone(response.choices)
        self.assertGreater(len(response.choices),0)
        self.assertIsNotNone(response.choices[0].message)
        self.assertIsNotNone(response.choices[0].message.content)
        self.assertEqual(response.choices[0].message.content[0]['type'],"text")
        self.assertIsNotNone(response.choices[0].message.content[0]['audio'])
        

    def test_rag(self):
        response = self.client.chat.completions.create(
            model="agi-model",
            extra_query={"need_speech": False,"feature": "rag","db_ids": ["web"]},
            messages=[
                {
                    "role": "user",
                    "content": "上海今天的天气"
                }
            ],
        )
        
        print(response)
        self.assertIsNotNone(response.choices)
        self.assertGreater(len(response.choices),0)
        self.assertIsNotNone(response.choices[0].message)
        self.assertIsNotNone(response.choices[0].message.content)
        self.assertEqual(response.choices[0].message.content[0]['type'],"text")
        self.assertIsNotNone(response.choices[0].message.content[0]['audio'])
        