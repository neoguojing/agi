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
                    "content": "生成一张战争的图片",
                }
            ],
        )
        
        self.assertIsNotNone(response.choices)
        self.assertGreater(len(response.choices),0)
        self.assertIsNotNone(response.choices[0].message)
        self.assertIsNotNone(response.choices[0].message.content)
        self.assertGreater(len(response.choices[0].message.content),0)
        self.assertEqual(response.choices[0].message.content[0]['type'],"image")
        self.assertIsNotNone(response.choices[0].message.content[0]['image'])
        # print(response)
        
    def test_image_image(self):
        response = self.client.chat.completions.create(
            model="agi-model",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "猫咪是黑猫警长"},
                        {
                            "type": "image",
                            "image": image_to_base64("./cat.jpg")
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