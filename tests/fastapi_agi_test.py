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
    # 设置类级别的启动参数
    # @classmethod
    # def setUpClass(cls):
        
    #     import uvicorn
    #     from agi.fastapi_agi import app
    #     import asyncio
    #     import threading
    #     cls.client = OpenAI(
    #         api_key="123", # This is the default and can be omitted
    #         base_url="http://localhost:8000/v1",
    #     )
    #      # 创建一个异步函数来启动Uvicorn服务器
    #     def start_uvicorn():
    #         config = uvicorn.Config(app, host="0.0.0.0", port=8000)
    #         server = uvicorn.Server(config)
    #         asyncio.run(server.serve())
        
    #     # 启动Uvicorn服务器的线程
    #     cls.server_thread = threading.Thread(target=start_uvicorn)
    #     cls.server_thread.daemon = True  # 设置为daemon线程，这样主程序结束时会自动退出
    #     cls.server_thread.start()

    #     # 等待服务器启动
    #     cls.wait_for_server_start()
        
    @classmethod
    def wait_for_server_start(cls, timeout=60):
        """等待Uvicorn服务器完全启动"""
        import time
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                # 尝试发送一个请求来检查服务器是否已启动
                response = cls.client.chat.completions.create(
                    model="agi-model",
                    messages=[
                        {
                            "role": "user",
                            "content": "hello",
                        }
                    ],
                )
                if response:
                    print("Server started successfully!")
                    return True
            except Exception as e:
                # 如果发生连接错误，继续等待
                print("Waiting for server to start...")
                time.sleep(1)
        raise TimeoutError("Uvicorn server did not start within the timeout period.")
    
    # def test_text(self):
    #     response = self.client.chat.completions.create(
    #         model="agi-model",
    #         messages=[
    #             {
    #                 "role": "user",
    #                 "content": "俄乌战争",
    #             }
    #         ],
    #         user="text"
    #     )
        
    #     self.assertIsNotNone(response.choices)
    #     self.assertGreater(len(response.choices),0)
    #     self.assertIsNotNone(response.choices[0].message)
    #     self.assertIsNotNone(response.choices[0].message.content)
    #     self.assertIsNotNone(response.usage)
    #     print(response)
    #     # TODO 统一用户的请求会串：比如上面的请求会在下面返回
    #     stream = self.client.chat.completions.create(
    #         model="agi-model",
    #         stream=True,
    #         messages=[
    #             {
    #                 "role": "user",
    #                 "content": "美国建国多少年了？",
    #             }
    #         ],
    #         user="text1"
    #     )
    #     is_stoped = False
    #     for chunk in stream:
    #         print("-----",chunk)
    #         self.assertIsNotNone(chunk.choices)
    #         self.assertGreater(len(chunk.choices),0)
    #         self.assertIsNotNone(chunk.choices[0].delta)
    #         if chunk.choices[0].finish_reason is None:
    #             self.assertIsNotNone(chunk.choices[0].delta.content)
    #         else:
    #             # self.assertEqual(chunk.choices[0].finish_reason,"stop")
    #             if chunk.choices[0].finish_reason == "stop":
    #                 is_stoped = True
    #     self.assertEqual(is_stoped,True)
            
    # # 通过agent，支持的图片生成
    # def test_text_image(self):
    #     response = self.client.chat.completions.create(
    #         model="agi-model",
    #         messages=[
    #             {
    #                 "role": "user",
    #                 "content": "中国神话哪吒的图片",
    #             }
    #         ],
    #         user="image"
    #     )
    #     print(response)
    #     self.assertIsNotNone(response.choices)
    #     self.assertGreater(len(response.choices),0)
    #     self.assertIsNotNone(response.choices[0].message)
    #     self.assertIsNotNone(response.choices[0].message.content)
    #     self.assertIsInstance(response.choices[0].message.content,list)
    #     self.assertEqual(response.choices[0].message.content[0]["type"],"image")
    #     self.assertIsInstance(response.choices[0].message.content[0]["image"],str)
    #     self.assertGreater(len(response.choices[0].message.content[0]["image"]),0)
        
    #     stream = self.client.chat.completions.create(
    #         model="agi-model",
    #         stream=True,
    #         messages=[
    #             {
    #                 "role": "user",
    #                 "content": "生成芯片架构图",
    #             }
    #         ],
    #         user="image1"
    #     )
    #     is_stoped = False
    #     for chunk in stream:
    #         print("------",chunk)
    #         # TODO 会返回一个finish_reason='tool_calls',
    #         self.assertIsNotNone(chunk.choices)
    #         self.assertGreater(len(chunk.choices),0)
    #         if chunk.choices[0].finish_reason is None:
    #             self.assertIsNotNone(chunk.choices[0])
    #             self.assertIsNotNone(chunk.choices[0].delta)
    #             self.assertIsNotNone(chunk.choices[0].delta.content)
    #             if isinstance(chunk.choices[0].delta.content,dict):
    #                 self.assertEqual(chunk.choices[0].delta.content.get("type"),"image")
    #                 self.assertIsNotNone(chunk.choices[0].delta.content.get("image"))
    #         else:
    #             if chunk.choices[0].finish_reason == "stop":
    #                 is_stoped = True
    #     self.assertEqual(is_stoped,True)
        
        
    # def test_image_image(self):
    #     response = self.client.chat.completions.create(
    #         model="agi-model",
    #         messages=[
    #             {
    #                 "role": "user",
    #                 "content": [
    #                     {"type": "text", "text": "变成人首蛇身"},
    #                     {
    #                         "type": "image",
    #                         "image": image_to_base64("tests/cat.jpg")
    #                     }
    #                 ]
    #             }
    #         ],
    #         user="image2"
    #     )
    #     print(response)
    #     self.assertIsNotNone(response.choices)
    #     self.assertGreater(len(response.choices),0)
    #     self.assertIsNotNone(response.choices[0].message)
    #     self.assertIsNotNone(response.choices[0].message.content)
    #     self.assertIsInstance(response.choices[0].message.content,list)
    #     self.assertEqual(response.choices[0].message.content[0]["type"],"image")
    #     self.assertIsInstance(response.choices[0].message.content[0]["image"],str)
    #     self.assertGreater(len(response.choices[0].message.content[0]["image"]),0)
        
    #     stream = self.client.chat.completions.create(
    #         model="agi-model",
    #         stream=True,
    #         messages=[
    #             {
    #                 "role": "user",
    #                 "content": [
    #                     {"type": "text", "text": "变成一条狗"},
    #                     {
    #                         "type": "image",
    #                         "image": image_to_base64("tests/cat.jpg")
    #                     }
    #                 ]
    #             }
    #         ],
    #         user="image3"
    #     )
    #     is_stoped = False
    #     for chunk in stream:
    #         print("------",chunk)
    #         self.assertIsNotNone(chunk.choices)
    #         self.assertGreater(len(chunk.choices),0)
    #         if chunk.choices[0].finish_reason is None:
    #             self.assertIsNotNone(chunk.choices[0])
    #             self.assertIsNotNone(chunk.choices[0].delta)
    #             self.assertIsNotNone(chunk.choices[0].delta.content)
    #             if isinstance(chunk.choices[0].delta.content,dict):
    #                 self.assertEqual(chunk.choices[0].delta.content.get("type"),"image")
    #                 self.assertIsNotNone(chunk.choices[0].delta.content.get("image"))
    #         else:
    #             if chunk.choices[0].finish_reason == "stop":
    #                 is_stoped = True
    #     self.assertEqual(is_stoped,True)
    
    # def test_speech_text(self):
    #     response = self.client.chat.completions.create(
    #         model="agi-model",
    #         messages=[
    #             {
    #                 "role": "user",
    #                 "content": [{"type": "audio", "audio": audio_to_base64("tests/zh-cn-sample.wav")}]
    #             }
    #         ],
    #         user="speech"
    #     )
    #     print(response)
    #     self.assertIsNotNone(response.choices)
    #     self.assertGreater(len(response.choices),0)
    #     self.assertIsNotNone(response.choices[0].message)
    #     self.assertIsNotNone(response.choices[0].message.content)
    #     # 测试直接语音转文字
    #     response = self.client.chat.completions.create(
    #         model="agi-model",
    #         extra_body={"need_speech": False,"feature": "speech"},
    #         messages=[
    #             {
    #                 "role": "user",
    #                 "content": [{"type": "audio", "audio": audio_to_base64("tests/zh-cn-sample.wav")}]
    #             }
    #         ],
    #         user="speech1"
    #     )
    #     print(response)
    #     self.assertIsNotNone(response.choices)
    #     self.assertGreater(len(response.choices),0)
    #     self.assertIsNotNone(response.choices[0].message)
    #     self.assertIsNotNone(response.choices[0].message.content)
    #     self.assertEqual(response.choices[0].message.content,"当我还只有六岁的时候,看到了一幅精彩的插画。")
        
    #     stream = self.client.chat.completions.create(
    #         model="agi-model",
    #         stream=True,
    #         # extra_body={"need_speech": False,"feature": "speech"},
    #         messages=[
    #             {
    #                 "role": "user",
    #                 "content": [{"type": "audio", "audio": audio_to_base64("tests/zh-cn-sample.wav")}]
    #             }
    #         ],
    #         user="speech2"
    #     )
    #     is_stoped = False
    #     for chunk in stream:
    #         print("------",chunk)
    #         # TODO 会返回一个finish_reason='tool_calls',
    #         self.assertIsNotNone(chunk.choices)
    #         self.assertGreater(len(chunk.choices),0)
    #         if chunk.choices[0].finish_reason is None:
    #             self.assertIsNotNone(chunk.choices[0])
    #             self.assertIsNotNone(chunk.choices[0].delta)
    #             self.assertIsNotNone(chunk.choices[0].delta.content)
    #         else:
    #             if chunk.choices[0].finish_reason == "stop":
    #                 is_stoped = True
    #     self.assertEqual(is_stoped,True)
        
        
    # def test_tts(self):
    #     # TODO 两个请求在一起会导致模型串掉 DONE 使用不同的用户可以避免
    #     response = self.client.chat.completions.create(
    #         model="agi-model",
    #         extra_body={"need_speech": True},
    #         messages=[
    #             {
    #                 "role": "user",
    #                 "content": "介绍下中国"
    #             }
    #         ],
    #         user="tts1"
    #     )
    #     print(response)
    #     self.assertIsNotNone(response.choices)
    #     self.assertGreater(len(response.choices),0)
    #     self.assertIsNotNone(response.choices[0].message)
    #     self.assertIsNotNone(response.choices[0].message.content)
    #     self.assertIsInstance(response.choices[0].message.content,list)
    #     self.assertEqual(response.choices[0].message.content[0]["type"],"audio")
    #     self.assertIsNotNone(response.choices[0].message.content[0]["file_path"])
    #     self.assertIsNotNone(response.choices[0].message.content[0]["text"])
    #     self.assertIsInstance(response.choices[0].message.content[0]["audio"],str)
    #     self.assertGreater(len(response.choices[0].message.content[0]["audio"]),0)
        
    #     stream = self.client.chat.completions.create(
    #         model="agi-model",
    #         stream=True,
    #         extra_body={"need_speech": False,"feature": "tts"},
    #         messages=[
    #             {
    #                 "role": "user",
    #                 "content": "介绍下中国"
    #             }
    #         ],
    #         user="tts2"
    #     )
    #     is_stoped = False
    #     for chunk in stream:
    #         print("------",chunk)
    #         self.assertIsNotNone(chunk.choices)
    #         self.assertGreater(len(chunk.choices),0)
    #         if chunk.choices[0].finish_reason is None:
    #             self.assertIsNotNone(chunk.choices[0])
    #             self.assertIsNotNone(chunk.choices[0].delta)
    #             self.assertIsNotNone(chunk.choices[0].delta.content)
    #             self.assertEqual(chunk.choices[0].delta.content[0]["type"],"audio")
    #             self.assertIsNotNone(chunk.choices[0].delta.content[0]["file_path"])
    #             self.assertIsNotNone(chunk.choices[0].delta.content[0]["text"])
    #             self.assertEqual(chunk.choices[0].delta.content[0]["text"],"介绍下中国")
    #             self.assertIsInstance(chunk.choices[0].delta.content[0]["audio"],str)
    #             self.assertGreater(len(chunk.choices[0].delta.content[0]["audio"]),0)
    #         else:
    #             if chunk.choices[0].finish_reason == "stop":
    #                 is_stoped = True
    #     self.assertEqual(is_stoped,True)
        

    def test_web_search(self):
        response = self.client.chat.completions.create(
            model="agi-model",
            extra_body={"need_speech": False,"feature": "web"},
            messages=[
                {
                    "role": "user",
                    "content": "esp32单片机"
                }
            ],
            user="web"
        )
        
        print(response)
        self.assertIsNotNone(response.choices)
        self.assertGreater(len(response.choices),0)
        self.assertIsNotNone(response.choices[0].message)
        self.assertIsNotNone(response.choices[0].message.content)
        self.assertIsNotNone(response.choices[0].message.content["citations"])
        self.assertIsInstance(response.choices[0].message.content["citations"],list)
    
        stream = self.client.chat.completions.create(
            model="agi-model",
            stream=True,
            extra_body={"need_speech": False,"feature": "web"},
            messages=[
                {
                    "role": "user",
                    "content": "esp32单片机",
                }
            ],
            user="web1"
        )
        is_stoped = False
        for chunk in stream:
            print("------",chunk)
            # TODO 重复的tool消息
            # TODO 查询结果关联性不高
            self.assertIsNotNone(chunk.choices)
            self.assertGreater(len(chunk.choices),0)
            if chunk.choices[0].finish_reason is None:
                self.assertIsNotNone(chunk.choices[0])
                self.assertIsNotNone(chunk.choices[0].delta)
                self.assertIsNotNone(chunk.choices[0].delta.content)
                if isinstance(chunk.choices[0].delta.content,dict):
                    self.assertEqual(chunk.choices[0].delta.content.get("type"),"text")
                    self.assertIsNotNone(chunk.choices[0].delta.content.get("text"))
                    self.assertIsNotNone(chunk.choices[0].delta.content.get("citations"))
                    self.assertIsInstance(chunk.choices[0].delta.content.get("citations"),list)
            else:
                self.assertEqual(chunk.choices[0].finish_reason,"stop")
                is_stoped = True
        self.assertEqual(is_stoped,True)
        

    # def test_rag(self):
    #     response = self.client.chat.completions.create(
    #         model="agi-model",
    #         extra_body={"db_ids":["test"],"need_speech": False,"feature": "rag"},
    #         messages=[
    #             {
    #                 "role": "user",
    #                 "content": "NTP3000Plus"
    #             }
    #         ],
    #     )
        
    #     print("----------------",response)
    #     self.assertIsNotNone(response.choices)
    #     self.assertGreater(len(response.choices),0)
    #     self.assertIsNotNone(response.choices[0].message)
    #     self.assertIsNotNone(response.choices[0].message.content)
    #     self.assertIsNotNone(response.choices[0].message.content["citations"])
    #     self.assertIsInstance(response.choices[0].message.content["citations"],list)

    #     stream = self.client.chat.completions.create(
    #         model="agi-model",
    #         stream=True,
    #         extra_body={"db_ids":["test"],"need_speech": False,"feature": "rag"},
    #         messages=[
    #             {
    #                 "role": "user",
    #                 "content": "NTP3000Plus",
    #             }
    #         ],
    #     )
    #     is_stoped = False
    #     for chunk in stream:
    #         print("------",chunk)
    #         # TODO 重复的tool消息
    #         self.assertIsNotNone(chunk.choices)
    #         self.assertGreater(len(chunk.choices),0)
    #         if chunk.choices[0].finish_reason is None:
    #             self.assertIsNotNone(chunk.choices[0])
    #             self.assertIsNotNone(chunk.choices[0].delta)
    #             self.assertIsNotNone(chunk.choices[0].delta.content)
    #             if isinstance(chunk.choices[0].delta.content,dict):
    #                 self.assertEqual(chunk.choices[0].delta.content.get("type"),"text")
    #                 self.assertIsNotNone(chunk.choices[0].delta.content.get("text"))
    #                 self.assertIsNotNone(chunk.choices[0].delta.content.get("citations"))
    #                 self.assertIsInstance(chunk.choices[0].delta.content.get("citations"),list)
    #         else:
    #             self.assertEqual(chunk.choices[0].finish_reason,"stop")
    #             is_stoped = True
    #     self.assertEqual(is_stoped,True)
        
    # def test_embedding(self):  
    #     response = self.client.embeddings.create(
    #         model='text-embedding-ada-002',
    #         input="我爱北京天安门"
    #     )

    #     print(response)
    #     self.assertIsNotNone(response.data)
    #     self.assertGreater(len(response.data),0)
    #     self.assertIsNotNone(response.data[0].embedding)
    
    # # 语音转文本
    # def test_transcription(self):  
    #     with open('tests/zh-cn-sample.wav', 'rb') as audio_file:
    #         response = self.client.audio.transcriptions.create(
    #             file=audio_file,
    #             model='whisper-1',
    #             response_format='json',  # 可选：'text', 'srt', 'vtt', 'verbose_json'
    #             language='zh'  # 可选：指定音频语言，例如 'en'、'zh' 等
    #         )
 

    #         print(response)
    #         self.assertIsNotNone(response.text)
    #         self.assertEqual(response.text,"当我还只有六岁的时候,看到了一幅精彩的插画。")

        
    # # 文本转语音
    # def test_speech(self):  
    #     with self.client.audio.speech.with_streaming_response.create(
    #         model="tts-1",
    #         voice="alloy",
    #         input="the quick brown fox jumped over the lazy dogs",
    #     ) as response:
    #         response.stream_to_file("tests/test.wav")
    #         import os
    #         self.assertTrue(os.path.exists("tests/test.wav"))
            
