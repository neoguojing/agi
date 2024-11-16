import time
from typing import Any
import threading
from agi.llms.model_factory import ModelFactory
from langchain_core.runnables import Runnable
TASK_AGENT = 100
TASK_TRANSLATE = 200
TASK_DATA_HANDLER = 300
TASK_IMAGE_GEN = 400
TASK_SPEECH = 500
TASK_GENERAL = 600
TASK_RETRIEVER = 700

class ImageGenTask():

    def init_model(self):
        model = ModelFactory.get_model("text2image")
        model1 = ModelFactory.get_model("image2image")
        return [model,model1]
    
    @function_stats
    def run(self,input:Any,**kwargs):
        if input is None:
            return ""
        image_path = kwargs.pop("image_path","")
        image_obj = kwargs.pop("image_obj",None)
        if image_path != "" or image_obj is not None:
            output = self.excurtor[1]._call(input.content,image_path=image_path,image_obj=image_obj)
        else:
            output = self.excurtor[0]._call(input.content,**kwargs)
        
        return output

class Speech():
    def init_model(self):
        # model = ModelFactory.get_model("speech")
        # return [model]
        model = ModelFactory.get_model("speech2text")
        model1 = ModelFactory.get_model("text2speech")
        return [model,model1]
        
    def run(self,input:Any,**kwargs):
        if input is None:
            return ""
        
        # output = self.excurtor[0]._call(input,**kwargs)
        if isinstance(input,str):
            output = self.excurtor[1]._call(input,**kwargs)
        else:
            output = self.excurtor[0]._call(input,**kwargs)
        
        return output
    
    async def arun(self,input:Any,**kwargs):
        return self.run(input,**kwargs)
    
    def set_tone(self,path:str):
        self.excurtor[1].set_tone(path)

    

class TranslateTask():
    def init_model(self):
        model = ModelFactory.get_model("translate")
        return [model]

    
class TaskFactory:
    _instances = {}
    _lock = threading.Lock()  # 异步锁

    @staticmethod
    def create_task(task_type) -> Runnable:
        if task_type not in TaskFactory._instances:
            with TaskFactory._lock:
                if task_type not in TaskFactory._instances:
                    try:
                        if task_type == TASK_AGENT:
                            instance = Agent()
                        elif task_type == TASK_TRANSLATE:
                            instance = TranslateTask()
                        elif task_type == TASK_IMAGE_GEN:
                            instance = ImageGenTask()
                        elif task_type == TASK_SPEECH:
                            instance = Speech()
                        elif task_type == TASK_GENERAL:
                            instance = General()
                        elif task_type == TASK_RETRIEVER:
                            instance = Retriever()

                        TaskFactory._instances[task_type] = instance
                    except Exception as e:
                        print(e)

        return TaskFactory._instances[task_type]



        