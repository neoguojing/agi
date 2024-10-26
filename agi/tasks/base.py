import abc
import asyncio
from typing import Any,Union,List,Dict
import time

from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

class ITask(abc.ABC):
    
    @abc.abstractmethod
    def run(self,input:str):
        pass
    
    @abc.abstractmethod
    def init_model(self):
        pass
    
def function_stats(func):
    call_stats = {"call_count": 0, "last_call_time": None}

    def wrapper(*args, **kwargs):
        nonlocal call_stats

        # 更新调用次数和时间
        call_stats["call_count"] += 1
        current_time = time.time()
        # if call_stats["last_call_time"] is not None:
        #     elapsed_time = current_time - call_stats["last_call_time"]
        #     print(f"函数 {func.__name__} 上次调用时间间隔: {elapsed_time}秒")
        call_stats["last_call_time"] = current_time

        # 执行目标函数
        return func(*args, **kwargs)

    # 添加访问方法到装饰器函数对象
    wrapper.get_call_count = lambda: call_stats["call_count"]
    wrapper.get_last_call_time = lambda: call_stats["last_call_time"]
    wrapper.reset = lambda: call_stats.update({"call_count": 0, "last_call_time": None})

    # 返回装饰后的函数
    return wrapper
    


class Task(ITask):
    _excurtor: list[Runnable] = None
    qinput = asyncio.Queue()
    qoutput: asyncio.Queue = None
    stop_event = asyncio.Event()

    def __init__(self,output:asyncio.Queue=None):
        self.qoutput = output

    @function_stats
    def run(self,input:Any,**kwargs):
        if input is None or input == "":
            return ""
        if isinstance(input,str):
            output = self.excurtor[0].invoke(input,**kwargs)
        else:
            output = self.excurtor[0]._call(input,**kwargs)
        return output
    
    async def arun(self,input:Any,**kwargs):
        return self.run(input,**kwargs)

    @property
    def get_last_call_time(self):
        return self.run.get_last_call_time()
    
    @property
    def get_call_count(self):
        return self.run.get_call_count()
    
    @property
    def excurtor(self):
        if self._excurtor is None:
            # 执行延迟初始化逻辑
            self._excurtor = self.init_model()
        return self._excurtor
    
    def init_model(self):
        return None
    
    def input(self,input:str):
        self.qinput.put_nowait(input)

    def destroy(self):
        self.stop_event.set()
        self._excurtor = None
        self.run.reset()


    def bind_model_name(self):
        if self._excurtor is not None:
            names = []
            for exc in self._excurtor:
                names.append(exc.model_name)
            return names
        return None

    def encode(self,input):
        return self.excurtor[0].encode(input)
 
        
    def decode(self,ids):
        self.excurtor[0].decode(input)
        
    def __call__(self, state: State, config: RunnableConfig):
        resp = []
        output = None
        messages = state["messages"]
        if messages and isinstance(messages, list):
            messages = messages[-1]

        input_type = state["input_type"]
        
        if input_type == "text":
            output = self.run(messages.content)
        elif input_type == "speech":
            if messages.content != "":
                output = self.run(messages.content)
            else:
                output = self.run(messages.additional_kwargs.get('speech'))
        elif input_type == "image":
            # text to image
            if messages.additional_kwargs.get('image') is None and messages.content != "":
                print("text to image:",messages)
                output = self.run(messages)
            else:
                # image to image
                if isinstance(messages.additional_kwargs.get('image'),str):
                    output = self.run(messages,image_path=messages.additional_kwargs.get('image'))
                else:
                    output = self.run(messages,image_obj=messages.additional_kwargs.get('image'))
        
        if isinstance(output,str):
            output = AIMessage(content=output)
        else:
            additional_kwargs = {"media":output}
            output = AIMessage(additional_kwargs=additional_kwargs)
            
        resp.append(output)
        return {"messages": resp}
