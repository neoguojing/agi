import torch
from langchain.llms.base import LLM
from pydantic import  Field
from typing import Any,Union,List,Dict


class CustomerLLM(LLM):
    device: str = Field(torch.device('cpu'))
    model: Any = None
    tokenizer: Any = None

    def __init__(self,llm,**kwargs):
        super(CustomerLLM, self).__init__()
        if torch.cuda.is_available():
            self.device = torch.device(0)
        else:
            self.device = torch.device('cpu')
        self.model = llm

    def destroy(self):
        if self.model is not None:
            del self.model
            torch.cuda.empty_cache()
            print(f"model {self.model_name} destroy success")

    def encode(self,input):
        if self.tokenizer is not None:
            return self.tokenizer.encode(input)
        return None
        
    def decode(self,ids):
        if self.tokenizer is not None:
            return self.tokenizer.decode(ids)
        return ""
    
    @property
    def model_name(self) -> str:
        return ""
