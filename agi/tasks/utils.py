from langchain_core.messages import AIMessage, HumanMessage,BaseMessage,ToolMessage
from typing import Any, List, Mapping, Optional, Union
from langchain_core.runnables import (
    RunnableLambda
)
def graph_response_format(message :Union[AIMessage,ToolMessage,list[BaseMessage]]):
    if isinstance(message,list):
        return {"messages": message}
    
    return {"messages": [message]} 

graph_response_format_runnable = RunnableLambda(graph_response_format)