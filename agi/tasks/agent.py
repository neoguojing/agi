from langgraph.prebuilt import create_react_agent
from agi.tasks.tools import tools
import sqlite3
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt.chat_agent_executor import AgentState

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Respond only in {language}."),
        ("placeholder", "{messages}"),
    ]
)

def _modify_state_messages(state: AgentState):
    return prompt.invoke({"messages": state["messages"],"language":"chinese"}).to_messages() + [
        ("user", "Also say 'Pandamonium!' after the answer.")
    ]

memory = MemorySaver()
def create_react_agent_task(llm):
    langgraph_agent_executor = create_react_agent(llm, 
                                                  tools,state_modifier=_modify_state_messages,
                                                  checkpointer=memory,
                                                  store=InMemoryStore())
    # langgraph_agent_executor.step_timeout = 2
    return langgraph_agent_executor

