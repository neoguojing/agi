from langchain.agents import Tool,create_react_agent
from agi.tasks.base import Task,function_stats
from typing import Any
from langchain.agents import AgentExecutor
from agi.llms.model_factory import ModelFactory
class Agent(Task):
    
    def __init__(self,tools):
        # prompt = QwenAgentPromptTemplate(
        #     tools=tools,
        #     # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
        #     # This includes the `intermediate_steps` variable because that is needed
        #     input_variables=["input", "intermediate_steps",'tools', 'tool_names', 'agent_scratchpad']
        # )
        # prompt = hub.pull("hwchase17/react-chat")

        prompt = AgentPromptTemplate(
            tools=tools,
            # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
            # This includes the `intermediate_steps` variable because that is needed
            input_variables=["input", "intermediate_steps",'tools', 'tool_names', 'agent_scratchpad']
        )

        print("agent prompt:",prompt)
        output_parser = QwenAgentOutputParser()

        agent = create_react_agent(
            llm=self.excurtor[0],
            tools=tools,
            prompt=prompt,
            output_parser=output_parser,
        )

        self._executor = AgentExecutor.from_agent_and_tools(agent=agent,tools=tools, verbose=True,
                                                            handle_parsing_errors=True,stream_runnable=False)

    @function_stats
    def run(self,input: Any=None,**kwargs):
        if input is None or input == "":
            return ""
        
        # print("Agent.run input---------------",input)
        output = self._executor.invoke({"input":input,"chat_history":""},**kwargs)
        # print("Agent.run output----------------------:",output)
        return output["output"]
    
    async def arun(self,input: Any=None,**kwargs):
        return self.run(input,**kwargs)
    
    def init_model(self):
        model = ModelFactory.get_model("ollama")
        return [model]
    
    def destroy(self):
        print("Agent model should not be destroy ")