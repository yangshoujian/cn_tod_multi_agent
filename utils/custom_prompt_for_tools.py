import os.path
from typing import Any, Awaitable, Callable, Optional, Union
from langchain.prompts.base import StringPromptTemplate
from utils.file_process import jsonl_writer, jsonl_reader
import json

file_memory = "/Users/chendongdong/Work/llm/huochebao/driver_agent/resources/memory.jsonl"

class CustomPromptTemplateForTools(StringPromptTemplate):
    # The template to use
    template: str
    ############## NEW ######################
    # The list of tools available
    tools_getter: Callable

    def format(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        # raw_tool_memory = []
        tool_memory = []
        if os.path.exists(file_memory):
            tool_memory = jsonl_reader(file_memory)

        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
            # if "valid" not in observation:
            tool_result = { action.tool : "结果：" + str(observation) }
            tool_memory.append(json.dumps(tool_result, ensure_ascii=False))
        jsonl_writer(file_memory, tool_memory)
        # Set the agent_scratchpad variable to that value
        kwargs["AGENT_SCRATCHPAD"] = thoughts
        ############## NEW ######################
        tools = self.tools_getter(kwargs["input"])
        # Create a tools variable from the list of tools provided
        kwargs["TOOLS"] = "\n".join(
            [f"{tool.name}: {tool.description}" for tool in tools]
        )
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in tools])
        return self.template.format(**kwargs)