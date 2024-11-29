import os.path
from typing import List, Dict, Union, Any
import json
import time
from langchain.agents import AgentExecutor, LLMSingleActionAgent
from langchain.chains import LLMChain
from langchain.chains.base import Chain
from langchain.llms import BaseLLM
from pydantic import Field

from chains.state import DriverStateChain
from database.person import get_person_info
from utils.driver_conv_output_parser import DriverConvOutputParser
from utils.custom_prompt_for_tools import CustomPromptTemplateForTools
from utils.constants import SALES_AGENT_TOOLS_PROMPT
from tools.get_tools import search_order_tool
from prompts.driver_v2 import *
from chains.stage import SellerStageChain, DriverStageChain
from chains.strategy import DriverStrategyChain
from chains.seller import SellerConversationChain
from chains.driver import DriverConversationChain
from utils.file_process import jsonl_reader

# from prompts.driver_v2 import DRIVER_PROMPT_TEMPLATE, STATE_PROMPT_TEMPLATE, STRATEGY_PROMPT_TEMPLATE
file_memory = "/Users/chendongdong/Work/llm/huochebao/driver_agent/resources/memory.jsonl"
file_person = "/Users/chendongdong/Work/llm/huochebao/driver_agent/resources/drivers.jsonl"
if os.path.exists(file_memory):
    os.remove(file_memory)


class DriverGPT(Chain):
    """货车司机agent"""

    conversation_history: List[str] = []
    thought: str = ""
    dialogue_strategy: str = ""
    dialogue_stage: str = ""
    dialogue_stage_name: str = "init"
    dialogue_state: str = ""
    dialogue_strategy_all: str = ""
    dst : Dict = {}
    stage_analyzer_chain: DriverStageChain = Field(...)
    state_analyzer_chain: DriverStateChain = Field(...)
    strategy_analyzer_chain: DriverStrategyChain = Field(...)
    driver_conversation_utterance_chain: DriverConversationChain = Field(...)
    driver_agent_executor: Union[AgentExecutor, None] = Field(...)
    personal_info = get_person_info(file_person)
    memory: str = ""
    streaming: bool  = True
    print(f"PERSONAL_INFO: {personal_info}")

    use_tools: bool = False

    @property
    def input_keys(self) -> List[str]:
        return []

    @property
    def output_keys(self) -> List[str]:
        return []

    def seed_agent(self):
        # Step 1: 初始化driver agent
        self.conversation_history = []

    def determine_conversation_state(self):
        last_dialogue_state = ""
        for i in range(3):
            try:
                last_dialogue_state = self.dialogue_state
                conversation_state = self.state_analyzer_chain.run(
                    DIALOGUE="\n".join(self.conversation_history),
                    DIALOGUE_STATES=DIALOGUE_STATES,
                    DS_RESPONSE_FORMAT=DS_RESPONSE_FORMAT,
                    MEMORY=self.memory
                )
                print(f"STATE : {conversation_state}")
                conversation_state_json = json.loads(conversation_state)

                self.dialogue_state = conversation_state_json

                print(f"Conversation State: {self.dialogue_state}")
                return self.dialogue_state
            except:
                self.dialogue_state = last_dialogue_state
                print(f"STATE Attempt {i + 1} failed")
                time.sleep(0.05)  # 等待一段时间再重试

    def determine_conversation_stage(self):
        last_dialogue_state = ""
        for i in range(3):
            try:
                conversation_stage = self.stage_analyzer_chain.run(
                    PERSONAL_INFO = self.personal_info,
                    DIALOGUE_STRATEGY = self.dialogue_strategy,
                    LAST_DIALOGUE_STATE = self.dialogue_state,
                    LAST_DIALOGUE_STAGE = self.dialogue_stage_name,
                    DIALOGUE = "\n".join(self.conversation_history),
                    DIALOGUE_STAGES = DIALOGUE_STAGES,
                    DIALOGUE_STATES=DIALOGUE_STATES,
                    DIALOGUE_STAGE_TRANSFER = DIALOGUE_STAGE_TRANSFER,
                    STATE_RESPONSE_FORMAT = STATE_RESPONSE_FORMAT,
                    STATE_RESPONSE_EXAMPLE = STATE_RESPONSE_EXAMPLE
                )
                print(f"STAGE : {conversation_stage}")
                conversation_state_json = json.loads(conversation_stage)

                self.thought = conversation_state_json["想法"]
                self.dialogue_stage_name = conversation_state_json["对话阶段"]
                self.dialogue_stage = DIALOGUE_STAGES[conversation_state_json["对话阶段"]]
                self.dialogue_strategy_all = "\n".join(DIALOGUE_STRATEGY[conversation_state_json["对话阶段"]])
                print(f"Thought: {self.thought}")
                print(f"Conversation Stage: {self.dialogue_stage}")
                print(f'Stage: {conversation_state_json["对话阶段"]}')
                return self.thought, self.dialogue_strategy, self.dialogue_stage
            except:
                print(f"STAGE Attempt {i + 1} failed")
                time.sleep(0.05)  # 等待一段时间再重试


    def determine_conversation_strategy(self):

        for i in range(3):
            try:
                conversation_strategy = self.strategy_analyzer_chain.run(
                    DIALOGUE_STRATEGY = self.dialogue_strategy_all,
                    PERSONAL_INFO = self.personal_info,
                    DIALOGUE_STAGE = self.dialogue_stage,
                    DIALOGUE_STATE=self.dialogue_state,
                    DIALOGUE = "\n".join(self.conversation_history),
                    DIALOGUE_STATES=DIALOGUE_STATES,
                    STRATEGY_RESPONSE_FORMAT = STRATEGY_RESPONSE_FORMAT
                )
                print(f"STRATEGY : {conversation_strategy}")
                conversation_strategy_json = json.loads(conversation_strategy)
                #
                # # self.thought = conversation_state_json["想法"]
                self.dialogue_strategy = conversation_strategy_json["最优策略"]

                print(f"Conversation Strategy: {self.dialogue_strategy}")
                # print(f"Thought: {self.thought}")
                return self.thought, self.dialogue_strategy
            except:
                print(f"STRATEGY Attempt {i + 1} failed")
                time.sleep(0.05)  # 等待一段时间再重试


    def step(self, stream=False):
        if stream:
            return self._call_stream(inputs={})
        else:
            return self._call(inputs={})


    def _call_stream(self, inputs: Dict[str, Any]) -> None:

        for event in self.driver_agent_executor.stream(
                input="",
                PERSONAL_INFO=self.personal_info,
                DIALOGUE_STRATEGY=self.dialogue_strategy,
                DIALOGUE_STATE=self.dialogue_state,
                DIALOGUE_STAGE=self.dialogue_stage,
                DIALOGUE="\n".join(self.conversation_history),
                DIALOGUE_STATES=DIALOGUE_STATES,
                MEMORY=self.memory,
        ):
            kind = event["event"]
            if kind == "on_chat_model_stream":
                content = event["data"]["chunk"].content
                if content:
                    yield content

    def join_history(self, text):
        self.conversation_history.append(text)


    def _call(self, inputs: Dict[str, Any]) -> None:
        """Run one step of the sales agent."""
        print(self.use_tools)
        if os.path.exists(file_memory):
            self.memory = "\n".join(jsonl_reader(file_memory))
        # Generate agent's utterance
        if self.use_tools:
            ai_message = self.driver_agent_executor.run(
                input = "",
                PERSONAL_INFO = self.personal_info,
                DIALOGUE_STRATEGY = self.dialogue_strategy,
                DIALOGUE_STATE = self.dialogue_state,
                DIALOGUE_STAGE = self.dialogue_stage,
                DIALOGUE = "\n".join(self.conversation_history),
                DIALOGUE_STATES = DIALOGUE_STATES,
                MEMORY=self.memory,
            )
        else:
            ai_message = self.driver_conversation_utterance_chain.run(
                PERSONAL_INFO = self.personal_info,
                DIALOGUE_STRATEGY = self.dialogue_strategy,
                DIALOGUE_STATE = self.dialogue_state,
                DIALOGUE_STAGE=self.dialogue_stage,
                DIALOGUE = "\n".join(self.conversation_history),
                DIALOGUE_STATES=DIALOGUE_STATES,
                MEMORY=self.memory
            )

        # Add agent's response to conversation history
        print(f"司机: ", ai_message.rstrip("<END_OF_TURN>"))
        raw_response = ai_message.rstrip("<END_OF_TURN>")
        agent_name = "司机"
        ai_message = agent_name + ": " + ai_message
        self.conversation_history.append(ai_message)

        return {"content": raw_response}

    def human_step(self, human_input=""):
        raw_human_input = human_input

        human_input = "货车宝：" + raw_human_input
        self.conversation_history.append(human_input)
        return raw_human_input

    @classmethod
    def from_llm(cls, llm: BaseLLM, verbose: bool = False, **kwargs) -> "DriverGPT":
        """Initialize the HuoCheBaoSalesGPT Controller."""
        stage_analyzer_chain = DriverStageChain.from_llm(llm, verbose=verbose)
        state_analyzer_chain = DriverStateChain.from_llm(llm, verbose=verbose)
        strategy_analyzer_chain = DriverStrategyChain.from_llm(llm, verbose=verbose)

        driver_conversation_utterance_chain = DriverConversationChain.from_llm(
            llm, verbose=verbose
        )

        if "use_tools" in kwargs.keys() and kwargs["use_tools"] is False:
            driver_agent_executor = None

        else:
            tools = search_order_tool()
            print('tools ', tools)

            prompt = CustomPromptTemplateForTools(
                template=DRIVER_PROMPT_TEMPLATE,
                tools_getter=lambda x: tools,
                # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
                # This includes the `intermediate_steps` variable because that is needed
                input_variables=[
                "input",
                "intermediate_steps",
                "PERSONAL_INFO",
                "DIALOGUE_STRATEGY",
                "DIALOGUE_STATE",
                "DIALOGUE_STAGE",
                "DIALOGUE",
                "MEMORY"
                ],
            )
            llm_chain = LLMChain(llm=llm, prompt=prompt, verbose=verbose)

            tool_names = [tool.name for tool in tools]

            # WARNING: this output parser is NOT reliable yet
            ## It makes assumptions about output from LLM which can break and throw an error
            output_parser = DriverConvOutputParser(
                ai_prefix="司机", verbose=verbose
            )

            driver_agent_with_tools = LLMSingleActionAgent(
                llm_chain=llm_chain,
                output_parser=output_parser,
                stop=["\nObservation:"],
                allowed_tools=tool_names,
                verbose=verbose,
            )

            driver_agent_executor = AgentExecutor.from_agent_and_tools(
                agent=driver_agent_with_tools, tools=tools, verbose=verbose
            )
        return cls(
            stage_analyzer_chain=stage_analyzer_chain,
            state_analyzer_chain=state_analyzer_chain,
            strategy_analyzer_chain=strategy_analyzer_chain,
            driver_conversation_utterance_chain=driver_conversation_utterance_chain,
            driver_agent_executor=driver_agent_executor,
            verbose=verbose,
            **kwargs,
        )


