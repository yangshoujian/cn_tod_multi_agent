from typing import List, Dict, Union, Any
import json
import time
from langchain.agents import AgentExecutor, LLMSingleActionAgent
from langchain.chains import LLMChain
from langchain.chains.base import Chain
from langchain.llms import BaseLLM
from pydantic import Field
from utils.driver_conv_output_parser import DriverConvOutputParser
from utils.custom_prompt_for_tools import CustomPromptTemplateForTools
from utils.constants import SALES_AGENT_TOOLS_PROMPT
from tools.get_tools import search_order_tool
from prompts.driver_v1 import *
from chains.stage import SellerStageChain, DriverStageChain
from chains.seller import SellerConversationChain
from chains.driver import DriverConversationChain
from prompts.driver_v1 import DRIVER_PROMPT_TEMPLATE, STATE_PROMPT_TEMPLATE

class DriverGPT(Chain):
    """货车司机agent"""

    conversation_history: List[str] = []
    thought: str = ""
    dialogue_strategy: str = ""
    dialogue_stage: str = "init"
    dialogue_state: str = ""
    dst : Dict = {}
    stage_analyzer_chain: DriverStageChain = Field(...)
    driver_conversation_utterance_chain: DriverConversationChain = Field(...)
    driver_agent_executor: Union[AgentExecutor, None] = Field(...)

    use_tools: bool = False
    personal_info: Dict = DRIVER_PERSONAL_INFO

    @property
    def input_keys(self) -> List[str]:
        return []

    @property
    def output_keys(self) -> List[str]:
        return []

    def seed_agent(self):
        # Step 1: 初始化driver agent
        self.conversation_history = []

    def determine_conversation_stage(self):
        for i in range(3):
            try:
                conversation_stage = self.stage_analyzer_chain.run(
                    PERSONAL_INFO = self.personal_info,
                    DIALOGUE_STRATEGY = self.dialogue_strategy,
                    LAST_DIALOGUE_STATE = self.dialogue_state,
                    LAST_DIALOGUE_STAGE = self.dialogue_stage,
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
                self.dialogue_strategy = conversation_state_json["对话策略"]
                self.dialogue_state = conversation_state_json["对话状态"]
                self.dialogue_stage = DIALOGUE_STAGES[conversation_state_json["对话阶段"]]

                print(f"Conversation Stage: {self.dialogue_state}")
                print(f"Conversation Strategy: {self.dialogue_strategy}")
                print(f"Thought: {self.thought}")
                return self.thought, self.dialogue_strategy, self.dialogue_state, self.dialogue_stage
            except:
                print(f"Attempt {i + 1} failed")
                time.sleep(0.05)  # 等待一段时间再重试


    def step(self):
        return self._call(inputs={})

    def _call(self, inputs: Dict[str, Any]) -> None:
        """Run one step of the sales agent."""
        print(self.use_tools)
        # Generate agent's utterance
        if self.use_tools:
            ai_message = self.driver_agent_executor.run(
                input = "",
                PERSONAL_INFO = self.personal_info,
                DIALOGUE_STRATEGY = self.dialogue_strategy,
                DIALOGUE_STATE = self.dialogue_state,
                DIALOGUE_STAGE = self.dialogue_stage,
                DIALOGUE = "\n".join(self.conversation_history),
                DIALOGUE_STATES = DIALOGUE_STATES
            )

        else:
            ai_message = self.driver_conversation_utterance_chain.run(
                PERSONAL_INFO = self.personal_info,
                DIALOGUE_STRATEGY = self.dialogue_strategy,
                DIALOGUE_STATE = self.dialogue_state,
                DIALOGUE_STAGE=self.dialogue_stage,
                DIALOGUE = "\n".join(self.conversation_history),
                DIALOGUE_STATES=DIALOGUE_STATES
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
                "DIALOGUE_STATES",
                "PERSONAL_INFO",
                "DIALOGUE_STRATEGY",
                "DIALOGUE_STATE",
                "DIALOGUE_STAGE",
                "DIALOGUE"
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
            driver_conversation_utterance_chain=driver_conversation_utterance_chain,
            driver_agent_executor=driver_agent_executor,
            verbose=verbose,
            **kwargs,
        )


