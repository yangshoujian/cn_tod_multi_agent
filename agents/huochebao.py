from typing import List, Dict, Union, Any
from langchain.agents import AgentExecutor, LLMSingleActionAgent
from langchain.chains import LLMChain
from langchain.chains.base import Chain
from langchain.llms import BaseLLM
from pydantic import Field
from utils.sale_conv_output_parser import SalesConvOutputParser
from utils.custom_prompt_for_tools import CustomPromptTemplateForTools
from utils.constants import SALES_AGENT_TOOLS_PROMPT
from tools.get_tools import search_order_tool
from chains.stage import SellerStageChain
from chains.seller import SellerConversationChain
from chains.driver import DriverConversationChain


class HuoCheBaoSalesGPT(Chain):
    """Controller model for the HuoCheBao Sales Agent."""

    conversation_history: List[str] = []
    current_conversation_stage: str = "1"
    stage_analyzer_chain: SellerStageChain = Field(...)
    sales_conversation_utterance_chain: SellerConversationChain = Field(...)

    sales_agent_executor: Union[AgentExecutor, None] = Field(...)
    use_tools: bool = False

    conversation_stage_dict: Dict = {
        "1": "介绍：通过介绍货车宝运输需求订单的方式开始对话。运输订单信息：出发城市和区，到达的城市和区，装载的货物种类，需要匹配的车型和车长，要求的装货时间",
        "2": "资质确认：通过与货车司机确认车长，车型是否满足要求，装货时间点能否接受，对于运输的货物能否承载，愿不愿意跑提供的线路等方式，确认司机是否适合和愿意接单。",
        "3": "议价：通过与货车司机对话确认司机是否咨询这一单的报酬，若有，则查询外部服务进行报价，并确认司机对当前的报价是否满意，是否要求提价。围绕指定的报价涨幅，尽量与司机成达一致，促进交易的达成。",
        "4": "信息采集：通过与货车司机的对话，分析司机拒绝此单的根本原因，若是车型，运输线路，货品种类，或时间等不匹配引起的原因，则返回具体的原因方便更新司机相关的信息。",
        "5": "异议处理：解决与货车司机对话中的任何异议。准备好提供证据或证明来支持您的说法。",
        "6": "结束对话：完成营销目标，结束对话。"
    }

    salesperson_name: str = "货小宝"
    send_city: str = "北京"
    send_city_dist: str = "海淀区"
    dest_city: str = "上海"
    dest_city_dist: str = "浦东新区"
    goods: str = "电子产品"
    car_length: str = "13米"
    car_type: str = "平板车"
    pickup_time: str = "明天早上8点"
    price: str = "8500"
    percent: str = "8%"
    tool_names: list = []
    driver_agent: DriverConversationChain = None

    def retrieve_conversation_stage(self, key):
        return self.conversation_stage_dict.get(key, "1")

    @property
    def input_keys(self) -> List[str]:
        return []

    @property
    def output_keys(self) -> List[str]:
        return []

    def seed_agent(self, order_info, driver_agent):
        # Step 1: seed the conversation
        self.current_conversation_stage = self.retrieve_conversation_stage("1")
        self.conversation_history = []
        self.salesperson_name = order_info.salesperson_name
        self.send_city = order_info.send_city
        self.send_city_dist = order_info.send_city_dist
        self.dest_city = order_info.dest_city
        self.dest_city_dist = order_info.dest_city_dist
        self.goods = order_info.goods
        self.car_length = order_info.car_length
        self.car_type = order_info.car_type
        self.pickup_time = order_info.pickup_time
        self.price = order_info.price
        self.percent = order_info.percent
        self.driver_agent = driver_agent

    def determine_conversation_stage(self):
        conversation_stage_id = self.stage_analyzer_chain.run(
            conversation_history='"\n"'.join(self.conversation_history),
            current_conversation_stage=self.current_conversation_stage,
        )

        self.current_conversation_stage = self.retrieve_conversation_stage(
            conversation_stage_id
        )

        print(f"Conversation Stage: {self.current_conversation_stage}")
        return conversation_stage_id

    def human_step(self, human_input=""):
        # process human input: 使用机器人模拟人类
        # raw_human_input = self.invoke({
        #     "salesperson_name": self.salesperson_name,
        #     "conversation_stage": self.current_conversation_stage,
        #     "conversation_history": self.conversation_history
        # })['text']
        # 读取输入
        raw_human_input = human_input

        human_input = "司机：" + raw_human_input
        self.conversation_history.append(human_input)
        return raw_human_input

    def step(self):
        return self._call(inputs={})

    def _call(self, inputs: Dict[str, Any]) -> None:
        """Run one step of the sales agent."""
        print(self.use_tools)
        # Generate agent's utterance
        if self.use_tools:
            print(self.percent)
            ai_message = self.sales_agent_executor.run(
                input="",
                conversation_stage=self.current_conversation_stage,
                conversation_history="\n".join(self.conversation_history),
                salesperson_name=self.salesperson_name,
                send_city=self.send_city,
                send_city_dist=self.send_city_dist,
                dest_city=self.dest_city,
                dest_city_dist=self.dest_city_dist,
                goods=self.goods,
                car_length=self.car_length,
                car_type=self.car_type,
                pickup_time=self.pickup_time,
                price=self.price,
                percent=self.percent
            )

        else:
            ai_message = self.sales_conversation_utterance_chain.run(
                conversation_stage=self.current_conversation_stage,
                conversation_history="\n".join(self.conversation_history),
                salesperson_name=self.salesperson_name,
                send_city=self.send_city,
                send_city_dist=self.send_city_dist,
                dest_city=self.dest_city,
                dest_city_dist=self.dest_city_dist,
                goods=self.goods,
                car_length=self.car_length,
                car_type=self.car_type,
                pickup_time=self.pickup_time,
                price=self.price,
                percent=self.percent
            )

        # Add agent's response to conversation history
        print(f"{self.salesperson_name}: ", ai_message.rstrip("<END_OF_TURN>"))
        raw_response = ai_message.rstrip("<END_OF_TURN>")
        agent_name = self.salesperson_name
        ai_message = agent_name + ": " + ai_message
        # if "<END_OF_TURN>" not in ai_message:
        #     ai_message += " <END_OF_TURN>"
        self.conversation_history.append(ai_message)

        return {"content": raw_response}

    @classmethod
    def from_llm(cls, llm: BaseLLM, verbose: bool = False, **kwargs) -> "HuoCheBaoSalesGPT":
        """Initialize the HuoCheBaoSalesGPT Controller."""
        stage_analyzer_chain = SellerStageChain.from_llm(llm, verbose=verbose)

        sales_conversation_utterance_chain = SellerConversationChain.from_llm(
            llm, verbose=verbose
        )

        if "use_tools" in kwargs.keys() and kwargs["use_tools"] is False:
            sales_agent_executor = None

        else:
            order_file = kwargs["huochebao_order_file"]
            tools = search_order_tool(order_file)
            print('tools ', tools)

            prompt = CustomPromptTemplateForTools(
                template=SALES_AGENT_TOOLS_PROMPT,
                tools_getter=lambda x: tools,
                # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
                # This includes the `intermediate_steps` variable because that is needed
                input_variables=[
                    "input",
                    "intermediate_steps",
                    "salesperson_name",
                    "send_city",
                    "send_city_dist",
                    "dest_city",
                    "dest_city_dist",
                    "goods",
                    "car_length",
                    "car_type",
                    "pickup_time",
                    "price",
                    "percent",
                    "conversation_stage",
                    "conversation_history"
                ],
            )
            llm_chain = LLMChain(llm=llm, prompt=prompt, verbose=verbose)

            tool_names = [tool.name for tool in tools]

            # WARNING: this output parser is NOT reliable yet
            ## It makes assumptions about output from LLM which can break and throw an error
            output_parser = SalesConvOutputParser(
                ai_prefix=kwargs["salesperson_name"], verbose=verbose
            )

            sales_agent_with_tools = LLMSingleActionAgent(
                llm_chain=llm_chain,
                output_parser=output_parser,
                stop=["\nObservation:"],
                allowed_tools=tool_names,
                verbose=verbose,
            )

            sales_agent_executor = AgentExecutor.from_agent_and_tools(
                agent=sales_agent_with_tools, tools=tools, verbose=verbose
            )
        return cls(
            stage_analyzer_chain=stage_analyzer_chain,
            sales_conversation_utterance_chain=sales_conversation_utterance_chain,
            sales_agent_executor=sales_agent_executor,
            verbose=verbose,
            **kwargs,
        )