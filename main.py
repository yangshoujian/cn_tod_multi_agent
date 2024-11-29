import json
from lib2to3.pgen2.driver import Driver

from langchain_openai import ChatOpenAI
from agents.driver import DriverGPT
from chains.driver import DriverConversationChain
from utils.llm_request_config import whale_api_key, whale_base_url, whale_model_name, \
    openai_base_url, openai_model_name, openai_api_key
from utils.constants import CONVERSATION_STAGES


class JSONSerializable:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __str__(self):
        attributes = ", ".join(f"{k}={v}" for k, v in self.__dict__.items())
        # return f"{self.__class__.__name__}({attributes})"
        return json.dumps(dict(self.__dict__.items()), ensure_ascii=False)


def main():
    config = dict(
        salesperson_name="货小宝",
        send_city="北京",
        send_city_dist="海淀区",
        dest_city="上海",
        dest_city_dist="浦东新区",
        goods="电子产品",
        car_length="13米",
        car_type="平板车",
        pickup_time="明天早上8点",
        price="8500",
        percent="8%",
        conversation_history=[],
        conversation_stage=CONVERSATION_STAGES.get(
            "1",
            "介绍：通过介绍货车宝运输需求订单的方式开始对话。运输订单信息：出发城市和区，到达的城市和区，装载的货物种类，需要匹配的车型和车长，要求的装货时间",
        ),
        use_tools=True,
        huochebao_order_file="/Users/chendongdong/Work/service/cnai-pe/PE/agent/huochebao_agent/huochebao_order.txt",
    )
    driver_llm = ChatOpenAI(
        model=openai_model_name,
        temperature=0.9,
        base_url=openai_base_url,
        api_key=openai_api_key
    )
    driver_agent = DriverGPT.from_llm(driver_llm, verbose=False, **config)

    driver_agent.seed_agent()
    driver_agent.human_step()
    driver_agent.determine_conversation_stage()
    driver_agent.step()



if __name__ == "__main__":
    main()
