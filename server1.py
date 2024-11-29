from flask import Flask, request, jsonify, render_template
import random
import langchain
import json
from langchain_openai import ChatOpenAI
from agents.driver_v2 import DriverGPT
from utils.llm_request_config import whale_api_key, whale_base_url, whale_model_name, \
    openai_base_url, openai_model_name, openai_api_key
from utils.constants import CONVERSATION_STAGES

from path import Path

ROOT = Path(__file__).dirname().parent

app = Flask(__name__)
driver_agent: DriverGPT = None

langchain.debug=True

class JSONSerializable:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __str__(self):
        attributes = ", ".join(f"{k}={v}" for k, v in self.__dict__.items())
        # return f"{self.__class__.__name__}({attributes})"
        return json.dumps(dict(self.__dict__.items()), ensure_ascii=False)


def _init():
    global driver_agent
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



@app.route('/')
def home():
    _init()
    return render_template('/index.html')


@app.route('/ask', methods=['POST'])
def ask():
    # 获取用户问题
    user_question = request.json.get('question')

    # 存储到对话历史中
    driver_agent.human_step(user_question)

    # # 决定当前的对话状态
    # dialogue_state = driver_agent.determine_conversation_state()
    #
    # # # 决定当前的对话阶段
    # _, _, dialogue_stage= driver_agent.determine_conversation_stage()
    #
    # # # 决定当前的对话策略
    # thought, dialogue_strategy = driver_agent.determine_conversation_strategy()

    # # 生成回复
    # answer = driver_agent.step()

    # 流式回复
    raw_txt = ""
    for chunk in driver_agent.step(stream=True):
        print("流式输出:" , chunk)
        raw_txt = raw_txt + chunk.data['output']['response']
    driver_agent.join_history(raw_txt)

    thought = "\n\n".join([f"dialogue_state: {dialogue_state}", f"dialogue_stage: {dialogue_stage}", f"thought: {thought}", f"dialogue_strategy: {dialogue_strategy}"])
    # #
    # thought = "... ..."
    return jsonify({"thought": "--------thought----------" + "\n" + thought, "answer": "--------answer----------" + "\n" + "司机: " + raw_txt})


if __name__ == '__main__':
    app.run(port=5001)  # 可修改为您想要的端口
