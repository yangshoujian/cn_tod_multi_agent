from sanic import Sanic, response
import sys

sys.path.append('.')
import time
import langchain
from langchain_openai import ChatOpenAI
from agents.driver_v2 import DriverGPT
from utils.llm_request_config import whale_api_key, whale_base_url, whale_model_name, \
    openai_base_url, openai_model_name, openai_api_key

langchain.debug=True

app = Sanic("driver_agent")
config = dict(
    use_tools=True,
    huochebao_order_file="/Users/chendongdong/Work/service/cnai-pe/PE/agent/huochebao_agent/huochebao_order.txt",
)
# 展示所有agent
driver_llm = ChatOpenAI(
    model=openai_model_name,
    temperature=0.9,
    base_url=openai_base_url,
    api_key=openai_api_key
)
driver_agent = DriverGPT.from_llm(driver_llm, verbose=False, **config)

@app.route("/")
async def test(request):
    return response.json({"success":True})

@app.route("/request/<query:str>")
async def call_driver_agent(request, query):
    driver_agent.human_step(query)
    # 决定当前的对话状态
    dialogue_state = driver_agent.determine_conversation_state()
    t2 = time.perf_counter()
    _, _, dialogue_stage = driver_agent.determine_conversation_stage()  # optional for demonstration, built into the prompt
    t3 = time.perf_counter()
    # # 决定当前的对话策略
    thought, dialogue_strategy = driver_agent.determine_conversation_strategy()
    t4 = time.perf_counter()
    reply = driver_agent.step(stream=False)["content"]
    return response.json({"success": True, "response": reply})

if __name__ == "__main__":
    app.config['KEEP_ALIVE_TIMEOUT'] = 30
    app.run(host="0.0.0.0", port=2404, debug=True, auto_reload=True, access_log=True, workers=1)