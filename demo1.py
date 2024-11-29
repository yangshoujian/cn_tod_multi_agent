import sys


sys.path.append('.')
import json
import os
import time
import uuid
import sqlite3

import gradio as gr

import modelscope_studio as mgr
import langchain
import json
from langchain_openai import ChatOpenAI
from agents.driver_v2 import DriverGPT
from utils.llm_request_config import whale_api_key, whale_base_url, whale_model_name, \
    openai_base_url, openai_model_name, openai_api_key
from utils.constants import CONVERSATION_STAGES

from path import Path
langchain.debug=True

session_db = "dialog_data.db"

# select your model - we support 50+ LLMs via LiteLLM https://docs.litellm.ai/docs/providers
# llm_policy = ChatWhale(model_name="cainiao_cnnlp_qwen2.5-7B-Instruct")
# llm_generate = ChatWhale(model_name="Qwen2.5-72B-Instruct")
# llm = ChatWhale(model_name="cainiao_cnnlp_qwen2.5-7B-Instruct")
# llm = ChatWhale(model_name="qiaoyun_ai_assistent_qwen2_7B")

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

conversation = None

chatbot = None


def create_connection(db_file):
  conn = None
  try:
    conn = sqlite3.connect(db_file)
    return conn
  except Exception as e:
    print(e)
    return conn


def create_tables():
  cursor = create_connection(session_db)
  cursor.execute(f'''
          CREATE TABLE IF NOT EXISTS dialog_detail (
              id INTEGER PRIMARY KEY,
              scene_id INTEGER , 
              session_id TEXT,
              detail TEXT,
              other TEXT
          )
      ''')


def draw():
  global chatbot
  with gr.Blocks() as demo:
    with gr.Row():
      with gr.Column():
        chatbot = mgr.Chatbot(
          value=conversation,
          height=800,
        )
        input = mgr.MultimodalInput()
        clean = gr.Button(value='重置对话')
      with gr.Column():
        scene_id = gr.Dropdown([("货车司机", 0)], value=0, label="当前场景")
        session_id = mgr.Markdown(label='session id', value=uuid.uuid4().hex)
        dialog_state = mgr.Markdown(label='dialog state', value='')
        sum_table = mgr.Markdown(value='<b>对话信息</b><br/>' + '\n| 订单 | key | value | note |\n|---|---|---|---|\n')
        sw = gr.Dropdown([0, 1], value=0, visible=False)

    input.submit(fn=submit, inputs=[input, chatbot, scene_id, session_id], outputs=[input, chatbot])
    clean.click(fn=clean_dialog, inputs=[chatbot, scene_id, session_id], outputs=[chatbot, session_id])
    chatbot.flushed(fn=flushed, outputs=[input, sw])

    sw.change(fn=need_compute, inputs=[sw, scene_id, session_id, dialog_state, sum_table],
              outputs=[sw, dialog_state, sum_table])

    scene_id.change(fn=clean_dialog, inputs=[chatbot, scene_id, session_id], outputs=[chatbot, session_id])

  demo.queue().launch(server_name="0.0.0.0", server_port=7863, share=False)


def dialog_reset(scene_id):
  global conversation
  global driver_agent
  driver_agent = driver_agent
  driver_agent.seed_agent()
  # driver_agent.determine_conversation_stage()
  # # agent
  # r = driver_agent.step()
  # conversation = [
  #   [
  #     None,
  #     {
  #       # The first message of bot closes the typewriter.
  #       "text": "",
  #       "flushing": False
  #     }
  #   ],
  # ]


def dialog_to_json(_chatbot):
  res = []
  for i in _chatbot:
    usr = i[0]
    if usr is not None:
      usr = usr.text
    bot = i[1]['text'] if isinstance(i[1], dict) else i[1].text
    res.append({'user': usr, 'bot': bot})
  return res


def submit(_input, _chatbot, scene_id, session_id):
  global driver_agent
  global chatbot
  _chatbot.append([_input, None])
  yield gr.update(interactive=False, value=None), _chatbot
  t1 = time.perf_counter()
  driver_agent.human_step(_input.text)
  # 决定当前的对话状态
  dialogue_state = driver_agent.determine_conversation_state()
  t2 = time.perf_counter()
  _, _, dialogue_stage = driver_agent.determine_conversation_stage()  # optional for demonstration, built into the prompt
  t3 = time.perf_counter()
  # # 决定当前的对话策略
  thought, dialogue_strategy = driver_agent.determine_conversation_strategy()
  t4 = time.perf_counter()

  _chatbot[-1][1] = {"text": ''}
  # 非流式
  _chatbot[-1][1] = {"text": f'[状态: {"%.2f" % (t2 - t1)}s] [阶段: {"%.2f" % (t3 - t2)}s] [策略: {"%.2f" % (t4 - t3)}s]' + driver_agent.step(stream=False)["content"]}
  # 流式
  # raw_txt = ''
  # for chunk in driver_agent.step(stream=True):
  #   if not chunk:
      # if len(_chatbot[-1][1]['text']) == 0:
      #   t3 = time.perf_counter()
      #   _chatbot[-1][1]['text'] = f'[A {"%.2f" % (t2 - t1)}s] [B {"%.2f" % (t3 - t2)}s]'
      # _chatbot[-1][1] = {"text": _chatbot[-1][1]['text'] + chunk}#chunk.data['output']['response']}
      # raw_txt = raw_txt + chunk.data['output']['response']
      # r['conversation_stage']
      # 'End conversation: It\'s time to end the call as there is nothing else to be said.'
      # yield {
      #   chatbot: _chatbot,
      # }
  # driver_agent.join_history(raw_txt)
  # t4 = time.perf_counter()
  # _chatbot[-1][1]['text'] = _chatbot[-1][1]['text'] + f' [C {"%.2f" % (t4 - t3)}s]'
  # cursor = create_connection(session_db).cursor()
  # cursor.execute("""
  #       INSERT INTO dialog_detail (scene_id, session_id, detail)
  #       VALUES (?, ?, ?)
  #   """, (scene_id, session_id, json.dumps(dialog_to_json([_chatbot[-1]]), ensure_ascii=False, indent=2)))
  # cursor.connection.commit()

  yield {
    chatbot: _chatbot,
  }


def flushed():
  return gr.update(interactive=True), 1


def clean_dialog(_chatbot, scene_id, session_id):
  global conversation
  dialog_reset(scene_id)
  session_id = uuid.uuid4().hex
  return conversation, session_id


def need_compute(sw, scene_id, session_id, dialog_state, sum_table):
  global driver_agent
  if sw != 0:
    sw = 0
    print(f'{scene_id}, {scene_id}{sum_table}')
    cnt = create_connection(session_db).cursor().execute(
      f'select count(*) from dialog_detail where session_id="{session_id}"', ).fetchone()[0]
    sum_table = f'<b>对话状态</b><br/>\n{driver_agent.dialogue_state}' + \
                f'<b>\n\n对话阶段</b><br/>\n{driver_agent.dialogue_stage}' + \
                f'<b>\n\n对话策略</b><br/>\n{driver_agent.dialogue_strategy}'# + \
                # '<br/>\n<br/>\n<b>对话信息</b><br/>' + '\n| topic | key | value | note |\n|---|---|---|---|\n' + \
                # analysis_compute()
  return sw, dialog_state, sum_table


role_map = {'Bot': 'assistant', 'User': 'user', }


def analysis_compute():
  global driver_agent
  dialog_list = [{'role': role_map[i[0]], 'content': i[1].strip('<END_OF_TURN>').strip('<END_OF_CALL>').strip()} for i
                 in [i.split(':') for i in driver_agent.conversation_history]]
  if len(dialog_list)>0:
    dialog_list = dialog_list[:-1]
  rdict = driver_agent.task_config['analysis_agents'](dialog_list)
  res = ""#DialogueInforCache.parse_data(None, rdict)
  return res


if __name__ == "__main__":
  create_tables()
  dialog_reset(0)
  draw()
