import random

from database.demands import setup_knowledge_base
from langchain.agents import Tool
from langchain.pydantic_v1 import BaseModel, Field
from typing import Optional, Union
from langchain.tools import BaseTool
import math
import json


class CalculatePrice(BaseModel):
    base_price: Optional[Union[float,int]] = Field(description="基础定价")
    raise_percent: Optional[Union[int, float]] = Field(description="价格涨幅")


def price_calculator(base_price, raise_percent):
    print("base price ", base_price, raise_percent)
    return math.floor(base_price*(1+raise_percent))


class CalculatePrice2(BaseTool):
    name = "计算货单价格"
    description = """当你需要根据涨幅计算运单价格，使用这个工具。
    使用这个工具，你必须提供2个参数
    'base_price', 'raise_percent'
    示例：{"base_price":1000, "raise_percent":0.1}"""

    def _run(self,
             input: str,
             # base_price: Optional[Union[float,int]],
             # raise_percent: Optional[Union[int, float]]
             ):
        data = json.loads(input)
        base_price = data["base_price"]
        raise_percent = data["raise_percent"]
        return math.floor(base_price * (1 + raise_percent))

    def _arun(self, query):
        raise NotImplementedError("This tool does not support async")


class CalculateTargetPrice(BaseTool):
    name = "计算运单最高期望价格"
    description = """当你需要计算运单最高期望价格的时候，使用这个工具。
    使用这个工具，你必须提供1个参数
    'base_price': 销售首次提出的价格
    示例：{"base_price":xxx}"""

    def _run(self,
             input: str,
             # base_price: Optional[Union[float,int]],
             # raise_percent: Optional[Union[int, float]]
             ):
        data = json.loads(input)
        base_price = data["base_price"]
        raise_percent = 0.3
        return base_price + max(500, math.floor(base_price * raise_percent))

    def _arun(self, query):
        raise NotImplementedError("This tool does not support async")


class CalculateMinPrice(BaseTool):
    name = "计算运单最低接受价格"
    description = """当你需要计算运单最低接受价格的时候，使用这个工具。
    使用这个工具，你必须提供1个参数
    'base_price': 销售首次提出的价格
    示例：{"base_price":xxx}"""

    def _run(self,
             input: str,
             # base_price: Optional[Union[float,int]],
             # raise_percent: Optional[Union[int, float]]
             ):
        data = json.loads(input)
        base_price = data["base_price"]
        raise_percent = 0.1
        return base_price + min(200, math.floor(base_price * raise_percent))

    def _arun(self, query):
        raise NotImplementedError("This tool does not support async")

def search_order_tool():
    # query to get_tools can be used to be embedded and relevant tools found
    # see here: https://langchain-langchain.vercel.app/docs/use_cases/agents/custom_agent_with_plugin_retrieval#tool-retriever

    # we only use one tool for now, but this is highly extensible!
    tools = [
        CalculateMinPrice(),
        CalculateTargetPrice()
    ]
    return tools

if __name__ == "__main__":
    test = CalculatePrice2()
    print(test.run("""{"base_price": 8500, "raise_percent": 0.08}"""))