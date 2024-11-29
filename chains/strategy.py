from langchain.chains import LLMChain
from langchain.llms import BaseLLM
from langchain.prompts import PromptTemplate
from utils.constants import HUOCHEBAO_STAGE_PROMPT_TEMPLATE
from prompts.driver_v2 import STRATEGY_PROMPT_TEMPLATE


class DriverStrategyChain(LLMChain):
    """Chain to generate the next utterance for the conversation."""

    @classmethod
    def from_llm(cls, llm: BaseLLM, verbose: bool = True) -> LLMChain:
        """Get the response parser."""

        prompt = PromptTemplate(
            template=STRATEGY_PROMPT_TEMPLATE,
            input_variables=[
                "DIALOGUE_STRATEGY",
                "PERSONAL_INFO",
                "DIALOGUE",
                "DIALOGUE_STATE",
                "STRATEGY_RESPONSE_FORMAT",
                "DIALOGUE_STAGE",
                "DIALOGUE_STATES",
            ],
        )
        return cls(prompt=prompt, llm=llm, verbose=verbose)