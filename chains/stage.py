from langchain.chains import LLMChain
from langchain.llms import BaseLLM
from langchain.prompts import PromptTemplate
from utils.constants import HUOCHEBAO_STAGE_PROMPT_TEMPLATE
from prompts.driver_v2 import STAGE_PROMPT_TEMPLATE


class SellerStageChain(LLMChain):
    """Chain to generate the next utterance for the conversation."""

    @classmethod
    def from_llm(cls, llm: BaseLLM, verbose: bool = True) -> LLMChain:
        """Get the response parser."""

        prompt = PromptTemplate(
            template=HUOCHEBAO_STAGE_PROMPT_TEMPLATE,
            input_variables=[
                "salesperson_name",
                "conversation_stage",
                "conversation_history",
            ],
        )
        return cls(prompt=prompt, llm=llm, verbose=verbose)


class DriverStageChain(LLMChain):
    """Chain to generate the next utterance for the conversation."""

    @classmethod
    def from_llm(cls, llm: BaseLLM, verbose: bool = True) -> LLMChain:
        """Get the response parser."""

        prompt = PromptTemplate(
            template=STAGE_PROMPT_TEMPLATE,
            input_variables=[
                "PERSONAL_INFO",
                "DIALOGUE",
                "DIALOGUE_STAGES",
                "LAST_DIALOGUE_STAGE",
                "LAST_DIALOGUE_STATE",
                "STATE_RESPONSE_FORMAT",
                "STATE_RESPONSE_EXAMPLE",
                "DIALOGUE_STATES",
                "DIALOGUE_STATE_TRANSFER",

            ],
        )
        return cls(prompt=prompt, llm=llm, verbose=verbose)