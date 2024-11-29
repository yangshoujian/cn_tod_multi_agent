from langchain.chains import LLMChain
from langchain.llms import BaseLLM
from langchain.prompts import PromptTemplate
from prompts.driver_v2 import STATE_PROMPT_TEMPLATE


class DriverStateChain(LLMChain):
    """Chain to generate the next utterance for the conversation."""

    @classmethod
    def from_llm(cls, llm: BaseLLM, verbose: bool = True) -> LLMChain:
        """Get the response parser."""

        prompt = PromptTemplate(
            template=STATE_PROMPT_TEMPLATE,
            input_variables=[
                "DIALOGUE",
                "DIALOGUE_STATES",
                "DS_RESPONSE_FORMAT",
                "MEMORY"
            ],
        )
        return cls(prompt=prompt, llm=llm, verbose=verbose)