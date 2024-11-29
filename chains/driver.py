from langchain.chains import LLMChain
from langchain.llms import BaseLLM
from langchain.prompts import PromptTemplate
from prompts.driver_v2 import DRIVER_PROMPT_TEMPLATE

class DriverConversationChain(LLMChain):
    """Chain to generate the next utterance for the conversation."""

    @classmethod
    def from_llm(cls, llm: BaseLLM, verbose: bool = True) -> LLMChain:
        """Get the response parser."""

        prompt = PromptTemplate(
            template=DRIVER_PROMPT_TEMPLATE,
            input_variables=[
                "PERSONAL_INFO",
                "DIALOGUE_STRATEGY",
                "DIALOGUE_STATE",
                "DIALOGUE"
            ],
        )
        return cls(prompt=prompt, llm=llm, verbose=verbose)