from langchain.chains import LLMChain
from langchain.llms import BaseLLM
from langchain.prompts import PromptTemplate
from utils.constants import SALES_AGENT_INCEPTION_PROMPT


class SellerConversationChain(LLMChain):
    """Chain to generate the next utterance for the conversation."""

    @classmethod
    def from_llm(cls, llm: BaseLLM, verbose: bool = True) -> LLMChain:
        """Get the response parser."""

        prompt = PromptTemplate(
            template=SALES_AGENT_INCEPTION_PROMPT,
            input_variables=[
                "salesperson_name",
                "conversation_stage",
                "conversation_history",
            ],
        )
        print(prompt)
        return cls(prompt=prompt, llm=llm, verbose=verbose)