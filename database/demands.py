from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from utils.llm_request_config import openai_api_key, openai_model_name, openai_base_url
import codecs
import json
import sentence_transformers

# 货车宝的运输订单信息
huochebao_order_file = "/Users/chendongdong/Work/service/cnai-pe/PE/agent/huochebao_agent/huochebao_order.txt"


def setup_knowledge_base(order_file: str = None):
    # assume file is in txt or excel format and one order each chunk
    texts = []
    with codecs.open(order_file, 'r', 'utf-8') as reader:
        for line in reader:
            try:
                content = json.loads(line.rstrip())
                texts.append(line.rstrip())
            except Exception as e:
                print(e)
                print(line)

    llm = ChatOpenAI(
        temperature=0,
        model=openai_model_name,
        base_url=openai_base_url,
        api_key=openai_api_key
    )
    # model_name = "sentence-transformers/all-mpnet-base-v2"
    model_name = "/Users/chendongdong/Work/models/embedding/all-mpnet-base-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    docsearch = Chroma.from_texts(
        texts, embeddings, collection_name="huochebao_order"
    )
    knowledge_base = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=docsearch.as_retriever()
    )
    return knowledge_base