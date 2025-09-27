import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()


def embedding_model(dim: int) -> OpenAIEmbeddings:
    api_key = os.getenv("OPENAI_API_KEY")
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large", api_key=api_key, dimensions=dim
    )
    return embeddings


def chat_model() -> ChatOpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    model = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        max_retries=1,
        api_key=api_key,
    )
    return model
