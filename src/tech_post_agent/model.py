import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()


def embedding_model(dim: int) -> OpenAIEmbeddings:
    api_key = os.getenv("OPENAI_API_KEY")
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small", api_key=api_key, dimensions=dim
    )
    return embeddings


def chat_model(model_name: str) -> ChatOpenAI:
    if model_name not in ("gpt-4o-mini", "gpt-4o"):
        raise ValueError("Unsupported model_name. Use 'gpt-4o-mini' or 'gpt-4o'.")

    api_key = os.getenv("OPENAI_API_KEY")
    model = ChatOpenAI(
        model=model_name,
        temperature=0,
        max_retries=1,
        api_key=api_key,
    )
    return model
