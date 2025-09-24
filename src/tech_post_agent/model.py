import os

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

load_dotenv()


def embedding_model(dim: int) -> OpenAIEmbeddings:
    api_key = os.getenv("OPENAI_API_KEY")
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large", api_key=api_key, dimensions=dim
    )
    return embeddings
