from typing import List

from openai import AsyncOpenAI, OpenAI

from config import OPENAI_API_KEY
from constants import DEFAULT_OPENAI_EMBEDDING_MODEL

OPENAI_CLIENT = OpenAI(api_key=OPENAI_API_KEY)
ASYNC_OPENAI_CLIENT = AsyncOpenAI(api_key=OPENAI_API_KEY)

def get_openai_embeddings(texts: List[str], batch_size: int) -> List[List[float]]:
    if not isinstance(texts, list):
        raise ValueError("texts must be a list")

    batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]
    embeddings = []
    for batch in batches:
        response = OPENAI_CLIENT.embeddings.create(
            input=batch,
            model=DEFAULT_OPENAI_EMBEDDING_MODEL
        )
        _embeddings = [response.data[0].embedding for _ in range(len(batch))]
        embeddings.extend(_embeddings)
    return embeddings

async def get_openai_embeddings_async(texts: List[str], batch_size: int) -> List[List[float]]:
    pass
