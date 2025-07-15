from typing import List

from utilities.embeddings import get_openai_embeddings

class PipelineEmbeddings:
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        return get_openai_embeddings(texts, batch_size=10)