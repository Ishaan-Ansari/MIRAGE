from typing import List

from utilities.async_utils import process_async_in_batches

DEFAULT_CHUNK_SIZE = 512    # adjust based on token limit
DEFAULT_OVERLAP_RATIO = 0.1 # 10% overlap change it accordingly


class PipelineDocumentChunker:
    @classmethod
    def create_chunks(
            cls,
            text: str,
            chunk_size: int = DEFAULT_CHUNK_SIZE,
            overlap_ratio: float = DEFAULT_OVERLAP_RATIO,
    ):
        """
        Splits the document into chunks, allowing the overlap between chunks for better context.
        If the document is small, It will have smaller number of chunks instead of a high number.
        """
        words = text.split()
        total_words = len(words)

        if total_words <= chunk_size:
            return [
                text
            ]  # return the full text as a single chunk if it's smaller than chunk_size

        overlap_size = int(chunk_size * overlap_ratio)
        step_size = chunk_size - overlap_size

        chunks = [
            " ".join(words[i : i + chunk_size])
            for i in range(0, total_words, step_size)
        ]

        return chunks

    async def create_chunks_multiple(
            cls,
            text: str,
            batch_size: int = 250,
            chunk_size: int = DEFAULT_CHUNK_SIZE,
            overlap_ratio: float = DEFAULT_OVERLAP_RATIO,
    ):
        chunked_text = await process_async_in_batches(
            func = cls.create_chunks,
            items = text,
            batch_size = batch_size,
            params_as_kwargs = True,
            chunk_size = chunk_size,
            overlap_ratio = overlap_ratio,
        )

        return chunked_text
