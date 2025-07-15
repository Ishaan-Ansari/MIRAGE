from typing import List

from utilities.async_utils import process_in_async_batches

DEFAULT_CHUNK_SIZE = 512    # adjust based on token limit
DEFAULT_OVERLAP_RATIO = 0.1 # 10% overlap change it accordingly

