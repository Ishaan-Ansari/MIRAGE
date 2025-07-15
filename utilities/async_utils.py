import asyncio
import traceback
from datetime import datetime
from typing import Any, Callable, List, TypeVar

from attrs import define

from logger import loggerUtils as logger

T = TypeVar('T')

@define
class BatchResult:
    """Container for results of batch operation"""
    results: List[Any]
    failed_items: List[tuple[Any, Exception]]
    total_retries: int
    execution_time: float


class BatchProcessingError(Exception):
    """Custom exception for batch processing errors"""
    pass

class NonRetryableError(Exception):
    """Custom exception for non-retryable errors"""
    pass

async def process_async_in_batches(

)->BatchResult:
    pass

async def _process_with_retry(

) -> tuple[Any, int]:
    pass

