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
        func: Callable[..., T],
        items: List[Any],
        batch_size: int = 100,
        num_retries: int = 3,
        retry_delay: float = 1.0,
        params_as_kwargs: bool = False,
        *args,
        **kwargs,
)->BatchResult:
    """
    Prcess items in batches with retry logic

    Args:
    func (Callable[...,T]): Function to execute
    items (List[Any]): List of items to process
    batch_size (int, optional): Number of items per batch. Defaults to 100
    num_retries (int, optional): Number of retries. Defaults to 3
    retry_delay (float, optional): Delay between retries ins seconds
    *args (Any): Additional positional arguments to pass to func
    **kwargs (Any): Additional keyword arguments to pass to func

    Returns:
    BatchResult: Batch result containing results and failed items
    """
    logger.info(f'Starting batch processing of {len(items)} items with batch size {batch_size}')


async def _process_with_retry(

) -> tuple[Any, int]:
    pass
