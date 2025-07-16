import asyncio
import traceback
from datetime import datetime, time
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

    start = datetime.now()

    results = []
    failed_items = []
    total_retries = 0

    # process the items in batches
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        batch_tasks = []

        # create tasks for current batch
        for item in batch:
            task = _process_with_retry(
                func=func,
                item=item,
                num_retries=num_retries,
                retry_delay=retry_delay,
                params_as_kwargs=params_as_kwargs,
                args=args,
                kwargs=kwargs,
            )
            batch_tasks.append(task)

        # wait for all task in current batch to complete
        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

        for item, result in zip(batch, batch_results):
            if isinstance(result, NonRetryableError):
                raise result

            elif isinstance(result, Exception):
                results.append(None)
                failed_items.append((item, result))
                logger.error(f'Failed to process the item: {result}')

            elif isinstance(result, tuple) and len(result)==2:
                results_value, retries = result
                results.append(results_value)
                total_retries += retries

            else:
                logger.error(f'Unexpected result format for the item: {item}')
                results.append(None)
                failed_items.append(
                    (item, BatchProcessingError("Unexpected result format")),
                )

            assert len(results) == len(items)

            logger.info(
                f'Processed batch {i // batch_size + 1}/{len(items)}/'
                f"{(len(items) + batch_size - 1) // batch_size}"
            )

    execution_time = (datetime.now() - start).total_seconds()

    logger.info(
        f"Batch processing completed. "
        f"Successful: {len(results) - len(failed_items)}. "
        f"Failed items: {len(failed_items)}. "
        f"Total retries: {total_retries}. "
        f"Execution time: {execution_time:.2f}s."
    )

    return BatchResult(
        results=results,
        failed_items=failed_items,
        total_retries=total_retries,
        execution_time=execution_time,
    )


async def _process_with_retry(
        func: Callable[..., T],
        item: Any,
        num_retries: int,
        retry_delay: float,
        params_as_kwargs: bool,
        *args,
        **kwargs,
) -> tuple[Any, int]:
    """
    Process a single item with retry logic
    return tuple of (result, number of retries used)

    """
