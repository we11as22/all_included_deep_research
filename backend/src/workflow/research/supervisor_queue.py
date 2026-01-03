"""Supervisor queue system for handling parallel agent completions."""

import asyncio
from typing import Any, Dict, Optional
import structlog

logger = structlog.get_logger(__name__)


class AgentCompletionEvent:
    """Event when agent completes a task."""

    def __init__(self, agent_id: str, task_title: str, result: Dict[str, Any]):
        self.agent_id = agent_id
        self.task_title = task_title
        self.result = result
        self.timestamp = asyncio.get_event_loop().time()


class SupervisorQueue:
    """Queue system for supervisor to handle agent task completions sequentially."""

    def __init__(self):
        self.queue: asyncio.Queue[AgentCompletionEvent] = asyncio.Queue()
        self.processing = False
        self.lock = asyncio.Lock()
        self._processor_task: Optional[asyncio.Task] = None

    async def agent_completed_task(self, agent_id: str, task_title: str, result: Dict[str, Any]):
        """
        Agent signals that it completed a task and needs supervisor review.

        Args:
            agent_id: Agent identifier
            task_title: Completed task title
            result: Task result data
        """
        event = AgentCompletionEvent(agent_id, task_title, result)
        await self.queue.put(event)
        logger.info(
            "Agent task completion queued for supervisor",
            agent_id=agent_id,
            task_title=task_title,
            queue_size=self.queue.qsize()
        )

    async def start_processing(self, supervisor_callback):
        """
        Start processing queue with supervisor callback.

        Args:
            supervisor_callback: Async function(event) -> bool
                Returns True if agent should continue, False if should stop
        """
        async with self.lock:
            if self.processing:
                logger.warning("Supervisor queue already processing")
                return

            self.processing = True

        logger.info("Supervisor queue processing started")

        try:
            while True:
                try:
                    # Wait for agent completion event
                    event = await asyncio.wait_for(self.queue.get(), timeout=1.0)

                    logger.info(
                        "Supervisor processing agent completion",
                        agent_id=event.agent_id,
                        task=event.task_title,
                        queue_remaining=self.queue.qsize()
                    )

                    # Call supervisor to review
                    should_continue = await supervisor_callback(event)

                    self.queue.task_done()

                    if not should_continue:
                        logger.info("Supervisor signaled to stop processing")
                        break

                except asyncio.TimeoutError:
                    # Check if we should stop (no more work)
                    if self.queue.empty() and not self.processing:
                        break
                    continue

                except Exception as e:
                    logger.error("Error processing agent completion", error=str(e), exc_info=True)
                    self.queue.task_done()

        finally:
            async with self.lock:
                self.processing = False
            logger.info("Supervisor queue processing stopped")

    async def stop_processing(self):
        """Stop processing queue."""
        async with self.lock:
            self.processing = False
        logger.info("Supervisor queue stop requested")

    async def wait_for_empty(self, timeout: float = 30.0):
        """
        Wait for queue to be empty.

        Args:
            timeout: Max time to wait in seconds
        """
        try:
            await asyncio.wait_for(self.queue.join(), timeout=timeout)
            logger.info("Supervisor queue is empty")
        except asyncio.TimeoutError:
            logger.warning("Timeout waiting for supervisor queue to empty", queue_size=self.queue.qsize())

    def is_processing(self) -> bool:
        """Check if queue is currently being processed."""
        return self.processing

    def size(self) -> int:
        """Get current queue size."""
        return self.queue.qsize()
