"""Supervisor call queue for handling concurrent agent completions.

When multiple agents finish simultaneously, they queue supervisor calls
to avoid race conditions and enable batch processing.
"""

import asyncio
import time
from collections import deque
from typing import Any, Dict, List

import structlog

logger = structlog.get_logger(__name__)


class SupervisorQueue:
    """Queue for supervisor wake-up calls from agents.

    Handles concurrent agent completions by queueing calls and
    processing them in batches for efficiency.
    """

    def __init__(self):
        """Initialize supervisor queue."""
        self.queue: deque = deque()
        self.lock: asyncio.Lock = asyncio.Lock()
        self.processing: bool = False

    async def enqueue(self, agent_id: str, action: str, result: Any):
        """
        Agent reports completion and requests supervisor review.

        Args:
            agent_id: ID of the agent
            action: Action that was completed
            result: Result of the action
        """
        async with self.lock:
            self.queue.append({
                "agent_id": agent_id,
                "action": action,
                "result": result,
                "timestamp": time.time(),
            })
            logger.debug(
                "Supervisor call queued",
                agent_id=agent_id,
                action=action,
                queue_length=len(self.queue)
            )

    async def process_batch(
        self,
        state: Dict[str, Any],
        supervisor_func: Any,
        max_batch_size: int = 10,
    ) -> List[Dict]:
        """
        Process all queued supervisor calls in batch.

        This is more efficient than waking the supervisor for each individual
        agent action. The supervisor reviews ALL agent updates at once.

        Args:
            state: Current research state
            supervisor_func: Supervisor ReAct function
            max_batch_size: Maximum calls to process at once

        Returns:
            List of supervisor directive dicts
        """
        async with self.lock:
            if not self.queue:
                return []

            # Extract batch
            batch_size = min(len(self.queue), max_batch_size)
            batch = []
            for _ in range(batch_size):
                if self.queue:
                    batch.append(self.queue.popleft())

        if not batch:
            return []

        logger.info(
            "Processing supervisor call batch",
            batch_size=len(batch),
            remaining_queue=len(self.queue)
        )

        # Supervisor reviews ALL agent actions at once
        try:
            supervisor_result = await supervisor_func(state, batch)
            logger.debug(
                "Supervisor batch processed",
                directives=len(supervisor_result.get("directives", []))
            )
            return supervisor_result

        except Exception as e:
            logger.error("Supervisor batch processing failed", error=str(e), exc_info=True)
            return {"directives": [], "should_continue": True}

    async def wait_for_batch(self, min_batch_size: int = 1, timeout: float = 5.0):
        """
        Wait for queue to reach minimum batch size or timeout.

        Useful for grouping agent completions that happen close together.

        Args:
            min_batch_size: Minimum queue size to wait for
            timeout: Maximum wait time in seconds
        """
        start_time = time.time()

        while len(self.queue) < min_batch_size:
            if time.time() - start_time > timeout:
                break
            await asyncio.sleep(0.1)

        logger.debug(
            "Batch wait complete",
            queue_size=len(self.queue),
            elapsed=time.time() - start_time
        )

    def clear(self):
        """Clear the queue (called when research is complete)."""
        async def _clear():
            async with self.lock:
                cleared_count = len(self.queue)
                self.queue.clear()
                logger.info("Supervisor queue cleared", cleared_count=cleared_count)

        return _clear()

    def size(self) -> int:
        """Get current queue size."""
        return len(self.queue)

    def is_empty(self) -> bool:
        """Check if queue is empty."""
        return len(self.queue) == 0
    
    async def agent_completed_task(self, agent_id: str, task_title: str, result: Any):
        """
        Agent reports task completion and requests supervisor review.
        
        Args:
            agent_id: ID of the agent
            task_title: Title of the completed task
            result: Finding result from the agent
        """
        await self.enqueue(agent_id, "task_completed", {
            "task_title": task_title,
            "result": result
        })
        logger.info(
            "Agent task completion queued for supervisor",
            agent_id=agent_id,
            queue_size=len(self.queue),
            task_title=task_title
        )


# ==================== Global Queue Instance ==========


# Singleton pattern for supervisor queue
_global_supervisor_queue: Dict[str, SupervisorQueue] = {}


def get_supervisor_queue(session_id: str) -> SupervisorQueue:
    """Get or create supervisor queue for a session."""
    if session_id not in _global_supervisor_queue:
        _global_supervisor_queue[session_id] = SupervisorQueue()
        logger.debug(f"Created supervisor queue for session: {session_id}")

    return _global_supervisor_queue[session_id]


def cleanup_supervisor_queue(session_id: str):
    """Clean up supervisor queue for a session."""
    if session_id in _global_supervisor_queue:
        del _global_supervisor_queue[session_id]
        logger.debug(f"Cleaned up supervisor queue for session: {session_id}")
