"""Session manager for deep research workflows."""

import structlog
from datetime import datetime, timedelta
from typing import Optional, Tuple
from uuid import uuid4

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.schema import ResearchSessionModel

logger = structlog.get_logger(__name__)


class SessionManager:
    """Manage deep research sessions with multi-chat support.

    Key features:
    - One active session per chat_id (enforced by DB unique constraint)
    - Session resume when returning to a chat with incomplete research
    - Automatic cleanup when switching modes within a chat
    """

    def __init__(self, session_factory):
        """Initialize session manager.

        Args:
            session_factory: AsyncSession factory for database access
        """
        self.session_factory = session_factory

    async def get_or_create_session(
        self, chat_id: str, query: str, mode: str
    ) -> Tuple[ResearchSessionModel, bool]:
        """Get active session for chat or create new one.

        This is the main entry point for multi-chat support. It handles:
        - Resuming an existing active session for the chat
        - Creating a new session if no active session exists

        Args:
            chat_id: Chat identifier
            query: User query (used only if creating new session)
            mode: Research mode (quality/balanced/speed)

        Returns:
            Tuple of (session, is_new) where is_new=True if session was created

        Example:
            session, is_new = await manager.get_or_create_session(
                chat_id="chat_123",
                query="Research quantum computing",
                mode="quality"
            )
            if is_new:
                logger.info("Created new session")
            else:
                logger.info("Resuming session", status=session.status)
        """
        logger.info("🔥 SessionManager.get_or_create_session CALLED",
                   chat_id=chat_id,
                   query=query[:100],
                   mode=mode)

        # Check for existing active session
        active_session = await self.get_active_session(chat_id)

        if active_session:
            logger.info(
                "Found active session for chat",
                session_id=active_session.id,
                chat_id=chat_id,
                status=active_session.status,
            )
            return active_session, False

        # No active session - create new one
        session = await self.create_session(chat_id, query, mode)
        logger.info(
            "Created new session for chat",
            session_id=session.id,
            chat_id=chat_id,
            mode=mode,
        )
        return session, True

    async def get_active_session(
        self, chat_id: str
    ) -> Optional[ResearchSessionModel]:
        """Get active deep_research session for chat_id.

        Active statuses: active, waiting_clarification, researching
        CRITICAL: Only returns sessions that are NOT completed
        This ensures each new deep research in a chat gets a new session

        Args:
            chat_id: Chat identifier

        Returns:
            Active session or None (if all sessions are completed)
        """
        async with self.session_factory() as session:
            result = await session.execute(
                select(ResearchSessionModel)
                .where(
                    ResearchSessionModel.chat_id == chat_id,
                    ResearchSessionModel.status.in_(
                        ["active", "waiting_clarification", "researching"]
                    ),
                    # CRITICAL: Exclude completed sessions
                    ResearchSessionModel.status != "completed",
                )
                .order_by(ResearchSessionModel.created_at.desc())
                .limit(1)
            )
            active = result.scalar_one_or_none()
            
            if active:
                logger.info(
                    "Found active session for chat",
                    session_id=active.id,
                    chat_id=chat_id,
                    status=active.status,
                    note="This session will be reused for continuation"
                )
            else:
                logger.info(
                    "No active session found for chat",
                    chat_id=chat_id,
                    note="All sessions are completed - new session will be created"
                )
            
            return active

    async def create_session(
        self, chat_id: str, query: str, mode: str
    ) -> ResearchSessionModel:
        """Create new research session.

        If there are existing active sessions for this chat, they will be
        marked as 'superseded' to maintain the constraint of one active
        session per chat.

        Args:
            chat_id: Chat identifier
            query: Original user query
            mode: Research mode (quality/balanced/speed)

        Returns:
            Newly created session
        """
        session_id = str(uuid4())

        async with self.session_factory() as session:
            # Mark any existing active sessions as superseded
            await session.execute(
                update(ResearchSessionModel)
                .where(
                    ResearchSessionModel.chat_id == chat_id,
                    ResearchSessionModel.status.in_(
                        ["active", "waiting_clarification", "researching"]
                    ),
                )
                .values(
                    status="superseded",
                    completed_at=datetime.now(),
                    session_metadata=ResearchSessionModel.session_metadata.op("||")(
                        {"superseded_reason": "New session created", "superseded_at": datetime.now().isoformat()}
                    ),
                )
            )

            # Create new session
            new_session = ResearchSessionModel(
                id=session_id,
                chat_id=chat_id,
                original_query=query,
                mode=mode,
                status="active",
            )

            session.add(new_session)
            await session.commit()
            await session.refresh(new_session)

            logger.info(
                "Session created",
                session_id=session_id,
                chat_id=chat_id,
                original_query=query[:100] if query else None,
            )

            return new_session

    async def get_session(self, session_id: str) -> Optional[ResearchSessionModel]:
        """Get session by ID.

        Args:
            session_id: Session identifier

        Returns:
            Session or None if not found
        """
        async with self.session_factory() as session:
            result = await session.execute(
                select(ResearchSessionModel).where(
                    ResearchSessionModel.id == session_id
                )
            )
            return result.scalar_one_or_none()

    async def update_status(self, session_id: str, status: str) -> None:
        """Update session status.

        Valid statuses:
        - active: Session is running
        - waiting_clarification: Waiting for user to answer clarification questions
        - researching: Multi-agent research in progress
        - completed: Research finished successfully
        - superseded: Replaced by a new session in the same chat
        - cancelled: Explicitly cancelled by user
        - expired: Timed out after 24+ hours

        Args:
            session_id: Session identifier
            status: New status
        """
        async with self.session_factory() as session:
            await session.execute(
                update(ResearchSessionModel)
                .where(ResearchSessionModel.id == session_id)
                .values(status=status, updated_at=datetime.now())
            )
            await session.commit()

            logger.info("Session status updated", session_id=session_id, status=status)

    async def complete_session(
        self, session_id: str, final_report: Optional[str] = None
    ) -> None:
        """Mark session as completed.

        Args:
            session_id: Session identifier
            final_report: Optional final report text
        """
        async with self.session_factory() as session:
            values = {
                "status": "completed",
                "completed_at": datetime.now(),
                "updated_at": datetime.now(),
            }
            if final_report is not None:
                values["final_report"] = final_report

            await session.execute(
                update(ResearchSessionModel)
                .where(ResearchSessionModel.id == session_id)
                .values(**values)
            )
            await session.commit()

            logger.info("Session completed", session_id=session_id)

    async def save_deep_search_result(self, session_id: str, result: str) -> None:
        """Save deep search result to session.

        CRITICAL: This must be atomic and synchronous to prevent race conditions.
        The result is saved and committed immediately, then verified.

        Args:
            session_id: Session identifier
            result: Deep search result text (must not be None)
        """
        # CRITICAL: Ensure result is not None - convert None to empty string
        if result is None:
            logger.warning("save_deep_search_result: result is None, converting to empty string",
                         session_id=session_id)
            result = ""
        
        try:
            async with self.session_factory() as session:
                # CRITICAL: Check if session exists before updating
                existing_session = await session.execute(
                    select(ResearchSessionModel).where(ResearchSessionModel.id == session_id)
                )
                session_obj = existing_session.scalar_one_or_none()
                
                if not session_obj:
                    logger.error("❌ save_deep_search_result: Session not found in DB",
                               session_id=session_id,
                               note="Cannot save deep_search_result - session does not exist")
                    return
                
                # CRITICAL: Save result and commit immediately
                session_obj.deep_search_result = result
                session_obj.updated_at = datetime.now()
                await session.commit()
                
                logger.info("✅ save_deep_search_result: Result saved to DB",
                           session_id=session_id,
                           result_length=len(result) if result else 0,
                           note="Result saved and committed")
                
                # CRITICAL: Verify that result was saved by reading it back in a NEW transaction
                # This ensures the save was successful and visible to other transactions
                async with self.session_factory() as verify_session:
                    verification = await verify_session.execute(
                        select(ResearchSessionModel.deep_search_result)
                        .where(ResearchSessionModel.id == session_id)
                    )
                    verified_result = verification.scalar_one_or_none()
                    if verified_result is not None:  # Allow empty string, but not None
                        logger.info("✅ save_deep_search_result: Result saved and VERIFIED in separate transaction",
                                 session_id=session_id,
                                 result_length=len(result) if result else 0,
                                 verified_length=len(verified_result) if verified_result else 0,
                                 note="CRITICAL: Result is now in DB and visible to other transactions. This prevents double execution.")
                    else:
                        logger.error("❌ save_deep_search_result: Result NOT found after save!",
                                   session_id=session_id,
                                   note="CRITICAL ERROR: Save may have failed or transaction not committed properly")
        except Exception as e:
            logger.error("❌ save_deep_search_result: Exception during save",
                       session_id=session_id,
                       error=str(e),
                       error_type=type(e).__name__,
                       exc_info=True,
                       note="CRITICAL ERROR: Failed to save deep_search_result to DB")
            raise

    async def save_clarification_answers(
        self, session_id: str, answers: str
    ) -> None:
        """Save user's clarification answers to session.

        Args:
            session_id: Session identifier
            answers: Clarification answers text
        """
        async with self.session_factory() as session:
            await session.execute(
                update(ResearchSessionModel)
                .where(ResearchSessionModel.id == session_id)
                .values(clarification_answers=answers, updated_at=datetime.now())
            )
            await session.commit()

    async def save_draft_report(self, session_id: str, draft: str) -> None:
        """Save draft report to session.

        Args:
            session_id: Session identifier
            draft: Draft report text
        """
        async with self.session_factory() as session:
            await session.execute(
                update(ResearchSessionModel)
                .where(ResearchSessionModel.id == session_id)
                .values(draft_report=draft, updated_at=datetime.now())
            )
            await session.commit()

    async def save_final_report(self, session_id: str, final_report: str) -> None:
        """Save final report to session.

        Args:
            session_id: Session identifier
            final_report: Final report text
        """
        async with self.session_factory() as session:
            await session.execute(
                update(ResearchSessionModel)
                .where(ResearchSessionModel.id == session_id)
                .values(final_report=final_report, updated_at=datetime.now())
            )
            await session.commit()

            logger.info("Final report saved to session", session_id=session_id)

    async def supersede_active_sessions(self, chat_id: str, reason: str) -> int:
        """Mark all active sessions for chat as superseded.

        This is called when user switches from deep_research to another mode
        within the same chat.

        Args:
            chat_id: Chat identifier
            reason: Reason for superseding (e.g. "User switched to search mode")

        Returns:
            Number of sessions superseded
        """
        async with self.session_factory() as session:
            result = await session.execute(
                update(ResearchSessionModel)
                .where(
                    ResearchSessionModel.chat_id == chat_id,
                    ResearchSessionModel.status.in_(
                        ["active", "waiting_clarification", "researching"]
                    ),
                )
                .values(
                    status="superseded",
                    completed_at=datetime.now(),
                    updated_at=datetime.now(),
                    metadata=ResearchSessionModel.metadata.op("||")(
                        {"superseded_reason": reason, "superseded_at": datetime.now().isoformat()}
                    ),
                )
            )
            await session.commit()

            count = result.rowcount
            if count > 0:
                logger.info(
                    "Superseded active sessions",
                    chat_id=chat_id,
                    count=count,
                    reason=reason,
                )

            return count

    async def cleanup_expired_sessions(self, hours: int = 24) -> int:
        """Mark old incomplete sessions as expired.

        This is a background task that should run periodically (e.g. hourly).
        Sessions in active states that are older than N hours are marked as expired.

        Args:
            hours: Age threshold in hours (default 24)

        Returns:
            Number of sessions expired
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)

        async with self.session_factory() as session:
            result = await session.execute(
                update(ResearchSessionModel)
                .where(
                    ResearchSessionModel.created_at < cutoff_time,
                    ResearchSessionModel.status.in_(
                        ["active", "waiting_clarification", "researching"]
                    ),
                )
                .values(
                    status="expired",
                    completed_at=datetime.now(),
                    updated_at=datetime.now(),
                    metadata=ResearchSessionModel.metadata.op("||")(
                        {"expired_at": datetime.now().isoformat(), "expiry_hours": hours}
                    ),
                )
            )
            await session.commit()

            count = result.rowcount
            if count > 0:
                logger.info("Expired old sessions", count=count, hours=hours)

            return count
