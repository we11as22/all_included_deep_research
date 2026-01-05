"""E2E tests for REST pagination and Socket.IO streaming."""

import asyncio
import os
from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest
import socketio

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
SOCKETIO_PATH = os.getenv("SOCKETIO_PATH", "socket.io")


async def wait_for_backend(client: httpx.AsyncClient) -> None:
    for _ in range(30):
        try:
            response = await client.get("/health", timeout=2.0)
            if response.status_code == 200:
                return
        except Exception:
            pass
        await asyncio.sleep(1)
    pytest.fail(f"Backend not reachable at {API_BASE_URL}")


async def create_chat(client: httpx.AsyncClient, title: str) -> str:
    response = await client.post("/api/chats", json={"title": title})
    assert response.status_code == 200, response.text
    return response.json()["id"]


async def add_message(
    client: httpx.AsyncClient,
    chat_id: str,
    role: str,
    content: str,
    message_id: str | None = None,
) -> dict:
    params = {"role": role, "content": content}
    if message_id:
        params["message_id"] = message_id
    response = await client.post(f"/api/chats/{chat_id}/messages", params=params)
    assert response.status_code == 200, response.text
    return response.json()


@pytest.mark.asyncio
async def test_rest_chat_crud_and_history_pagination():
    async with httpx.AsyncClient(base_url=API_BASE_URL, timeout=30.0) as client:
        await wait_for_backend(client)

        chat_id = await create_chat(client, f"e2e-pagination-{uuid4().hex[:8]}")

        try:
            await add_message(client, chat_id, "user", "Hello")
            await add_message(client, chat_id, "assistant", "Hi there")
            await add_message(client, chat_id, "user", "What is 2+2?")

            messages_page = await client.get(
                f"/api/chats/{chat_id}/messages",
                params={"limit": 2, "offset": 0, "order": "asc"},
            )
            assert messages_page.status_code == 200, messages_page.text
            page_data = messages_page.json()
            assert page_data["pagination"]["limit"] == 2
            assert page_data["pagination"]["offset"] == 0
            assert page_data["pagination"]["has_more"] is True
            assert len(page_data["messages"]) == 2

            second_page = await client.get(
                f"/api/chats/{chat_id}/messages",
                params={"limit": 2, "offset": 2, "order": "asc"},
            )
            assert second_page.status_code == 200, second_page.text
            second_data = second_page.json()
            assert second_data["pagination"]["has_more"] is False
            assert len(second_data["messages"]) == 1

            chat_with_pagination = await client.get(
                f"/api/chats/{chat_id}",
                params={"limit": 1, "offset": 0},
            )
            assert chat_with_pagination.status_code == 200, chat_with_pagination.text
            chat_page = chat_with_pagination.json()
            assert chat_page["pagination"]["limit"] == 1
            assert chat_page["pagination"]["offset"] == 0
            assert len(chat_page["messages"]) == 1

            chat_without_pagination = await client.get(f"/api/chats/{chat_id}")
            assert chat_without_pagination.status_code == 200, chat_without_pagination.text
            chat_full = chat_without_pagination.json()
            assert chat_full["pagination"] is None
            assert len(chat_full["messages"]) >= 3
        finally:
            await client.delete(f"/api/chats/{chat_id}")


@pytest.mark.asyncio
async def test_rest_list_chats_pagination_consistency():
    async with httpx.AsyncClient(base_url=API_BASE_URL, timeout=30.0) as client:
        await wait_for_backend(client)

        response = await client.get("/api/chats", params={"limit": 1, "offset": 0})
        assert response.status_code == 200, response.text
        payload = response.json()

        pagination = payload["pagination"]
        chats = payload["chats"]
        assert pagination["limit"] == 1
        assert pagination["offset"] == 0
        assert pagination["total"] >= len(chats)
        assert pagination["has_more"] == (pagination["offset"] + len(chats) < pagination["total"])


@pytest.mark.asyncio
async def test_rest_config_endpoint():
    async with httpx.AsyncClient(base_url=API_BASE_URL, timeout=30.0) as client:
        await wait_for_backend(client)

        response = await client.get("/api/config")
        assert response.status_code == 200, response.text
        data = response.json()

        assert "search_provider" in data
        assert "embedding_provider" in data
        assert "speed_max_iterations" in data
        assert "balanced_max_iterations" in data
        assert "quality_max_iterations" in data


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["search", "deep_search"])
async def test_socketio_streaming_search_mode(mode: str):
    async with httpx.AsyncClient(base_url=API_BASE_URL, timeout=30.0) as client:
        await wait_for_backend(client)
        chat_id = await create_chat(client, f"e2e-socket-{uuid4().hex[:8]}")

    stream_events: list[tuple[str, dict]] = []
    done_event = asyncio.Event()
    error_event = asyncio.Event()
    error_payload = SimpleNamespace(value=None)

    sio = socketio.AsyncClient(reconnection=False, logger=False, engineio_logger=False)

    @sio.on("stream:init")
    async def on_init(data):
        stream_events.append(("init", data))

    @sio.on("stream:report_chunk")
    async def on_chunk(data):
        stream_events.append(("report_chunk", data))

    @sio.on("stream:final_report")
    async def on_final(data):
        stream_events.append(("final_report", data))

    @sio.on("stream:error")
    async def on_error(data):
        stream_events.append(("error", data))
        error_payload.value = data
        error_event.set()

    @sio.on("stream:done")
    async def on_done(data):
        stream_events.append(("done", data))
        done_event.set()

    await sio.connect(API_BASE_URL, socketio_path=SOCKETIO_PATH)

    try:
        response = await sio.call(
            "chat:send",
            {
                "chatId": chat_id,
                "message": "What is 2+2?",
                "mode": mode,
                "messageId": f"msg-{uuid4().hex[:8]}",
            },
            timeout=10,
        )
        assert response.get("success") is True

        await asyncio.wait_for(done_event.wait(), timeout=120)

        assert any(event[0] == "init" for event in stream_events)
        assert any(event[0] in {"report_chunk", "final_report"} for event in stream_events)
        assert not error_event.is_set(), f"Streaming error: {error_payload.value}"
    finally:
        await sio.disconnect()
        async with httpx.AsyncClient(base_url=API_BASE_URL, timeout=30.0) as client:
            await client.delete(f"/api/chats/{chat_id}")
