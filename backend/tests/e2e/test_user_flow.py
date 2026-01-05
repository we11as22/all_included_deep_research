"""User-facing flow tests (REST + Socket.IO)."""

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


@pytest.mark.asyncio
async def test_user_flow_create_chat_stream_and_search():
    async with httpx.AsyncClient(base_url=API_BASE_URL, timeout=30.0) as client:
        await wait_for_backend(client)

        payload = {
            "title": "User flow chat",
            "message": "Hello! This is a test message for search.",
            "message_id": f"msg-{uuid4().hex[:8]}",
        }
        response = await client.post("/api/chats/create-with-message", json=payload)
        assert response.status_code == 200, response.text
        data = response.json()
        chat_id = data["chat"]["id"]

    stream_events: list[tuple[str, dict]] = []
    done_event = asyncio.Event()
    error_event = asyncio.Event()
    error_payload = SimpleNamespace(value=None)
    report_chunks: list[str] = []
    final_report = SimpleNamespace(value=None)

    sio = socketio.AsyncClient(reconnection=False, logger=False, engineio_logger=False)

    @sio.on("stream:init")
    async def on_init(data):
        stream_events.append(("init", data))

    @sio.on("stream:report_chunk")
    async def on_chunk(data):
        stream_events.append(("report_chunk", data))
        if data.get("content"):
            report_chunks.append(data["content"])

    @sio.on("stream:final_report")
    async def on_final(data):
        stream_events.append(("final_report", data))
        final_report.value = data.get("report")

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

    assistant_message_id = f"msg-{uuid4().hex[:8]}"

    try:
        response = await sio.call(
            "chat:send",
            {
                "chatId": chat_id,
                "message": "Write a short overview of the topic.",
                "mode": "search",
                "messageId": assistant_message_id,
            },
            timeout=10,
        )
        assert response.get("success") is True

        await asyncio.wait_for(done_event.wait(), timeout=180)
        assert any(event[0] == "init" for event in stream_events)
        assert not error_event.is_set(), f"Streaming error: {error_payload.value}"

        answer = final_report.value or "".join(report_chunks)
        assert answer

        async with httpx.AsyncClient(base_url=API_BASE_URL, timeout=30.0) as client:
            await client.post(
                f"/api/chats/{chat_id}/messages",
                params={"role": "assistant", "content": answer, "message_id": assistant_message_id},
            )

            messages_response = await client.get(
                f"/api/chats/{chat_id}/messages",
                params={"limit": 10, "offset": 0, "order": "asc"},
            )
            assert messages_response.status_code == 200, messages_response.text
            messages_payload = messages_response.json()
            roles = [msg["role"] for msg in messages_payload["messages"]]
            assert "assistant" in roles

            search_response = await client.get(
                "/api/chats/search",
                params={"q": "test message", "limit": 5},
            )
            assert search_response.status_code == 200, search_response.text
            search_payload = search_response.json()
            assert search_payload["messages"]
    finally:
        await sio.disconnect()
        async with httpx.AsyncClient(base_url=API_BASE_URL, timeout=30.0) as client:
            await client.delete(f"/api/chats/{chat_id}")
