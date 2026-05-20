import json

import pytest


class _FakeWebSocket:
    def __init__(self) -> None:
        self.accepted = False
        self.sent: list[str] = []

    async def accept(self) -> None:
        self.accepted = True

    async def send_text(self, data: str) -> None:
        self.sent.append(data)


@pytest.mark.asyncio
async def test_connection_manager_routes_to_matching_client_id():
    from bangers.ws.manager import ConnectionManager

    manager = ConnectionManager()
    client_a = _FakeWebSocket()
    client_b = _FakeWebSocket()

    await manager.connect(client_a, client_id="client-a")
    await manager.connect(client_b, client_id="client-b")

    await manager.broadcast({"type": "progress", "job_id": "job-a"}, client_id="client-a")

    assert client_a.accepted is True
    assert client_b.accepted is True
    assert [json.loads(payload) for payload in client_a.sent] == [
        {"type": "progress", "job_id": "job-a"}
    ]
    assert client_b.sent == []


@pytest.mark.asyncio
async def test_connection_manager_broadcast_without_client_id_reaches_all():
    from bangers.ws.manager import ConnectionManager

    manager = ConnectionManager()
    client_a = _FakeWebSocket()
    client_b = _FakeWebSocket()

    await manager.connect(client_a, client_id="client-a")
    await manager.connect(client_b, client_id="client-b")

    await manager.broadcast({"status": "ok"})

    assert [json.loads(payload) for payload in client_a.sent] == [{"status": "ok"}]
    assert [json.loads(payload) for payload in client_b.sent] == [{"status": "ok"}]
