import json
import asyncio
import math
from dataclasses import dataclass
from typing import Any
from fastapi import WebSocket
from loguru import logger


@dataclass
class _Connection:
    websocket: WebSocket
    client_id: str | None = None


class ConnectionManager:
    """Manages WebSocket connections for real-time generation progress."""

    def __init__(self) -> None:
        self._connections: list[_Connection] = []
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket, client_id: str | None = None) -> None:
        await websocket.accept()
        async with self._lock:
            self._connections.append(_Connection(websocket=websocket, client_id=client_id))
        logger.info(f"WebSocket connected. Total: {len(self._connections)}")

    async def disconnect(self, websocket: WebSocket) -> None:
        async with self._lock:
            self._connections = [
                connection
                for connection in self._connections
                if connection.websocket is not websocket
            ]
        logger.info(f"WebSocket disconnected. Total: {len(self._connections)}")

    async def broadcast(self, message: dict[str, Any], client_id: str | None = None) -> None:
        # Replace NaN/Inf floats with None (JSON spec doesn't support them)
        sanitized = {
            k: (None if isinstance(v, float) and not math.isfinite(v) else v)
            for k, v in message.items()
        }
        data = json.dumps(sanitized)
        disconnected: list[WebSocket] = []
        async with self._lock:
            connections = [
                connection
                for connection in self._connections
                if client_id is None or connection.client_id == client_id
            ]

        for connection in connections:
            try:
                await connection.websocket.send_text(data)
            except Exception:
                disconnected.append(connection.websocket)

        if disconnected:
            async with self._lock:
                self._connections = [
                    connection
                    for connection in self._connections
                    if connection.websocket not in disconnected
                ]

    async def send_to(self, websocket: WebSocket, message: dict[str, Any]) -> None:
        try:
            await websocket.send_text(json.dumps(message))
        except Exception:
            await self.disconnect(websocket)


generation_ws_manager = ConnectionManager()
health_ws_manager = ConnectionManager()
