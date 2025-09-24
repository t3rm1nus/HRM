# comms/message_bus.py
# Núcleo del EventBus en memoria para toda la arquitectura HRM.

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Coroutine

from core.logging import logger


@dataclass
class Message:
    topic: str
    payload: Dict[str, Any]


class MessageBus:
    """
    MessageBus asíncrono en memoria.
    - Permite publish/subscribe a tópicos arbitrarios.
    - Los handlers son corutinas (async def).
    - Base para L1, L2, L3.
    """

    def __init__(self):
        self._subscribers: Dict[str, List[Callable[[Message], Awaitable[None]]]] = {}
        self._lock = asyncio.Lock()

    async def subscribe(self, topic: str, handler: Callable[[Message], Coroutine[Any, Any, None]]):
        """Suscribir un handler asíncrono a un tópico."""
        async with self._lock:
            if topic not in self._subscribers:
                self._subscribers[topic] = []
            self._subscribers[topic].append(handler)
            logger.debug(f"[MessageBus] Subscribed handler={handler.__name__} to topic={topic}")

    async def publish(self, message: Message):
        """Publicar un mensaje a todos los suscriptores del tópico."""
        async with self._lock:
            handlers = self._subscribers.get(message.topic, []).copy()

        if not handlers:
            logger.debug(f"[MessageBus] No subscribers for topic={message.topic}")
            return

        logger.debug(f"[MessageBus] Publishing to topic={message.topic} subscribers={len(handlers)}")
        for handler in handlers:
            try:
                asyncio.create_task(handler(message))
            except Exception as e:
                logger.exception(f"[MessageBus] Error publishing to {handler}: {e}")

    async def unsubscribe(self, topic: str, handler: Callable):
        """Desuscribir un handler de un tópico."""
        async with self._lock:
            if topic in self._subscribers:
                self._subscribers[topic] = [h for h in self._subscribers[topic] if h != handler]
                logger.debug(f"[MessageBus] Unsubscribed handler={handler.__name__} from topic={topic}")
