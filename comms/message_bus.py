import asyncio
from collections import defaultdict
from typing import Dict, List

class MessageBus:
    def __init__(self):
        self._topics: Dict[str, List[asyncio.Queue]] = defaultdict(list)

    def subscribe(self, topic: str) -> asyncio.Queue:
        q = asyncio.Queue()
        self._topics[topic].append(q)
        return q

    async def publish(self, topic: str, message: dict):
        for q in self._topics.get(topic, []):
            await q.put(message)

    # Compatibilidad opcional
    def create_queue(self, topic: str) -> asyncio.Queue:
        return self.subscribe(topic)

if __name__ == "__main__":
    # Ejemplo de uso manual (no se ejecuta en import)
    async def demo():
        bus = MessageBus()
        q = bus.subscribe("demo")

        async def consumer():
            while True:
                m = await q.get()
                print("[consumer]", m)

        asyncio.create_task(consumer())
        await bus.publish("demo", {"hello": "world"})

    asyncio.run(demo())
