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

# Consumidores por nivel
async def L1(queue):
    while True:
        msg = await queue.get()
        # Aquí iría la ejecución de órdenes o simulación de bajo riesgo
        print(f"[L1] Acción final con: {msg}")

async def L2(queue, bus):
    while True:
        msg = await queue.get()
        # Lógica L2: filtra o ajusta señal
        new_msg = {'level': 'L2', 'strategy': f"{msg['strategy']}_L2"}
        print(f"[L2] Procesando: {msg} → {new_msg}")
        await bus.publish(new_msg)  # pasa al siguiente nivel

async def L3(queue, bus):
    while True:
        msg = await queue.get()
        # Lógica L3: transforma o añade info
        new_msg = {'level': 'L3', 'strategy': f"{msg['strategy']}_L3"}
        print(f"[L3] Procesando: {msg} → {new_msg}")
        await bus.publish(new_msg)  # pasa a L2

async def L4(bus):
    strategies = ['trend_following', 'mean_reversion', 'breakout']
    while True:
        # Publica una señal cada x segundos
        strategy = random.choice(strategies)
        msg = {'level': 'L4', 'strategy': strategy}
        print(f"[L4] Publicando: {msg}")
        await bus.publish(msg)
        await asyncio.sleep(random.uniform(1, 3))  # simula tiempo real

async def main():
    bus = MessageBus()

    # Crear queues por nivel
    queue_L3 = bus.create_queue()
    queue_L2 = bus.create_queue()
    queue_L1 = bus.create_queue()

    # Lanzar consumidores
    asyncio.create_task(L3(queue_L3, bus))
    asyncio.create_task(L2(queue_L2, bus))
    asyncio.create_task(L1(queue_L1))

    # L4 publica señales
    await L4(bus)

asyncio.run(main())
