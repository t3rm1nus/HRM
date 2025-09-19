# l3_strategy/bus_integration.py
from core.logging import logger
from comms.message_bus import MessageBus, Message
from datetime import datetime
from enum import Enum
from typing import Any, Dict

class L3MessageType(Enum):
    STRATEGIC_DECISION = "l3.strategic_decision"
    MARKET_REGIME_UPDATE = "l3.market_regime_update"
    PORTFOLIO_ALLOCATION = "l3.portfolio_allocation"

def publish_event(bus: MessageBus, event_type: L3MessageType, payload: Dict[str, Any]):
    try:
        msg = Message(
            topic=event_type.value,
            payload={
                "timestamp": datetime.utcnow().isoformat(),
                "data": payload
            }
        )
        return bus.publish(msg)
    except Exception as e:
        logger.error(f"❌ Error publicando evento {event_type}: {e}")

async def subscribe_event(bus: MessageBus, event_type: L3MessageType, handler):
    try:
        await bus.subscribe(event_type.value, handler)
        logger.info(f"✅ Suscrito a {event_type.value}")
    except Exception as e:
        logger.error(f"❌ Error suscribiendo a {event_type.value}: {e}")
