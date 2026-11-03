# Servicio WebSocket - Eventos del sistema HRM
#
# Proporciona conectividad WebSocket para eventos en tiempo real del sistema HRM.
# Soporta autenticación por token, heartbeat, límite de clientes y descarte
# de eventos antiguos para clientes lentos sin bloquear el bucle de trading.
#
from typing import Dict, List, Optional, Any, Set, AsyncIterable
import json
import asyncio
import time
from datetime import datetime, timezone
#
# Tipos de evento definidos en el contrato
from enum import Enum
class EventoTipo(str, Enum):
    CIERRE = " ciclo\n