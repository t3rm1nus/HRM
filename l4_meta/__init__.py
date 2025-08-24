from .strategy_selector import seleccionar_estrategia
from .portfolio_allocator import asignar_portfolio
from .drift_detector import detectar_deriva

def procesar_l4(state):
    deriva = detectar_deriva(state["mercado"])
    estrategia = seleccionar_estrategia(deriva, state["mercado"])
    portfolio_actualizado = asignar_portfolio({"capital": 1000}, estrategia)

    # 🔧 Normalizar claves del portafolio a símbolos válidos
    symbol_map = {"BTC": "BTC/USDT", "ETH": "ETH/USDT", "USDT": "USDT"}
    portfolio_normalizado = {
        symbol_map.get(sym, sym): val for sym, val in portfolio_actualizado.items()
    }

    state["estrategia"] = estrategia
    state["portfolio"] = portfolio_normalizado
    state["deriva"] = deriva
    return state
