from .strategy_selector import seleccionar_estrategia
from .portfolio_allocator import asignar_portfolio
from .drift_detector import detectar_deriva

def procesar_l4(state):
    deriva = detectar_deriva(state["mercado"])
    estrategia = seleccionar_estrategia(deriva, state["mercado"])
    portfolio_actualizado = asignar_portfolio({"capital": 1000}, estrategia)

    state["estrategia"] = estrategia
    state["portfolio"] = portfolio_actualizado
    state["deriva"] = deriva
    return state
