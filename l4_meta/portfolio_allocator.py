# l4_meta/portfolio_allocator.py

def asignar_portfolio(portfolio_state, estrategia):
    """
    Distribuye el capital según la estrategia seleccionada.
    """
    capital_total = float(portfolio_state["capital"])  # <-- asegurar tipo numérico

    if estrategia == "estrategia_defensiva":
        return {
            "BTC": 0.5 * capital_total,
            "ETH": 0.3 * capital_total,
            "USDT": 0.2 * capital_total
        }
    else:  # estrategia_agresiva
        return {
            "BTC": 0.7 * capital_total,
            "ETH": 0.2 * capital_total,
            "USDT": 0.1 * capital_total
        }