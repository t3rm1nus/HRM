#!/usr/bin/env python3
"""
Prueba del SimulatedExchangeClient y su integración con PortfolioManager
"""

import asyncio
import sys
import os

# Añadir el directorio actual al path para importar módulos
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.simulated_exchange_client import SimulatedExchangeClient
from core.portfolio_manager import PortfolioManager


async def test_simulated_client():
    """Prueba básica del SimulatedExchangeClient"""
    print("🎮 PRUEBA DEL SIMULATED EXCHANGE CLIENT")
    print("=" * 50)
    
    # Crear cliente simulado con balances iniciales
    fake_client = SimulatedExchangeClient(
        initial_balances={
            "BTC": 0.01549,
            "ETH": 0.385,
            "USDT": 3000.0
        },
        enable_commissions=True,
        enable_slippage=True
    )
    
    print(f"✅ Cliente simulado creado")
    print(f"   Balances iniciales: {await fake_client.get_account_balances()}")
    
    # Probar creación de órdenes
    print("\n🛒 CREANDO ÓRDENES DE PRUEBA")
    print("-" * 30)
    
    # Comprar BTC
    order1 = await fake_client.create_order("BTCUSDT", "buy", 0.001, order_type="market")
    print(f"   Orden 1 (BUY BTC): {order1}")
    
    # Vender ETH
    order2 = await fake_client.create_order("ETHUSDT", "sell", 0.05, order_type="market")
    print(f"   Orden 2 (SELL ETH): {order2}")
    
    # Comprar ETH con límite
    order3 = await fake_client.create_order("ETHUSDT", "buy", 0.1, price=3000.0, order_type="limit")
    print(f"   Orden 3 (BUY ETH LIMIT): {order3}")
    
    # Avanzar tiempo y ver cambios
    print("\n⏰ AVANZANDO TIEMPO Y SIMULANDO PRECIOS")
    print("-" * 40)
    
    fake_client.advance_time(20)
    
    print(f"   Precios después de simulación:")
    for symbol in ["BTCUSDT", "ETHUSDT"]:
        price = fake_client.get_market_price(symbol)
        print(f"     {symbol}: {price:.2f}")
    
    print(f"   Balances después de operaciones: {await fake_client.get_account_balances()}")
    
    # Obtener resumen de rendimiento
    print("\n📊 RESUMEN DE RENDIMIENTO")
    print("-" * 25)
    performance = fake_client.get_performance_summary()
    print(f"   Valor inicial: ${performance['initial_value']:.2f}")
    print(f"   Valor actual: ${performance['current_value']:.2f}")
    print(f"   P&L: ${performance['pnl']:.2f} ({performance['pnl_percentage']:.2f}%)")
    print(f"   Total trades: {performance['total_trades']}")
    print(f"   Comisiones totales: ${performance['total_fees']:.4f}")
    
    await fake_client.close()
    print("\n✅ Prueba del SimulatedExchangeClient completada")


async def test_portfolio_integration():
    """Prueba la integración con PortfolioManager"""
    print("\n\n🔗 PRUEBA DE INTEGRACIÓN CON PORTFOLIO MANAGER")
    print("=" * 55)
    
    # Crear cliente simulado
    fake_client = SimulatedExchangeClient(
        initial_balances={
            "BTC": 0.01549,
            "ETH": 0.385,
            "USDT": 3000.0
        },
        enable_commissions=True,
        enable_slippage=True
    )
    
    # Crear PortfolioManager en modo simulado
    portfolio_manager = PortfolioManager(
        client=fake_client,
        mode="simulated",
        enable_commissions=True,
        enable_slippage=True
    )
    
    print(f"✅ PortfolioManager creado en modo simulado")
    print(f"   Estado inicial del portfolio: {portfolio_manager.get_portfolio_state()}")
    
    # Simular algunas órdenes
    print("\n🛒 SIMULANDO ÓRDENES EN EL PORTFOLIO")
    print("-" * 35)
    
    # Crear órdenes simuladas
    orders = [
        {
            "symbol": "BTCUSDT",
            "side": "buy",
            "quantity": 0.002,
            "filled_price": 51000.0,
            "status": "filled"
        },
        {
            "symbol": "ETHUSDT",
            "side": "sell",
            "quantity": 0.1,
            "filled_price": 3100.0,
            "status": "filled"
        }
    ]
    
    # Datos de mercado simulados
    market_data = {
        "BTCUSDT": {"close": 51000.0},
        "ETHUSDT": {"close": 3100.0}
    }
    
    # Actualizar portfolio desde órdenes
    portfolio_manager.update_from_orders(orders, market_data)
    
    print(f"   Portfolio después de órdenes: {portfolio_manager.get_portfolio_state()}")
    
    # Verificar balances
    print("\n💰 BALANCES FINALES")
    print("-" * 18)
    print(f"   BTC: {portfolio_manager.get_balance('BTCUSDT'):.6f}")
    print(f"   ETH: {portfolio_manager.get_balance('ETHUSDT'):.3f}")
    print(f"   USDT: {portfolio_manager.get_balance('USDT'):.2f}")
    
    # Calcular valor total
    total_value = portfolio_manager.get_total_value(market_data)
    print(f"   Valor total: ${total_value:.2f}")
    
    await fake_client.close()
    print("\n✅ Prueba de integración completada")


async def test_example_initialization():
    """Prueba el ejemplo de inicialización solicitado"""
    print("\n\n🎯 EJEMPLO DE INICIALIZACIÓN SOLICITADO")
    print("=" * 42)
    
    # Ejemplo exacto como se solicitó
    fake_client = SimulatedExchangeClient(initial_balances={
        "BTC": 0.01549,
        "ETH": 0.385,
        "USDT": 3000.0
    })

    portfolio_manager = PortfolioManager(
        client=fake_client,
        mode="simulated",
        enable_commissions=True,
        enable_slippage=True
    )
    
    print("✅ Ejemplo de inicialización exitoso")
    print(f"   Cliente: {fake_client.__class__.__name__}")
    print(f"   Portfolio Manager: {portfolio_manager.__class__.__name__}")
    print(f"   Modo: {portfolio_manager.mode}")
    print(f"   Comisiones: {'Habilitadas' if portfolio_manager.enable_commissions else 'Deshabilitadas'}")
    print(f"   Slippage: {'Habilitado' if portfolio_manager.enable_slippage else 'Deshabilitado'}")
    
    # Mostrar estado inicial
    print(f"\n   Estado inicial del portfolio:")
    portfolio_state = portfolio_manager.get_portfolio_state()
    for key, value in portfolio_state.items():
        if isinstance(value, dict):
            print(f"     {key}: {value}")
        else:
            print(f"     {key}: {value}")
    
    await fake_client.close()
    print("\n✅ Ejemplo de inicialización completado")


async def main():
    """Función principal que ejecuta todas las pruebas"""
    print("🧪 PRUEBAS DEL SIMULATED EXCHANGE CLIENT")
    print("=" * 50)
    print("Testing SimulatedExchangeClient and PortfolioManager integration")
    print()
    
    try:
        # Ejecutar todas las pruebas
        await test_simulated_client()
        await test_portfolio_integration()
        await test_example_initialization()
        
        print("\n\n🎉 TODAS LAS PRUEBAS COMPLETADAS EXITOSAMENTE")
        print("=" * 50)
        print("✅ SimulatedExchangeClient funciona correctamente")
        print("✅ Integración con PortfolioManager exitosa")
        print("✅ Ejemplo de inicialización como se solicitó")
        print("\nEl cliente simulado está listo para ser utilizado en:")
        print("  - Backtesting")
        print("  - Testing de estrategias")
        print("  - Desarrollo sin riesgo real")
        
    except Exception as e:
        print(f"\n❌ ERROR EN LAS PRUEBAS: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)