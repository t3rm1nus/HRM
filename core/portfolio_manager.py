# core/portfolio_manager.py - Gestión del portfolio
import os
import csv
import json
from datetime import datetime
import logging
import pandas as pd
from core.logging import logger
from l2_tactic.utils import safe_float
from typing import Dict, Any, Optional

# Importar TaxTracker para seguimiento fiscal
try:
    from hacienda.tax_tracker import TaxTracker
    TAX_TRACKER_AVAILABLE = True
except ImportError:
    TAX_TRACKER_AVAILABLE = False
    logger.warning("⚠️ TaxTracker no disponible - seguimiento fiscal deshabilitado")

# Importar Paper Trade Logger para trades simulados
try:
    from storage.paper_trade_logger import get_paper_logger
    PAPER_LOGGER_AVAILABLE = True
except ImportError:
    PAPER_LOGGER_AVAILABLE = False
    logger.warning("⚠️ PaperTradeLogger no disponible - trades simulados no se guardarán")

# Importar SimulatedExchangeClient para modo simulado
try:
    from core.simulated_exchange_client import SimulatedExchangeClient
    SIMULATED_CLIENT_AVAILABLE = True
except ImportError:
    SIMULATED_CLIENT_AVAILABLE = False
    logger.warning("⚠️ SimulatedExchangeClient no disponible - modo simulado limitado")

# REMOVED: Global function update_portfolio_from_orders has been deprecated
# The PortfolioManager.update_from_orders_async method should be used instead

async def save_portfolio_to_csv(state):
    """Guarda la línea del portfolio en un CSV externo usando los valores de state."""
    try:
        # FIX FINAL - CAUSA RAÍZ DEFINITIVA
        # Usar state["portfolio"] en lugar de campos raíz legacy
        portfolio = state.get("portfolio", {})
        
        total_value = portfolio.get("total_value", state.get("total_value", 0.0))
        btc_balance = portfolio.get("btc_balance", state.get("btc_balance", 0.0))
        btc_value = state.get("btc_value", 0.0)  # Mantener legacy field para compatibilidad
        eth_balance = portfolio.get("eth_balance", state.get("eth_balance", 0.0))
        eth_value = state.get("eth_value", 0.0)  # Mantener legacy field para compatibilidad
        usdt_balance = portfolio.get("usdt_balance", state.get("usdt_balance", 0.0))
        cycle_id = state.get("cycle_id", 0)

        output_dir = "data/portfolios"
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, "portfolio_log.csv")

        file_exists = os.path.isfile(csv_path)
        headers = [
            "timestamp", "cycle_id", "total_value",
            "btc_balance", "btc_value",
            "eth_balance", "eth_value",
            "usdt_balance"
        ]

        with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            if not file_exists:
                writer.writeheader()
            writer.writerow({
                "timestamp": datetime.now(datetime.timezone.utc).isoformat(),
                "cycle_id": cycle_id,
                "total_value": total_value,
                "btc_balance": btc_balance,
                "btc_value": btc_value,
                "eth_balance": eth_balance,
                "eth_value": eth_value,
                "usdt_balance": usdt_balance
            })
        logger.info(f"📝 Portfolio guardado en {csv_path}")

    except Exception as e:
        logger.error(f"❌ Error guardando portfolio en CSV: {e}", exc_info=True)


class PortfolioManager:
    """
    Clase principal para gestión del portfolio con modos múltiples:
    - live: Para runtime real (sincroniza con exchange real)
    - testnet: Para testing con datos reales pero sin riesgo
    - backtest: Para backtesting histórico (portfolio completamente limpio)
    - simulated: Para simulación con comisiones y slippage
    """

    def __init__(self, mode: str = "live", initial_balance: float = 3000.0,
                 client: Optional[Any] = None, symbols: list = None,
                 enable_commissions: bool = True, enable_slippage: bool = True):
        """
        Inicializa el PortfolioManager

        Args:
            mode: "live", "testnet", "backtest", "simulated"
            initial_balance: Balance inicial
            client: Cliente del exchange
            symbols: Lista de símbolos a manejar
            enable_commissions: Habilitar comisiones de trading
            enable_slippage: Habilitar slippage en órdenes
        """
        self.mode = mode
        self.initial_balance = initial_balance
        self.client = client
        self.symbols = symbols or ['BTCUSDT', 'ETHUSDT']
        self.enable_commissions = enable_commissions
        self.enable_slippage = enable_slippage

        # Configuración de comisiones y slippage por modo
        self._configure_trading_costs()

        # PROTECCIONES POR MODO
        if self.mode in ["backtest"]:
            # Deshabilitar completamente cualquier sincronización externa
            self.client = None  # Forzar None para evitar lecturas accidentales
            self.save_disabled = True  # Nunca guardar estado en modo backtest
            self.sync_disabled = True  # Nunca sincronizar con exchange
            self.persist_enabled = False  # No leer/escribir archivos de estado
            self.state_file = f"portfolio_state_{mode}.json"  # Archivo separado por modo
            logger.info(f"🛡️ MODO {mode.upper()}: Protecciones activadas - Sin sincronización externa")
            logger.info(f"📁 MODO {mode.upper()}: Usando archivo de estado separado - {self.state_file}")
            
        elif self.mode == "simulated":
            # ✅ CRITICAL: Permitir conexión con exchange real en modo paper
            # Mantener cliente para obtener datos reales de mercado
            self.save_disabled = False  # Permitir guardar estado en modo paper
            self.sync_disabled = False  # Permitir sincronización para datos reales
            self.persist_enabled = True  # Permitir persistencia en modo paper
            self.state_file = f"portfolio_state_{mode}.json"  # Archivo separado por modo
            logger.info(f"🛡️ MODO {mode.upper()}: Paper trading con datos reales - Conexión a exchange permitida")
            logger.info(f"📁 MODO {mode.upper()}: Usando archivo de estado separado - {self.state_file}")

        elif self.mode == "testnet":
            # Testnet: sincronización limitada, archivos separados
            self.state_file = "portfolio_state_testnet.json"
            self.persist_enabled = True
            logger.info("🧪 MODO TESTNET: Archivos separados, sincronización limitada")

        elif self.mode == "live":
            # Live: sincronización completa, archivos de producción
            self.state_file = "portfolio_state_live.json"
            self.persist_enabled = True
            logger.info("🔴 MODO LIVE: Sincronización completa con exchange real")

        # Estado del portfolio
        self.portfolio = {}
        self.peak_value = initial_balance
        self.total_fees = 0.0

        # SOLUTION 2: Position age tracking for rebalance grace period
        self.position_age = {}  # Track when positions were last modified (timestamp)
        self.MIN_HOLD_TIME = 60  # Minimum hold time in seconds before allowing rebalance (can be N cycles)

        # Inicializar portfolio según modo
        self._init_portfolio()

        logger.info(f"✅ PortfolioManager inicializado en modo '{mode}' con balance inicial: {initial_balance}")
        logger.info(f"   Comisiones: {'Habilitadas' if self.enable_commissions else 'Deshabilitadas'}")
        logger.info(f"   Slippage: {'Habilitado' if self.enable_slippage else 'Deshabilitado'}")

    def _init_portfolio(self):
        """Inicializa el portfolio según el modo configurado"""
        if self.mode == "simulated":
            # Portfolio COMPLETAMENTE limpio para simulación
            self.portfolio = {
                'BTCUSDT': {'position': 0.0, 'free': 0.0},
                'ETHUSDT': {'position': 0.0, 'free': 0.0},
                'USDT': {'free': self.initial_balance},
                'total': self.initial_balance,
                'peak_value': self.initial_balance,
                'total_fees': 0.0
            }
            logger.info(f"🎯 Portfolio simulado LIMPIO inicializado: {self.initial_balance} USDT")

        elif self.mode == "backtest":
            # Portfolio limpio para backtesting
            self.portfolio = {
                'BTCUSDT': {'position': 0.0, 'free': 0.0},
                'ETHUSDT': {'position': 0.0, 'free': 0.0},
                'USDT': {'free': self.initial_balance},
                'total': self.initial_balance,
                'peak_value': self.initial_balance,
                'total_fees': 0.0
            }
            logger.info(f"📊 Portfolio backtest LIMPIO inicializado: {self.initial_balance} USDT")

        elif self.mode == "testnet":
            # Testnet: intentar sincronizar pero con precaución
            try:
                import asyncio
                self.portfolio = asyncio.run(self._sync_with_exchange_async())
                logger.info("🧪 Portfolio testnet sincronizado")
            except Exception as e:
                logger.warning(f"⚠️ Error sincronizando testnet: {e}, usando balance inicial")
                self.portfolio = {
                    'BTCUSDT': {'position': 0.0, 'free': 0.0},
                    'ETHUSDT': {'position': 0.0, 'free': 0.0},
                    'USDT': {'free': self.initial_balance},
                    'total': self.initial_balance,
                    'peak_value': self.initial_balance,
                    'total_fees': 0.0
                }

        elif self.mode == "live":
            # Live: sincronizar con exchange real
            try:
                import asyncio
                self.portfolio = asyncio.run(self._sync_with_exchange_async())
                logger.info("🔴 Portfolio live sincronizado con exchange real")
            except Exception as e:
                logger.error(f"❌ Error sincronizando live: {e}")
                logger.warning("⚠️ Usando balance inicial como fallback - REVISA CONEXIÓN!")
                self.portfolio = {
                    'BTCUSDT': {'position': 0.0, 'free': 0.0},
                    'ETHUSDT': {'position': 0.0, 'free': 0.0},
                    'USDT': {'free': self.initial_balance},
                    'total': self.initial_balance,
                    'peak_value': self.initial_balance,
                    'total_fees': 0.0
                }

    async def _sync_with_exchange_async(self):
        """Sincroniza el portfolio con el exchange real (versión async)"""
        try:
            if not hasattr(self.client, 'get_account_balances'):
                logger.warning("⚠️ Cliente no tiene método get_account_balances, usando balance inicial")
                # Fall back to initial balance instead of empty portfolio
                return {
                    'BTCUSDT': {'position': 0.0, 'free': 0.0},
                    'ETHUSDT': {'position': 0.0, 'free': 0.0},
                    'USDT': {'free': self.initial_balance},
                    'total': self.initial_balance,
                    'peak_value': self.initial_balance,
                    'total_fees': 0.0
                }

            # Obtener balances reales del exchange
            balances = await self.client.get_account_balances()

            # Convertir a estructura interna
            portfolio = {
                'USDT': {'free': safe_float(balances.get('USDT', 0.0))},
                'total': 0.0,
                'peak_value': 0.0,
                'total_fees': 0.0
            }

            total_value = 0.0

            # Procesar cada símbolo
            for symbol in self.symbols:
                if symbol in balances:
                    position = safe_float(balances[symbol])
                    portfolio[symbol] = {'position': position, 'free': position}
                    total_value += position
                else:
                    portfolio[symbol] = {'position': 0.0, 'free': 0.0}

            # Agregar USDT al total
            total_value += portfolio['USDT']['free']

            portfolio['total'] = total_value
            portfolio['peak_value'] = total_value

            logger.info(f"🔄 Portfolio sincronizado - Total: {total_value:.2f} USDT")
            return portfolio

        except Exception as e:
            logger.error(f"❌ Error sincronizando con exchange: {e}")
            logger.info("⚠️ Usando balance inicial como fallback")
            # Fall back to initial balance instead of empty portfolio
            return {
                'BTCUSDT': {'position': 0.0, 'free': 0.0},
                'ETHUSDT': {'position': 0.0, 'free': 0.0},
                'USDT': {'free': self.initial_balance},
                'total': self.initial_balance,
                'peak_value': self.initial_balance,
                'total_fees': 0.0
            }

    def _configure_trading_costs(self):
        """Configura comisiones y slippage según el modo"""
        if self.mode == "live":
            # Comisiones reales de Binance
            self.maker_fee = 0.001  # 0.1%
            self.taker_fee = 0.001  # 0.1%
            self.slippage_bps = 2  # 2 basis points (0.02%)
        elif self.mode == "testnet":
            # Comisiones de testnet (más altas para testing)
            self.maker_fee = 0.0015  # 0.15%
            self.taker_fee = 0.0015  # 0.15%
            self.slippage_bps = 5  # 5 basis points (0.05%)
        elif self.mode == "backtest":
            # Backtest: comisiones históricas promedio
            self.maker_fee = 0.0012  # 0.12%
            self.taker_fee = 0.0012  # 0.12%
            self.slippage_bps = 3  # 3 basis points (0.03%)
        elif self.mode == "simulated":
            # Simulación: comisiones más altas para testing conservador
            self.maker_fee = 0.002  # 0.2%
            self.taker_fee = 0.002  # 0.2%
            self.slippage_bps = 10  # 10 basis points (0.1%)
        else:
            # Default
            self.maker_fee = 0.001
            self.taker_fee = 0.001
            self.slippage_bps = 5

        # Deshabilitar si se solicita
        if not self.enable_commissions:
            self.maker_fee = 0.0
            self.taker_fee = 0.0
        if not self.enable_slippage:
            self.slippage_bps = 0

    def reset(self):
        """Resetea el portfolio según el modo actual"""
        logger.info(f"🔄 Reseteando portfolio en modo '{self.mode}'")
        self._init_portfolio()

    def force_clean_reset(self):
        """Fuerza un reset completo del portfolio, ignorando cualquier estado previo"""
        logger.info("🧹 FORZANDO RESET COMPLETO DEL PORTFOLIO")

        # Limpiar completamente todas las variables de instancia
        self.portfolio = {}
        self.peak_value = 0.0
        self.total_fees = 0.0

        # Forzar portfolio limpio con balance inicial, ignorando sincronización con exchange
        self.portfolio = {
            'BTCUSDT': {'position': 0.0, 'free': 0.0},
            'ETHUSDT': {'position': 0.0, 'free': 0.0},
            'USDT': {'free': self.initial_balance},
            'total': self.initial_balance,
            'peak_value': self.initial_balance,
            'total_fees': 0.0
        }
        self.peak_value = self.initial_balance
        self.total_fees = 0.0

        logger.info(f"✅ Portfolio forzado a estado limpio: USDT={self.initial_balance}")

        # Verificar que esté realmente limpio
        self._verify_clean_state()

    def _verify_clean_state(self):
        """Verifica que el portfolio esté en estado limpio"""
        btc_balance = self.get_balance("BTCUSDT")
        eth_balance = self.get_balance("ETHUSDT")
        usdt_balance = self.get_balance("USDT")

        expected_usdt = self.initial_balance
        if abs(usdt_balance - expected_usdt) > 0.01 or btc_balance != 0.0 or eth_balance != 0.0:
            logger.error(f"❌ VERIFICACIÓN FALLIDA - Portfolio no está limpio!")
            logger.error(f"   Esperado: BTC=0.0, ETH=0.0, USDT={expected_usdt}")
            logger.error(f"   Actual: BTC={btc_balance}, ETH={eth_balance}, USDT={usdt_balance}")
            # Forzar corrección
            self.portfolio = {
                'BTCUSDT': {'position': 0.0, 'free': 0.0},
                'ETHUSDT': {'position': 0.0, 'free': 0.0},
                'USDT': {'free': self.initial_balance},
                'total': self.initial_balance,
                'peak_value': self.initial_balance,
                'total_fees': 0.0
            }
            self.peak_value = self.initial_balance
            self.total_fees = 0.0
            logger.info("🔄 Portfolio corregido forzosamente")
        else:
            logger.info("✅ VERIFICACIÓN EXITOSA - Portfolio está completamente limpio")

    def get_portfolio_state(self) -> Dict[str, Any]:
        """Retorna el estado actual del portfolio"""
        return self.portfolio.copy()

    async def update_from_orders_async(self, orders: list, market_data: Dict[str, Any]):
        """
        Actualiza el portfolio basado en órdenes ejecutadas (implementación directa)
        """
        try:
            # Filtrar órdenes ejecutadas
            filled_orders = [o for o in orders if o.get("status") == "filled"]
            if len(filled_orders) == 0:
                logger.debug("ℹ️ No hay órdenes procesadas - manteniendo portfolio actual")
                return

            logger.info(f"📈 Procesando {len(filled_orders)} órdenes ejecutadas")

            # Obtener balances actuales
            btc_balance = self.get_balance("BTCUSDT")
            eth_balance = self.get_balance("ETHUSDT")
            usdt_balance = self.get_balance("USDT")

            # Obtener precios actuales del mercado
            btc_price = None
            eth_price = None

            btc_market = market_data.get("BTCUSDT", {})
            if isinstance(btc_market, dict):
                btc_price = safe_float(btc_market.get("close", 50000.0))
            elif isinstance(btc_market, (pd.Series, pd.DataFrame)) and 'close' in btc_market:
                btc_price = safe_float(btc_market['close'].iloc[-1] if isinstance(btc_market, pd.DataFrame) else btc_market['close'])
            else:
                btc_price = 50000.0

            eth_market = market_data.get("ETHUSDT", {})
            if isinstance(eth_market, dict):
                eth_price = safe_float(eth_market.get("close", 4327.46))
            elif isinstance(eth_market, (pd.Series, pd.DataFrame)) and 'close' in eth_market:
                eth_price = safe_float(eth_market['close'].iloc[-1] if isinstance(eth_market, pd.DataFrame) else eth_market['close'])
            else:
                eth_price = 4327.46

            # Procesar órdenes ejecutadas
            for order in filled_orders:
                symbol = order.get("symbol")
                side = order.get("side")
                quantity = safe_float(order.get("quantity", 0.0))
                price = safe_float(order.get("filled_price", btc_price if symbol == "BTCUSDT" else eth_price))

                if not price or price <= 0:
                    logger.warning(f"⚠️ Precio inválido para {symbol}: {price}, omitiendo orden")
                    continue

                # CRÍTICO: Usar valor absoluto de quantity para cálculos (el signo indica dirección)
                abs_quantity = abs(quantity)

                # Calcular costos de trading usando cantidad absoluta
                order_value = abs_quantity * price
                trading_fee_rate = 0.001  # 0.1% comisión
                trading_fee = order_value * trading_fee_rate

                # SOLUTION 2: Update position age tracking for rebalance grace period
                current_time = datetime.now().timestamp()

                # Procesar órdenes según dirección (signo de quantity)
                if symbol == "BTCUSDT":
                    if side.lower() == "buy":
                        total_cost = order_value + trading_fee
                        if usdt_balance < total_cost:
                            logger.warning(f"⚠️ FONDOS INSUFICIENTES para comprar BTC: {usdt_balance:.6f} < {total_cost:.6f}")
                            continue
                        btc_balance += abs_quantity  # Siempre positivo para posiciones
                        usdt_balance -= total_cost
                        # SOLUTION 2: Update position age when buying
                        self.position_age[symbol] = current_time
                        logger.info(f"✅ BUY BTC: {abs_quantity:.6f} @ {price:.2f} (costo total: {total_cost:.4f})")
                    elif side.lower() == "sell":
                        if btc_balance < abs_quantity:
                            logger.warning(f"⚠️ BTC INSUFICIENTE para vender: {btc_balance:.6f} < {abs_quantity:.6f}")
                            continue
                        proceeds = order_value - trading_fee
                        btc_balance -= abs_quantity  # Reducir posición
                        usdt_balance += proceeds  # Agregar proceeds
                        logger.info(f"✅ SELL BTC: {abs_quantity:.6f} @ {price:.2f} (proceeds: {proceeds:.4f})")

                elif symbol == "ETHUSDT":
                    if side.lower() == "buy":
                        total_cost = order_value + trading_fee
                        if usdt_balance < total_cost:
                            logger.warning(f"⚠️ FONDOS INSUFICIENTES para comprar ETH: {usdt_balance:.6f} < {total_cost:.6f}")
                            continue
                        eth_balance += abs_quantity  # Siempre positivo para posiciones
                        usdt_balance -= total_cost
                        # SOLUTION 2: Update position age when buying
                        self.position_age[symbol] = current_time
                        logger.info(f"✅ BUY ETH: {abs_quantity:.6f} @ {price:.2f} (costo total: {total_cost:.4f})")
                    elif side.lower() == "sell":
                        if eth_balance < abs_quantity:
                            logger.warning(f"⚠️ ETH INSUFICIENTE para vender: {eth_balance:.6f} < {abs_quantity:.6f}")
                            continue
                        proceeds = order_value - trading_fee
                        eth_balance -= abs_quantity  # Reducir posición
                        usdt_balance += proceeds  # Agregar proceeds
                        logger.info(f"✅ SELL ETH: {abs_quantity:.6f} @ {price:.2f} (proceeds: {proceeds:.4f})")

                # Acumular fees (siempre positivo)
                self.total_fees += trading_fee

            # Validar balances finales
            if usdt_balance < -0.001 or btc_balance < -0.000001 or eth_balance < -0.000001:
                logger.error(f"🚨 BALANCES NEGATIVOS DETECTADOS: BTC={btc_balance}, ETH={eth_balance}, USDT={usdt_balance}")
                # Corregir balances negativos
                btc_balance = max(0, btc_balance)
                eth_balance = max(0, eth_balance)
                usdt_balance = max(0, usdt_balance)
                logger.info("🔄 Balances negativos corregidos")

            # Actualizar portfolio
            self.portfolio = {
                'BTCUSDT': {'position': btc_balance, 'free': btc_balance},
                'ETHUSDT': {'position': eth_balance, 'free': eth_balance},
                'USDT': {'free': usdt_balance},
                'total': self.get_total_value(market_data),
                'peak_value': max(self.peak_value, self.get_total_value(market_data)),
                'total_fees': self.total_fees
            }

            # LOGGEAR TRADES PAPER SI ESTAMOS EN MODO SIMULADO
            if self.mode in ["simulated", "backtest"] and PAPER_LOGGER_AVAILABLE:
                try:
                    paper_logger = get_paper_logger()
                    # Obtener cycle_id del contexto (si está disponible en market_data o pasar None)
                    cycle_id = market_data.get('cycle_id') if isinstance(market_data, dict) else None

                    for order in filled_orders:
                        paper_logger.log_paper_trade(
                            order=order,
                            market_data=market_data,
                            cycle_id=cycle_id,
                            strategy="portfolio_manager_simulated"
                        )
                except Exception as log_error:
                    logger.warning(f"⚠️ Error logging paper trade: {log_error}")

            logger.info(f"✅ Portfolio actualizado: BTC={btc_balance:.6f}, ETH={eth_balance:.3f}, USDT={usdt_balance:.2f}")

        except Exception as e:
            logger.error(f"❌ Error actualizando portfolio desde órdenes: {e}")

    def update_from_orders(self, orders: list, market_data: Dict[str, Any]):
        """
        Actualiza el portfolio basado en órdenes ejecutadas (wrapper sync para compatibilidad)
        """
        import asyncio
        try:
            # Try to get the current event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're in an async context, call the async version directly
                import asyncio
                # Create a new task for the async operation
                task = loop.create_task(self.update_from_orders_async(orders, market_data))
                # Wait for it to complete (this will work in async contexts)
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(lambda: asyncio.run_coroutine_threadsafe(task, loop).result())
                    future.result(timeout=10)  # 10 second timeout
            else:
                # No running loop, use asyncio.run
                asyncio.run(self.update_from_orders_async(orders, market_data))
        except RuntimeError:
            # Fallback: try to run in current thread
            try:
                asyncio.run(self.update_from_orders_async(orders, market_data))
            except RuntimeError as e2:
                logger.warning(f"Could not update portfolio async, falling back to sync: {e2}")
                # Last resort: skip the update to avoid crashes
                pass

    def get_balance(self, symbol: str) -> float:
        """Obtiene el balance de un símbolo específico"""
        if symbol == "USDT":
            return safe_float(self.portfolio.get('USDT', {}).get('free', 0.0))
        else:
            return safe_float(self.portfolio.get(symbol, {}).get('position', 0.0))

    def get_usdt_balance(self) -> float:
        """Obtiene el balance de USDT (método específico para USDT)"""
        return float(self.portfolio.get('USDT', {}).get('free', 0.0))

    def get_total_value(self, market_data: Optional[Dict[str, Any]] = None) -> float:
        """Calculate total portfolio value"""
        if market_data is None:
            return safe_float(self.portfolio.get('total', 0.0))

        total_value = self.get_balance("USDT")

        for symbol in self.symbols:
            balance = self.get_balance(symbol)
            if balance > 0 and symbol in market_data:
                symbol_data = market_data[symbol]
                if isinstance(symbol_data, dict) and 'close' in symbol_data:
                    price = safe_float(symbol_data['close'])
                    total_value += balance * price
                elif isinstance(symbol_data, pd.DataFrame) and 'close' in symbol_data.columns:
                    # Handle pandas DataFrame - use latest close
                    try:
                        price = safe_float(symbol_data['close'].iloc[-1])
                        total_value += balance * price
                    except (KeyError, IndexError):
                        logger.debug(f"No close price available for {symbol} DataFrame")
                elif isinstance(symbol_data, pd.Series) and len(symbol_data) > 0:
                    # Handle pandas Series - use latest value
                    price = safe_float(symbol_data.iloc[-1])
                    total_value += balance * price
                elif isinstance(symbol_data, (pd.Series, pd.DataFrame)) and len(symbol_data) > 0:
                    # Fallback for other formats
                    try:
                        if isinstance(symbol_data, pd.DataFrame):
                            price = safe_float(symbol_data.iloc[-1].get('close', symbol_data.iloc[-1].get('price', 0)))
                        else:  # Series
                            price = safe_float(symbol_data.iloc[-1])
                        if price > 0:
                            total_value += balance * price
                    except:
                        logger.debug(f"Could not extract price for {symbol}")

        return total_value

    def save_to_csv(self, cycle_id: int = 0):
        """Guarda el estado del portfolio en CSV"""
        # PROTECCIÓN PARA MODO SIMULADO: Nunca guardar estado
        if self.mode == "simulated":
            if hasattr(self, 'save_disabled') and self.save_disabled:
                logger.debug("🛡️ MODO SIMULADO: Guardado de estado deshabilitado")
                return
            else:
                logger.warning("⚠️ MODO SIMULADO: Intentando guardar estado - esto no debería suceder")

        try:
            # Crear estado mock para usar la función existente
            state = {
                "total_value": self.get_total_value(),
                "btc_balance": self.get_balance("BTCUSDT"),
                "btc_value": 0.0,  # TODO: calcular con precios
                "eth_balance": self.get_balance("ETHUSDT"),
                "eth_value": 0.0,  # TODO: calcular con precios
                "usdt_balance": self.get_balance("USDT"),
                "cycle_id": cycle_id
            }

            # Usar la función existente
            import asyncio
            asyncio.run(save_portfolio_to_csv(state))

        except Exception as e:
            logger.error(f"❌ Error guardando portfolio en CSV: {e}")

    def save_to_json(self):
        """Guarda el estado completo del portfolio en JSON para persistencia entre sesiones"""
        try:
            # PROTECCIÓN PARA MODO SIMULADO: Nunca guardar estado persistente
            if self.mode == "simulated":
                if hasattr(self, 'persist_enabled') and not self.persist_enabled:
                    logger.debug("🛡️ MODO SIMULADO: Persistencia JSON deshabilitada")
                    return
                else:
                    logger.warning("⚠️ MODO SIMULADO: Intentando guardar estado persistente - esto no debería suceder")

            # Crear estado completo para guardar
            portfolio_state = {
                "portfolio": self.portfolio.copy(),
                "peak_value": self.peak_value,
                "total_fees": self.total_fees,
                "mode": self.mode,
                "initial_balance": self.initial_balance,
                "symbols": self.symbols,
                "timestamp": datetime.utcnow().isoformat(),
                "version": "1.0"
            }

            # Determinar archivo según modo
            if self.mode == "live":
                state_file = "portfolio_state_live.json"
            else:
                state_file = "portfolio_state.json"

            # Guardar en JSON
            with open(state_file, 'w', encoding='utf-8') as f:
                json.dump(portfolio_state, f, indent=2, default=str)

            logger.info(f"💾 Portfolio guardado en {state_file}")
            logger.debug(f"   Estado guardado: BTC={self.get_balance('BTCUSDT'):.6f}, ETH={self.get_balance('ETHUSDT'):.3f}, USDT={self.get_balance('USDT'):.2f}")

        except Exception as e:
            logger.error(f"❌ Error guardando portfolio en JSON: {e}")

    async def sync_with_exchange(self):
        """
        CRÍTICO: Sincroniza el portfolio con el estado REAL de Binance.
        Esto previene pérdidas catastróficas por desincronización.
        """
        try:
            if self.mode == "simulated":
                logger.debug("🛡️ MODO SIMULADO: Saltando sincronización con exchange")
                return True  # ✅ CAMBIO OBLIGATORIO: Nunca debe fallar en simulated

            if not self.client:
                logger.warning("⚠️ No hay cliente de exchange disponible para sincronización")
                return False

            logger.info("🔄 Sincronizando portfolio con estado real de Binance...")

            # Obtener balances reales de Binance
            exchange_balances = await self.client.get_account_balances()

            if not exchange_balances:
                logger.error("❌ No se pudieron obtener balances de Binance")
                return False

            # Convertir a estructura interna del portfolio
            synced_portfolio = {
                'USDT': {'free': exchange_balances.get('USDT', 0.0), 'position': exchange_balances.get('USDT', 0.0)},
                'total': 0.0,
                'peak_value': self.peak_value,  # Mantener peak value local
                'total_fees': self.total_fees   # Mantener fees locales
            }

            total_value = exchange_balances.get('USDT', 0.0)

            # Procesar otros activos
            for symbol in self.symbols:
                if symbol in exchange_balances:
                    balance = exchange_balances[symbol]
                    synced_portfolio[symbol] = {
                        'position': balance,
                        'free': balance
                    }
                    # Calcular valor aproximado (usando precios hardcoded por simplicidad)
                    if symbol == "BTCUSDT":
                        total_value += balance * 50000  # Precio aproximado
                    elif symbol == "ETHUSDT":
                        total_value += balance * 3000   # Precio aproximado

            synced_portfolio['total'] = total_value

            # CRÍTICO: Comparar con estado local para detectar discrepancias
            local_btc = self.get_balance("BTCUSDT")
            local_eth = self.get_balance("ETHUSDT")
            local_usdt = self.get_balance("USDT")

            exchange_btc = exchange_balances.get("BTCUSDT", 0.0)
            exchange_eth = exchange_balances.get("ETHUSDT", 0.0)
            exchange_usdt = exchange_balances.get("USDT", 0.0)

            # Detectar discrepancias significativas
            btc_diff = abs(local_btc - exchange_btc)
            eth_diff = abs(local_eth - exchange_eth)
            usdt_diff = abs(local_usdt - exchange_usdt)

            if btc_diff > 0.0001 or eth_diff > 0.001 or usdt_diff > 1.0:
                logger.warning("🚨 DESINCRONIZACIÓN DETECTADA entre estado local y exchange:")
                logger.warning(f"   BTC: Local={local_btc:.6f}, Exchange={exchange_btc:.6f} (diff={btc_diff:.6f})")
                logger.warning(f"   ETH: Local={local_eth:.3f}, Exchange={exchange_eth:.3f} (diff={eth_diff:.3f})")
                logger.warning(f"   USDT: Local={local_usdt:.2f}, Exchange={exchange_usdt:.2f} (diff={usdt_diff:.2f})")
                logger.warning("   ✅ SINCRONIZANDO con estado real de exchange")

            # Aplicar estado sincronizado
            self.portfolio = synced_portfolio
            self.peak_value = max(self.peak_value, total_value)  # Actualizar peak value

            logger.info("✅ Portfolio sincronizado con Binance:")
            logger.info(f"   BTC: {exchange_btc:.6f}, ETH: {exchange_eth:.3f}, USDT: {exchange_usdt:.2f}")
            logger.info(f"   Valor total: ${total_value:.2f}")

            return True

        except Exception as e:
            logger.error(f"❌ Error sincronizando con exchange: {e}")
            logger.warning("⚠️ Continuando con estado local - RIESGO DE DESINCRONIZACIÓN")
            return False

    def load_from_json(self):
        """Carga el estado del portfolio desde JSON para restaurar entre sesiones"""
        try:
            # Determinar archivo según modo
            if self.mode == "live":
                state_file = "portfolio_state_live.json"
            else:
                state_file = "portfolio_state.json"

            # Verificar si el archivo existe
            if not os.path.exists(state_file):
                logger.info(f"📄 Archivo de estado {state_file} no existe, usando inicialización por defecto")
                return False

            # Cargar desde JSON
            with open(state_file, 'r', encoding='utf-8') as f:
                portfolio_state = json.load(f)

            # Validar versión y estructura
            if portfolio_state.get("version") != "1.0":
                logger.warning(f"⚠️ Versión de estado incompatible: {portfolio_state.get('version')}, ignorando")
                return False

            # Restaurar estado
            self.portfolio = portfolio_state.get("portfolio", {})
            self.peak_value = portfolio_state.get("peak_value", self.initial_balance)
            self.total_fees = portfolio_state.get("total_fees", 0.0)

            # Validar que el estado cargado sea consistente
            loaded_mode = portfolio_state.get("mode")
            if loaded_mode != self.mode:
                logger.warning(f"⚠️ Modo de estado cargado ({loaded_mode}) diferente al actual ({self.mode}), ajustando")

            # Verificar balances
            btc_balance = self.get_balance("BTCUSDT")
            eth_balance = self.get_balance("ETHUSDT")
            usdt_balance = self.get_balance("USDT")

            logger.info(f"📂 Portfolio cargado desde {state_file}")
            logger.info(f"   Estado restaurado: BTC={btc_balance:.6f}, ETH={eth_balance:.3f}, USDT={usdt_balance:.2f}")
            logger.info(f"   Peak Value: {self.peak_value:.2f}, Total Fees: {self.total_fees:.4f}")

            return True

        except Exception as e:
            logger.error(f"❌ Error cargando portfolio desde JSON: {e}")
            logger.info("⚠️ Usando inicialización por defecto")
            return False

    def check_for_contamination_sources(self):
        """Verifica posibles fuentes de contaminación del portfolio"""
        contamination_sources = []

        # Verificar archivos de estado
        possible_files = [
            "portfolio_state.json",
            "portfolio_state_live.json",
            "portfolio_state_simulated.json",
            "data/portfolios/portfolio_log.csv"
        ]

        for file_path in possible_files:
            if os.path.exists(file_path):
                try:
                    if file_path.endswith('.json'):
                        with open(file_path, 'r') as f:
                            data = json.load(f)
                        contamination_sources.append(f"Archivo JSON: {file_path}")
                    elif file_path.endswith('.csv'):
                        with open(file_path, 'r') as f:
                            lines = f.readlines()
                        if len(lines) > 1:  # Más de solo headers
                            contamination_sources.append(f"Archivo CSV: {file_path} ({len(lines)-1} registros)")
                except Exception as e:
                    logger.debug(f"Error leyendo {file_path}: {e}")

        # Verificar si hay cliente de exchange disponible
        if self.client is not None and self.mode == "simulated":
            contamination_sources.append("Cliente de exchange disponible en modo simulado")

        if contamination_sources:
            logger.warning("🚨 POSIBLES FUENTES DE CONTAMINACIÓN DETECTADAS:")
            for source in contamination_sources:
                logger.warning(f"   - {source}")
        else:
            logger.info("✅ No se detectaron fuentes de contaminación")

        return contamination_sources

    def clean_contamination_sources(self):
        """Limpia archivos de estado que puedan estar contaminando el portfolio"""
        if self.mode != "simulated":
            logger.warning("⚠️ clean_contamination_sources() solo debe usarse en modo simulado")
            return

        logger.info("🧹 LIMPIANDO FUENTES DE CONTAMINACIÓN EN MODO SIMULADO...")

        # Archivos a limpiar en modo simulado
        files_to_clean = [
            "portfolio_state.json",  # Archivo genérico que puede contaminar
            "portfolio_state_simulated.json",  # Archivo específico de simulación
            "data/portfolios/portfolio_log.csv"  # Historial de portfolio que puede tener datos antiguos
        ]

        cleaned_files = []
        for file_path in files_to_clean:
            if os.path.exists(file_path):
                try:
                    # Hacer backup antes de eliminar
                    backup_path = f"{file_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    os.rename(file_path, backup_path)
                    cleaned_files.append(f"{file_path} -> {backup_path}")
                    logger.info(f"📁 Backup creado: {file_path} -> {backup_path}")
                except Exception as e:
                    logger.warning(f"⚠️ Error haciendo backup de {file_path}: {e}")

        if cleaned_files:
            logger.info("✅ ARCHIVOS LIMPIADOS PARA EVITAR CONTAMINACIÓN:")
            for cleaned in cleaned_files:
                logger.info(f"   - {cleaned}")
        else:
            logger.info("✅ No se encontraron archivos para limpiar")

        return cleaned_files

    def add_restore_logging(self, operation_name: str):
        """Agrega logging detallado para operaciones de restauración"""
        if self.mode == "simulated":
            logger.info(f"🔍 MODO SIMULADO - {operation_name}")
            logger.info(f"   Estado actual - BTC: {self.get_balance('BTCUSDT')}, ETH: {self.get_balance('ETHUSDT')}, USDT: {self.get_balance('USDT')}")
            logger.info(f"   Cliente disponible: {self.client is not None}")
            logger.info(f"   Persistencia habilitada: {getattr(self, 'persist_enabled', True)}")

    def diagnose_portfolio_explosion(self, orders: list, market_data: Dict[str, Any]):
        """Diagnóstico avanzado para detectar explosiones de valor en portfolio"""
        if self.mode != "simulated":
            return

        logger.info("🔬 DIAGNÓSTICO AVANZADO DE PORTFOLIO:")

        # Verificar estado actual antes de cualquier cambio
        current_btc = self.get_balance("BTCUSDT")
        current_eth = self.get_balance("ETHUSDT")
        current_usdt = self.get_balance("USDT")
        current_total = self.get_total_value(market_data)

        logger.info(f"   Estado actual: BTC={current_btc:.6f}, ETH={current_eth:.3f}, USDT={current_usdt:.2f}, Total={current_total:.2f}")

        # Simular cambios que harían las órdenes
        simulated_btc = current_btc
        simulated_eth = current_eth
        simulated_usdt = current_usdt

        for order in orders:
            if order.get("status") != "filled":
                continue

            symbol = order.get("symbol")
            side = order.get("side")
            quantity = safe_float(order.get("quantity", 0))
            price = safe_float(order.get("filled_price", 0))

            if symbol == "BTCUSDT":
                if side == "buy":
                    cost = quantity * price * 1.001  # incluir fee
                    if simulated_usdt >= cost:
                        simulated_btc += quantity
                        simulated_usdt -= cost
                    else:
                        logger.warning(f"   Orden BUY BTC omitida - USDT insuficiente: {simulated_usdt:.2f} < {cost:.2f}")
                elif side == "sell":
                    if simulated_btc >= quantity:
                        proceeds = quantity * price * 0.999  # incluir fee
                        simulated_btc -= quantity
                        simulated_usdt += proceeds
                    else:
                        logger.warning(f"   Orden SELL BTC omitida - BTC insuficiente: {simulated_btc:.6f} < {quantity:.6f}")

            elif symbol == "ETHUSDT":
                if side == "buy":
                    cost = quantity * price * 1.001  # incluir fee
                    if simulated_usdt >= cost:
                        simulated_eth += quantity
                        simulated_usdt -= cost
                    else:
                        logger.warning(f"   Orden BUY ETH omitida - USDT insuficiente: {simulated_usdt:.2f} < {cost:.2f}")
                elif side == "sell":
                    if simulated_eth >= quantity:
                        proceeds = quantity * price * 0.999  # incluir fee
                        simulated_eth -= quantity
                        simulated_usdt += proceeds
                    else:
                        logger.warning(f"   Orden SELL ETH omitida - ETH insuficiente: {simulated_eth:.3f} < {quantity:.3f}")

        # Calcular valores simulados
        btc_price = safe_float(market_data.get("BTCUSDT", {}).get("close", 50000))
        eth_price = safe_float(market_data.get("ETHUSDT", {}).get("close", 3000))
        simulated_total = (simulated_btc * btc_price) + (simulated_eth * eth_price) + simulated_usdt

        logger.info(f"   Estado simulado: BTC={simulated_btc:.6f}, ETH={simulated_eth:.3f}, USDT={simulated_usdt:.2f}, Total={simulated_total:.2f}")

        # Verificar si hay problemas
        total_change = simulated_total - current_total
        if abs(total_change) > 1000:  # Cambio muy grande
            logger.error(f"🚨 CAMBIO EXCESIVO DETECTADO: {total_change:+.2f} USDT")
            logger.error("   Posibles causas:")
            logger.error("   - Órdenes duplicadas")
            logger.error("   - Precios incorrectos")
            logger.error("   - Acumulación de fees")
            logger.error("   - Error en cálculo de posiciones")

        # Verificar balances negativos
        if simulated_btc < 0 or simulated_eth < 0 or simulated_usdt < 0:
            logger.error("🚨 BALANCES NEGATIVOS EN SIMULACIÓN:")
            logger.error(f"   BTC: {simulated_btc}, ETH: {simulated_eth}, USDT: {simulated_usdt}")

        return {
            "current_total": current_total,
            "simulated_total": simulated_total,
            "change": total_change,
            "has_problems": abs(total_change) > 1000 or simulated_btc < 0 or simulated_eth < 0 or simulated_usdt < 0
        }

    def update_portfolio_allocation(self):
        """Actualiza la asignación de capital dinámicamente"""

        total_portfolio = self.get_total_value()

        # Límites por símbolo
        max_per_symbol = total_portfolio * 0.3  # Máximo 30% por símbolo

        # Capital disponible para nuevas órdenes
        available_trading_capital = self.get_balance("USDT") * 0.8  # Usar 80% máximo

        logger.info(f"💰 Portfolio Allocation | Total: ${total_portfolio:.2f} | Available: ${available_trading_capital:.2f}")

        return available_trading_capital, max_per_symbol

    def save_state(self):
        """Guarda el estado actual del portfolio (alias para save_to_json)"""
        self.save_to_json()

    def get_position_age_seconds(self, symbol: str) -> float:
        """
        SOLUTION 2: Get position age in seconds for rebalance grace period

        Returns:
            Age in seconds since position was last bought, or float('inf') if never bought
        """
        if symbol in self.position_age:
            current_time = datetime.now().timestamp()
            age_seconds = current_time - self.position_age[symbol]
            return max(0, age_seconds)  # Ensure non-negative
        return float('inf')  # Never bought = infinite age

    def can_rebalance_position(self, symbol: str) -> bool:
        """
        SOLUTION 2: Check if position can be rebalanced based on age

        Returns:
            True if position is old enough to be rebalanced, False if too new
        """
        age_seconds = self.get_position_age_seconds(symbol)
        can_rebalance = age_seconds >= self.MIN_HOLD_TIME

        if not can_rebalance:
            logger.info(f"⏰ SOLUTION 2: REBALANCE SKIPPED for {symbol} - Age: {age_seconds:.1f}s < {self.MIN_HOLD_TIME}s grace period")
        else:
            logger.debug(f"✅ SOLUTION 2: REBALANCE ALLOWED for {symbol} - Age: {age_seconds:.1f}s >= {self.MIN_HOLD_TIME}s")

        return can_rebalance

    def log_status(self):
        """Registra el estado actual del portfolio"""
        total_value = self.get_total_value()
        btc_balance = self.get_balance("BTCUSDT")
        eth_balance = self.get_balance("ETHUSDT")
        usdt_balance = self.get_balance("USDT")

        logger.info(f"📊 Portfolio Status - Total: {total_value:.2f} USDT")
        logger.info(f"   BTC: {btc_balance:.6f}, ETH: {eth_balance:.4f}, USDT: {usdt_balance:.2f}")
        logger.info(f"   Peak Value: {self.peak_value:.2f}, Total Fees: {self.total_fees:.4f}")


# EJEMPLOS DE USO DEL PORTFOLIO MANAGER
"""
Ejemplos de uso del PortfolioManager con modos duales:

# 1. MODO SIMULADO (para backtesting - RECOMENDADO)
from core.portfolio_manager import PortfolioManager

# Inicializar para backtesting con portfolio COMPLETAMENTE limpio
pm_simulated = PortfolioManager(
    mode="simulated",
    initial_balance=3000.0,
    symbols=['BTCUSDT', 'ETHUSDT']
)

# FORZAR RESET COMPLETO para asegurar estado limpio
pm_simulated.force_clean_reset()

# Verificar que está limpio
print("VERIFICACIÓN ESTADO LIMPIO:")
print(f"  BTC: {pm_simulated.get_balance('BTCUSDT')} (debe ser 0.0)")
print(f"  ETH: {pm_simulated.get_balance('ETHUSDT')} (debe ser 0.0)")
print(f"  USDT: {pm_simulated.get_balance('USDT')} (debe ser 3000.0)")

# 2. MODO LIVE (para runtime/production - CUIDADO)
from l1_operational.binance_client import BinanceClient

# Inicializar cliente de Binance
binance_client = BinanceClient()

# Inicializar para producción con sincronización real
pm_live = PortfolioManager(
    mode="live",
    client=binance_client,
    symbols=['BTCUSDT', 'ETHUSDT']
)

# El portfolio se sincroniza automáticamente con tus posiciones reales en Binance
print("Portfolio sincronizado con Binance:", pm_live.get_portfolio_state())

# ⚠️ ADVERTENCIA: En modo live, reset() sincronizará con exchange, no limpiará

# 3. USO EN BACKTESTING (ya integrado)
# En backtesting/hrm_tester.py:
portfolio_manager = PortfolioManager(
    mode="simulated",
    initial_balance=initial_capital,
    symbols=symbols
)
portfolio_manager.force_clean_reset()  # ✅ GARANTIZA ESTADO LIMPIO

# 4. USO EN RUNTIME/PRODUCCIÓN
# En main.py o cualquier script de producción:
pm = PortfolioManager(mode="live", client=binance_client)
# ⚠️ NUNCA uses reset() en modo live - perderías posiciones reales

# 5. MÉTODOS DISPONIBLES
# Obtener balance de un símbolo
btc_balance = pm.get_balance("BTCUSDT")
usdt_balance = pm.get_balance("USDT")

# Obtener estado completo
portfolio_state = pm.get_portfolio_state()

# Calcular valor total con datos de mercado
total_value = pm.get_total_value(market_data)

# Actualizar desde órdenes ejecutadas
pm.update_from_orders(orders, market_data)

# Resetear portfolio (modo simulated) o sincronizar (modo live)
pm.reset()

# Forzar reset completo (solo modo simulated)
pm.force_clean_reset()

# Log status detallado
pm.log_status()

# Guardar en CSV
pm.save_to_csv(cycle_id=123)
"""
