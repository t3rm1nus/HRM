import asyncio
import math
import os
import socket
from typing import Any, Dict, List, Optional

import aiohttp
import ccxt.async_support as ccxt

from core.logging import logger


class BinanceClient:
    """
    Cliente de Binance para HRM.

    Responsabilidades:
    - Obtener datos públicos reales de Binance.
    - Consultar balances de cuenta.
    - Gestionar órdenes cuando el modo lo permite.
    - Delegar la ejecución PAPER en SimulatedExchangeClient.
    - Gestionar correctamente los recursos asíncronos.

    PAPER nunca ejecuta órdenes reales.
    """

    URL_PUBLICA = "https://api.binance.com/api/v3"
    URL_TESTNET = "https://testnet.binance.vision/api/v3"

    TIMEFRAMES_VALIDOS = {
        "1m", "3m", "5m", "15m", "30m",
        "1h", "2h", "4h", "6h", "8h",
        "12h", "1d", "3d", "1w", "1M"
    }

    def __init__(
        self,
        config_dict: Optional[dict] = None,
        mode: Optional[str] = None,
        simulated_client=None
    ):
        self.config = self._cargar_configuracion(config_dict)

        self.mode = self._normalizar_modo(mode)
        self.paper_mode = self.mode in {"paper", "simulated"}

        self.simulated_client = simulated_client

        self.exchange = None
        self.public_exchange = None
        self._http_session = None
        self._closed = False
        self._session_lock = asyncio.Lock()

        self._validar_credenciales()
        self._inicializar_exchanges()

        logger.info(
            "BinanceClient inicializado: modo=%s, paper=%s",
            self.mode,
            self.paper_mode
        )

    # ------------------------------------------------------------------
    # CONFIGURACIÓN
    # ------------------------------------------------------------------

    @staticmethod
    def _cargar_configuracion(config_dict: Optional[dict]) -> dict:
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            logger.warning("python-dotenv no está disponible")

        configuracion = {
            "BINANCE_API_KEY": os.getenv("BINANCE_API_KEY", ""),
            "BINANCE_API_SECRET": os.getenv("BINANCE_API_SECRET", ""),
            "USE_TESTNET": os.getenv(
                "USE_TESTNET", "false"
            ).lower() == "true",
            "PAPER_MODE": os.getenv(
                "PAPER_MODE", "true"
            ).lower() == "true"
        }

        if config_dict:
            configuracion.update(config_dict)

        return configuracion

    def _normalizar_modo(self, mode: Optional[str]) -> str:
        if mode:
            modo = mode.strip().lower()
        elif self.config.get("PAPER_MODE", True):
            modo = "paper"
        elif self.config.get("USE_TESTNET", False):
            modo = "testnet"
        else:
            modo = "live"

        modos_validos = {"paper", "simulated", "testnet", "live"}

        if modo not in modos_validos:
            raise ValueError(f"Modo de Binance no válido: {modo}")

        return modo

    def _validar_credenciales(self) -> None:
        if self.paper_mode:
            return

        api_key = self.config.get("BINANCE_API_KEY")
        api_secret = self.config.get("BINANCE_API_SECRET")

        if not api_key or not api_secret:
            raise RuntimeError(
                f"El modo {self.mode} requiere credenciales API. "
                "No se permite cambiar silenciosamente a PAPER."
            )

    def _inicializar_exchanges(self) -> None:
        api_key = self.config.get("BINANCE_API_KEY", "")
        api_secret = self.config.get("BINANCE_API_SECRET", "")

        usar_testnet = self.mode == "testnet"

        opciones = {
            "apiKey": api_key,
            "secret": api_secret,
            "enableRateLimit": True,
            "timeout": 15000,
            "options": {
                "defaultType": "spot"
            }
        }

        self.exchange = ccxt.binance(opciones)

        if usar_testnet:
            self.exchange.set_sandbox_mode(True)

        self.public_exchange = ccxt.binance({
            "enableRateLimit": True,
            "timeout": 15000,
            "options": {
                "defaultType": "spot"
            }
        })

    # ------------------------------------------------------------------
    # CLIENTE SIMULADO
    # ------------------------------------------------------------------

    def _obtener_cliente_simulado(self):
        if self.simulated_client is not None:
            return self.simulated_client

        from l1_operational.simulated_exchange_client import (
            SimulatedExchangeClient
        )

        cliente = SimulatedExchangeClient._instance

        if cliente is None:
            raise RuntimeError(
                "SimulatedExchangeClient todavía no está inicializado. "
                "Debe inicializarse desde bootstrap con capital configurado."
            )

        self.simulated_client = cliente
        return cliente

    # ------------------------------------------------------------------
    # SESIONES HTTP
    # ------------------------------------------------------------------

    async def _inicializar_sesion_http(self) -> None:
        async with self._session_lock:
            if self._closed:
                raise RuntimeError("BinanceClient está cerrado")

            if (
                self._http_session is not None
                and not self._http_session.closed
            ):
                return

            conector = aiohttp.TCPConnector(
                family=socket.AF_INET,
                ttl_dns_cache=300,
                limit=30,
                enable_cleanup_closed=True
            )

            timeout = aiohttp.ClientTimeout(
                total=15,
                connect=5,
                sock_read=10
            )

            self._http_session = aiohttp.ClientSession(
                connector=conector,
                timeout=timeout,
                trust_env=True
            )

            logger.info("Sesión HTTP pública inicializada")

    async def _asegurar_sesion(self) -> None:
        if self._closed:
            raise RuntimeError("BinanceClient está cerrado")

        if (
            self._http_session is None
            or self._http_session.closed
        ):
            await self._inicializar_sesion_http()

    async def __aenter__(self):
        await self._asegurar_sesion()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def close(self) -> None:
        if self._closed:
            return

        self._closed = True

        if self._http_session and not self._http_session.closed:
            await self._http_session.close()

        for exchange in (self.exchange, self.public_exchange):
            if exchange is not None:
                try:
                    await exchange.close()
                except Exception:
                    logger.exception(
                        "Error cerrando conexión CCXT"
                    )

        logger.info("BinanceClient cerrado")

    # ------------------------------------------------------------------
    # VALIDACIONES
    # ------------------------------------------------------------------

    @staticmethod
    def _validar_simbolo(symbol: str) -> str:
        if not isinstance(symbol, str) or not symbol.strip():
            raise ValueError("El símbolo no puede estar vacío")

        return symbol.strip().upper()

    @staticmethod
    def _validar_numero_positivo(
        valor: float,
        nombre: str
    ) -> float:
        try:
            numero = float(valor)
        except (TypeError, ValueError):
            raise ValueError(f"{nombre} debe ser numérico")

        if not math.isfinite(numero) or numero <= 0:
            raise ValueError(f"{nombre} debe ser positivo y finito")

        return numero

    # ------------------------------------------------------------------
    # DATOS DE MERCADO
    # ------------------------------------------------------------------

    async def get_klines(
        self,
        symbol: str,
        timeframe: str = "1m",
        limit: int = 50
    ) -> list:
        symbol = self._validar_simbolo(symbol)

        if timeframe not in self.TIMEFRAMES_VALIDOS:
            raise ValueError(
                f"Timeframe no soportado: {timeframe}"
            )

        if isinstance(limit, bool) or not isinstance(limit, int):
            raise ValueError("limit debe ser un entero")

        if not 1 <= limit <= 1000:
            raise ValueError("limit debe estar entre 1 y 1000")

        await self._asegurar_sesion()

        return await self._obtener_klines_reales(
            symbol,
            timeframe,
            limit
        )

    async def _obtener_klines_reales(
        self,
        symbol: str,
        timeframe: str,
        limit: int
    ) -> list:
        url = f"{self.URL_PUBLICA}/klines"

        parametros = {
            "symbol": symbol,
            "interval": timeframe,
            "limit": limit
        }

        for intento in range(3):
            try:
                await self._asegurar_sesion()

                async with self._http_session.get(
                    url,
                    params=parametros
                ) as respuesta:

                    if respuesta.status == 429:
                        raise RuntimeError(
                            "Binance ha limitado las peticiones"
                        )

                    if respuesta.status != 200:
                        contenido = await respuesta.text()

                        raise RuntimeError(
                            f"Binance HTTP {respuesta.status}: "
                            f"{contenido[:300]}"
                        )

                    datos = await respuesta.json()

                    if not isinstance(datos, list):
                        raise RuntimeError(
                            "Respuesta OHLCV no válida"
                        )

                    return datos

            except (aiohttp.ClientError, asyncio.TimeoutError):
                if intento == 2:
                    logger.exception(
                        "No se pudieron obtener klines de %s",
                        symbol
                    )
                    raise

                await asyncio.sleep(2 ** intento)

        raise RuntimeError("No se pudieron obtener datos OHLCV")

    async def get_ticker_price(self, symbol: str) -> float:
        symbol = self._validar_simbolo(symbol)

        await self._asegurar_sesion()

        url = f"{self.URL_PUBLICA}/ticker/price"

        try:
            async with self._http_session.get(
                url,
                params={"symbol": symbol}
            ) as respuesta:

                if respuesta.status != 200:
                    contenido = await respuesta.text()

                    raise RuntimeError(
                        f"Error obteniendo precio: "
                        f"HTTP {respuesta.status}: {contenido[:300]}"
                    )

                datos = await respuesta.json()

                if not isinstance(datos, dict) or "price" not in datos:
                    raise RuntimeError(
                        "Respuesta de ticker inválida"
                    )

                return self._validar_numero_positivo(
                    datos["price"],
                    "Precio de mercado"
                )

        except Exception:
            logger.exception(
                "Error obteniendo precio de %s",
                symbol
            )
            raise

    async def _get_real_price(self, symbol: str) -> float:
        return await self.get_ticker_price(symbol)

    async def _get_real_klines(
        self,
        symbol: str,
        timeframe: str,
        limit: int
    ) -> list:
        return await self.get_klines(
            symbol,
            timeframe,
            limit
        )

    # ------------------------------------------------------------------
    # BALANCES
    # ------------------------------------------------------------------

    async def get_account_balances(self) -> Dict[str, float]:
        if self.paper_mode:
            cliente = self._obtener_cliente_simulado()
            return cliente.get_balances()

        try:
            balance = await self.exchange.fetch_balance()

            resultado = {}

            for tipo in ("free", "used"):
                activos = balance.get(tipo, {})

                for activo, cantidad in activos.items():
                    cantidad = float(cantidad or 0)

                    if cantidad > 0:
                        resultado[activo] = (
                            resultado.get(activo, 0.0) + cantidad
                        )

            return resultado

        except Exception:
            logger.exception("Error consultando balances de Binance")
            raise

    # ------------------------------------------------------------------
    # CONTROL DEL CAPITAL SIMULADO
    # ------------------------------------------------------------------

    def force_reset(
        self,
        initial_balances: Optional[Dict[str, float]] = None
    ) -> None:
        if not self.paper_mode:
            raise RuntimeError(
                "force_reset solamente está permitido en PAPER"
            )

        cliente = self._obtener_cliente_simulado()

        if initial_balances is None:
            raise ValueError(
                "Es obligatorio indicar los balances iniciales"
            )

        cliente.force_reset(initial_balances)

    # ------------------------------------------------------------------
    # ÓRDENES
    # ------------------------------------------------------------------

    async def place_stop_loss_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        stop_price: float,
        limit_price: Optional[float] = None
    ) -> Dict[str, Any]:

        symbol = self._validar_simbolo(symbol)
        quantity = self._validar_numero_positivo(
            quantity,
            "quantity"
        )
        stop_price = self._validar_numero_positivo(
            stop_price,
            "stop_price"
        )

        side = side.lower()

        if side not in {"buy", "sell"}:
            raise ValueError("side debe ser buy o sell")

        if self.paper_mode:
            raise NotImplementedError(
                "Las órdenes stop-loss pendientes requieren un "
                "gestor de órdenes simuladas. No se ejecutarán "
                "como órdenes de mercado."
            )

        if limit_price is not None:
            limit_price = self._validar_numero_positivo(
                limit_price,
                "limit_price"
            )

            return await self.exchange.create_order(
                symbol,
                "STOP_LOSS_LIMIT",
                side,
                quantity,
                limit_price,
                {
                    "stopPrice": stop_price,
                    "timeInForce": "GTC"
                }
            )

        return await self.exchange.create_order(
            symbol,
            "STOP_LOSS",
            side,
            quantity,
            None,
            {
                "stopPrice": stop_price
            }
        )

    async def cancel_order(
        self,
        symbol: str,
        order_id: str
    ) -> bool:

        symbol = self._validar_simbolo(symbol)

        if not order_id:
            raise ValueError("order_id no puede estar vacío")

        if self.paper_mode:
            return False

        try:
            await self.exchange.cancel_order(
                order_id,
                symbol
            )
            return True

        except ccxt.OrderNotFound:
            return False

        except Exception:
            logger.exception(
                "Error cancelando orden %s",
                order_id
            )
            raise

    async def get_open_orders(
        self,
        symbol: Optional[str] = None
    ) -> List[Dict[str, Any]]:

        if self.paper_mode:
            return []

        if symbol is not None:
            symbol = self._validar_simbolo(symbol)

        try:
            return await self.exchange.fetch_open_orders(symbol)

        except Exception:
            logger.exception("Error obteniendo órdenes abiertas")
            raise

    async def place_limit_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        stop_price: Optional[float] = None,
        order_type: str = "LIMIT"
    ) -> Dict[str, Any]:

        symbol = self._validar_simbolo(symbol)
        quantity = self._validar_numero_positivo(
            quantity,
            "quantity"
        )
        price = self._validar_numero_positivo(
            price,
            "price"
        )

        side = side.lower()
        order_type = order_type.upper()

        if side not in {"buy", "sell"}:
            raise ValueError("side debe ser buy o sell")

        if order_type not in {"LIMIT", "MARKET"}:
            raise ValueError("Tipo de orden no soportado")

        if self.paper_mode:
            if order_type == "LIMIT":
                raise NotImplementedError(
                    "Las órdenes LIMIT pendientes no están "
                    "implementadas en el simulador."
                )

            cliente = self._obtener_cliente_simulado()

            resultado = cliente.execute_order(
                symbol=symbol,
                side=side.upper(),
                qty=quantity,
                market_price=price
            )

            return {
                **resultado,
                "status": "filled",
                "simulated": True
            }

        if order_type == "LIMIT":
            return await self.exchange.create_order(
                symbol,
                "limit",
                side,
                quantity,
                price
            )

        return await self.exchange.create_order(
            symbol,
            "market",
            side,
            quantity
        )

    # ------------------------------------------------------------------
    # COMPATIBILIDAD
    # ------------------------------------------------------------------

    def get_market_price(self, symbol: str) -> float:
        """
        Método síncrono obsoleto.

        La obtención de precios debe realizarse mediante
        await get_ticker_price(symbol).

        Nunca devuelve precios aleatorios ni inventados.
        """
        raise RuntimeError(
            "get_market_price() síncrono está obsoleto. "
            "Utiliza await get_ticker_price(symbol)."
        )