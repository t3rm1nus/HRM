# l2_tactic/risk_controls/manager.py

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from ..config import L2Config
from ..models import TacticalSignal, PositionSize, MarketFeatures
from .alerts import AlertType, RiskAlert, RiskLevel
from .positions import RiskPosition
from .stop_losses import DynamicStopLoss, StopLossOrder
from .portfolio import PortfolioRiskManager

logger = logging.getLogger(__name__)


class RiskControlManager:
    """
    Orquesta el riesgo táctico:
      - Stop dinámico y trailing
      - Chequeos pre-trade (liquidez, correlación, límites de cartera, drawdown de estrategia/señal)
      - Seguimiento de posiciones y disparo de SL/TP
      - Métricas y alertas agregadas
    """

    def __init__(self, config: L2Config):
        self.config = config
        self.stop_loss_manager = DynamicStopLoss(config)
        self.portfolio_manager = PortfolioRiskManager(config)
        self.current_positions: Dict[str, RiskPosition] = {}
        self.active_alerts: List[RiskAlert] = []

        # Drawdown por estrategia / señal
        self.strategy_equity: Dict[str, List[tuple[datetime, float]]] = {}  # strategy_id -> [(ts, equity)]
        self.signal_equity: Dict[str, List[tuple[datetime, float]]] = {}    # signal_id   -> [(ts, equity)]

        # Límites específicos
        self.max_signal_drawdown = getattr(config, "max_signal_drawdown", 0.20)
        self.max_strategy_drawdown = getattr(config, "max_strategy_drawdown", 0.25)

        # Liquidez
        self.min_liquidity_notional = getattr(config, "min_liquidity_notional", 1_000.0)  # Reducir aún más
        self.min_liquidity_ratio = getattr(config, "min_liquidity_ratio", 0.005)  #

        logger.info("Initialized RiskControlManager")


    def generate_risk_signals(self, market_data: dict, portfolio_data: dict) -> List[TacticalSignal]:
        logger.debug(f"🛡️ Generando señales de riesgo - Mercado: {market_data}, Portfolio: {portfolio_data}")
        risk_signals = []
        alerts = self.evaluate_pre_trade_risk(market_data, portfolio_data)
        
        for symbol in market_data.keys():
            liquidity = market_data[symbol].get("volume", {}).get("volume", 0.0) * market_data[symbol].get("ohlcv", {}).get("close", 0.0)
            logger.debug(f"🛡️ {symbol}: Liquidez={liquidity:.2f}, Min_liquidity_notional={self.min_liquidity_notional}, Min_liquidity_ratio={self.min_liquidity_ratio}")
            if liquidity < self.min_liquidity_notional:
                risk_signals.append(TacticalSignal(
                    symbol=symbol,
                    signal_type='low_liquidity',
                    strength=0.8,
                    confidence=0.9,
                    side='sell',
                    features={'liquidity': liquidity},
                    timestamp=datetime.now().timestamp(),
                    source='risk',  # Añadir source
                    metadata={'reason': 'low liquidity', 'threshold': self.min_liquidity_notional}
                ))
                logger.debug(f"🛡️ Señal de riesgo generada para {symbol}: low_liquidity")
        
        for alert in alerts:
            logger.debug(f"🛡️ Alerta generada: {alert.alert_type}, Severidad: {alert.severity}, Mensaje: {alert.message}")
            risk_signals.append(TacticalSignal(
                symbol=alert.symbol,
                signal_type=alert.alert_type,
                strength=alert.severity,
                confidence=0.9,
                side='sell',
                features={'alert': alert.message},
                timestamp=datetime.now().timestamp(),
                source='risk',  # Añadir source
                metadata={'reason': alert.message}
            ))
        
        logger.info(f"🛡️ Señales de riesgo generadas: {len(risk_signals)}")
        return risk_signals
    # ----- helpers -----

    @staticmethod
    def _last_equity(values: List[tuple[datetime, float]]) -> float:
        return values[-1][1] if values else 0.0

    @staticmethod
    def _max_dd(values: List[tuple[datetime, float]]) -> float:
        if len(values) < 2:
            return 0.0
        series = [v for _, v in values]
        peak = series[0]
        mdd = 0.0
        for v in series[1:]:
            peak = max(peak, v)
            mdd = max(mdd, (peak - v) / peak if peak > 0 else 0.0)
        return mdd

    # ----- liquidez -----

    def _check_liquidity(
        self,
        symbol: str,
        position_size_notional: float,
        mf: MarketFeatures
    ) -> Optional[RiskAlert]:
        """
        Valida liquidez mínima (notional y ratio contra volumen/liq).
        Busca en MarketFeatures los campos comunes:
          - liquidity_usd, volume_24h_usd, rolling_volume_usd, book_liquidity_usd
        Si no hay, intenta aproximar: last_volume * last_close.
        """
        liqu_candidates = [
            getattr(mf, "liquidity_usd", None),
            getattr(mf, "book_liquidity_usd", None),
            getattr(mf, "rolling_volume_usd", None),
            getattr(mf, "volume_24h_usd", None),
        ]
        liquidity_usd = next((x for x in liqu_candidates if isinstance(x, (int, float)) and x is not None), None)

        if liquidity_usd is None:
            try:
                # fallback crudo
                last_vol = float(getattr(mf, "last_volume", 0) or 0)
                last_px = float(getattr(mf, "last_close", 0) or 0)
                liquidity_usd = last_vol * last_px
            except Exception:
                liquidity_usd = 0.0

        if liquidity_usd <= 0:
            return RiskAlert(
                alert_type=AlertType.LIQUIDITY_INSUFFICIENT,
                severity=RiskLevel.CRITICAL,
                symbol=symbol,
                message="No liquidity data available",
                current_value=0.0,
                threshold=float(self.min_liquidity_notional),
                timestamp=datetime.utcnow(),
                metadata={"position_notional": position_size_notional},
            )

        # Demasiado grande para la liquidez o demasiado pequeño para ejecutar de forma eficiente
        if position_size_notional > liquidity_usd * self.min_liquidity_ratio or position_size_notional < self.min_liquidity_notional:
            severity = RiskLevel.HIGH if position_size_notional > liquidity_usd * self.min_liquidity_ratio else RiskLevel.MODERATE
            return RiskAlert(
                alert_type=AlertType.LIQUIDITY_INSUFFICIENT,
                severity=severity,
                symbol=symbol,
                message=f"Position notional {position_size_notional:.0f} vs liquidity {liquidity_usd:.0f}",
                current_value=float(position_size_notional / max(1.0, liquidity_usd)),
                threshold=float(self.min_liquidity_ratio),
                timestamp=datetime.utcnow(),
                metadata={"liquidity_usd": liquidity_usd},
            )
        return None

    # ----- drawdown entidad (estrategia / señal) -----

    def _check_drawdown_entity(
        self,
        entity_id: str,
        equity_history: Dict[str, List[tuple[datetime, float]]],
        limit: float,
        entity_name: str,
        symbol: str
    ) -> Optional[RiskAlert]:
        values = equity_history.get(entity_id, [])
        dd = self._max_dd(values)
        if dd > limit:
            return RiskAlert(
                alert_type=AlertType.STRATEGY_DRAWDOWN if entity_name == "strategy" else AlertType.SIGNAL_DRAWDOWN,
                severity=RiskLevel.CRITICAL,
                symbol=symbol,
                message=f"{entity_name.capitalize()} drawdown {dd:.1%} exceeds limit {limit:.0%}",
                current_value=dd,
                threshold=limit,
                timestamp=datetime.utcnow(),
                metadata={"entity_id": entity_id},
            )
        return None

    def update_strategy_equity(self, strategy_id: str, equity: float):
        self.strategy_equity.setdefault(strategy_id, []).append((datetime.utcnow(), float(equity)))
        # mantener 90 días
        cutoff = datetime.utcnow() - timedelta(days=90)
        self.strategy_equity[strategy_id] = [(t, v) for t, v in self.strategy_equity[strategy_id] if t >= cutoff]

    def update_signal_equity(self, signal_id: str, equity: float):
        self.signal_equity.setdefault(signal_id, []).append((datetime.utcnow(), float(equity)))
        cutoff = datetime.utcnow() - timedelta(days=30)
        self.signal_equity[signal_id] = [(t, v) for t, v in self.signal_equity[signal_id] if t >= cutoff]

    # ----- pre-trade -----

    def evaluate_pre_trade_risk(
        self,
        signal: TacticalSignal,
        position_size: PositionSize,
        market_features: MarketFeatures,
        portfolio_state: Dict,
        correlation_matrix: Optional["pd.DataFrame"] = None   # type: ignore[name-defined]
    ) -> tuple[bool, List[RiskAlert], Optional[PositionSize]]:
        """
        Devuelve: (allow_trade, alerts, adjusted_position_size or None)
        - Inyecta SL/TP si faltan (con DynamicStopLoss)
        - Aplica reducción si hay alertas HIGH
        - Bloquea si hay CRITICAL
        """
        alerts: List[RiskAlert] = []
        # Copia defensiva (PositionSize suele ser dataclass con .asdict(); si no, usamos attrs directos)
        try:
            adjusted = PositionSize(**position_size.asdict())  # type: ignore[attr-defined]
        except Exception:
            # fallback si no tiene .asdict()
            adjusted = PositionSize(
                symbol=position_size.symbol,
                side=position_size.side,
                price=position_size.price,
                size=position_size.size,
                notional=getattr(position_size, "notional", position_size.size * signal.price),
                risk_amount=getattr(position_size, "risk_amount", 0.0),
                kelly_fraction=getattr(position_size, "kelly_fraction", 0.0),
                vol_target_leverage=getattr(position_size, "vol_target_leverage", 1.0),
                max_loss=getattr(position_size, "max_loss", 0.0),
                stop_loss=getattr(position_size, "stop_loss", None),
                take_profit=getattr(position_size, "take_profit", None),
                leverage=getattr(position_size, "leverage", 1.0),
                margin_required=getattr(position_size, "margin_required", 0.0),
                metadata=getattr(position_size, "metadata", {}),
            )

        # (0) Liquidez
        notional = float(adjusted.notional or (adjusted.size * signal.price))
        liq_alert = self._check_liquidity(signal.symbol, notional, market_features)
        if liq_alert:
            alerts.append(liq_alert)

        # (1) correlaciones
        allow_corr, corr_alerts = self.portfolio_manager.check_correlation_risk(
            signal, self.current_positions, correlation_matrix
        )
        alerts.extend(corr_alerts)

        # (2) límites de portfolio
        total_capital = float(portfolio_state.get("total_capital", 100_000.0))
        daily_pnl = float(portfolio_state.get("daily_pnl", 0.0))
        alerts.extend(self.portfolio_manager.check_portfolio_limits(self.current_positions, total_capital, daily_pnl))

        # (3) Drawdown por estrategia / señal (si hay equity precargado)
        if getattr(signal, "strategy_id", None):
            strategy_alert = self._check_drawdown_entity(
                signal.strategy_id, self.strategy_equity, self.max_strategy_drawdown, "strategy", signal.symbol
            )
            if strategy_alert:
                alerts.append(strategy_alert)

        if getattr(signal, "id", None):  # si TacticalSignal lleva id único
            sig_alert = self._check_drawdown_entity(
                signal.id, self.signal_equity, self.max_signal_drawdown, "signal", signal.symbol
            )
            if sig_alert:
                alerts.append(sig_alert)

        # (4) Asegurar STOP-LOSS + TP si faltan
        if adjusted.stop_loss is None:
            rp = RiskPosition(
                symbol=signal.symbol,
                size=adjusted.size if signal.is_long() else -adjusted.size,
                entry_price=signal.price,
                current_price=signal.price,
                unrealized_pnl=0.0,
                unrealized_pnl_pct=0.0,
            )
            computed = self.stop_loss_manager.calculate_initial_stop(signal, market_features, rp)
            adjusted.stop_loss = computed
            signal.stop_loss = computed
            logger.info(f"[RISK] Assigned initial SL for {signal.symbol}: {computed:.8f}")

        if adjusted.take_profit is None:
            tp = self.stop_loss_manager.suggest_take_profit(signal, adjusted.stop_loss)
            adjusted.take_profit = tp
            signal.take_profit = tp

        # (5) ajuste por severidad: si hay HIGH -> 50%; CRITICAL -> bloquear
        if any(a.severity == RiskLevel.HIGH for a in alerts):
            adjusted.size *= 0.5
            adjusted.notional *= 0.5
            adjusted.risk_amount *= 0.5
            try:
                adjusted.metadata["risk_adjustment"] = "reduced_50pct_high_risk"
            except Exception:
                pass
            logger.warning(f"Reduced size for {signal.symbol} due to high risk alerts")

        allow_trade = (
            allow_corr
            and not any(a.severity == RiskLevel.CRITICAL for a in alerts)
        )
        self.active_alerts.extend(alerts)
        return allow_trade, alerts, (adjusted if allow_trade else None)

    # ----- on-trade / tracking -----

    def add_position(self, signal: TacticalSignal, position_size: PositionSize, mf: MarketFeatures):
        rp = RiskPosition(
            symbol=signal.symbol,
            size=position_size.size if signal.is_long() else -position_size.size,
            entry_price=signal.price,
            current_price=signal.price,
            unrealized_pnl=0.0,
            unrealized_pnl_pct=0.0,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            risk_amount=position_size.risk_amount,
        )
        self.current_positions[signal.symbol] = rp

        # Stop inicial (si no vino en signal)
        if rp.stop_loss is None:
            rp.stop_loss = self.stop_loss_manager.calculate_initial_stop(signal, mf, rp)

        # Registrar stop dinámico (trailing por defecto)
        self.stop_loss_manager.active_stops[signal.symbol] = StopLossOrder(
            symbol=signal.symbol,
            stop_price=float(rp.stop_loss),
            original_price=float(rp.stop_loss),
            entry_price=float(rp.entry_price),
            position_size=abs(float(rp.size)),
            stop_type="trailing",
            last_updated=datetime.utcnow(),
        )
        logger.info(f"Position added to risk tracking: {signal.symbol} size={rp.size:.6f} SL={rp.stop_loss:.6f} TP={rp.take_profit}")

    def remove_position(self, symbol: str):
        self.current_positions.pop(symbol, None)
        self.stop_loss_manager.active_stops.pop(symbol, None)
        logger.info(f"Position removed from risk tracking: {symbol}")

    def monitor_existing_positions(self, price_data: Dict[str, float], portfolio_value: float) -> List[RiskAlert]:
        alerts: List[RiskAlert] = []
        for sym, pos in list(self.current_positions.items()):
            px = float(price_data.get(sym, 0.0) or 0.0)
            if px <= 0:
                continue

            pos.current_price = px
            if pos.size > 0:
                pos.unrealized_pnl = (px - pos.entry_price) * pos.size
            else:
                pos.unrealized_pnl = (pos.entry_price - px) * abs(pos.size)

            denom = max(1e-9, pos.entry_price * abs(pos.size))
            pos.unrealized_pnl_pct = pos.unrealized_pnl / denom

            # excursiones
            pos.max_adverse_excursion = min(pos.max_adverse_excursion, pos.unrealized_pnl)
            pos.max_favorable_excursion = max(pos.max_favorable_excursion, pos.unrealized_pnl)

            # SL: trigger
            sl_alert = self._check_stop_loss_trigger(sym, pos, px)
            if sl_alert:
                alerts.append(sl_alert)

            # SL: trailing update
            self.stop_loss_manager.update_trailing_stop(sym, px, pos)

            # TP
            if pos.take_profit:
                tp_hit = (pos.size > 0 and px >= pos.take_profit) or (pos.size < 0 and px <= pos.take_profit)
                if tp_hit:
                    alerts.append(
                        RiskAlert(
                            alert_type=AlertType.TAKE_PROFIT,
                            severity=RiskLevel.LOW,
                            symbol=sym,
                            message=f"Take profit triggered at {px:.6f}",
                            current_value=px,
                            threshold=float(pos.take_profit),
                            timestamp=datetime.utcnow(),
                        )
                    )

        # metrics de portfolio (opcional)
        self.portfolio_manager.update_portfolio_value(float(portfolio_value))
        return alerts

    def _check_stop_loss_trigger(self, symbol: str, position: RiskPosition, price: float) -> Optional[RiskAlert]:
        if position.stop_loss is None:
            return None
        trig = (position.size > 0 and price <= position.stop_loss) or (position.size < 0 and price >= position.stop_loss)
        if not trig:
            return None
        return RiskAlert(
            alert_type=AlertType.STOP_LOSS,
            severity=RiskLevel.HIGH,
            symbol=symbol,
            message=f"Stop loss triggered at {price:.6f}",
            current_value=price,
            threshold=float(position.stop_loss),
            timestamp=datetime.utcnow(),
            metadata={"position_size": position.size, "unrealized_pnl": position.unrealized_pnl},
        )
