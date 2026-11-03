# -*- coding: utf-8 -*-
"""
Backtest OrderFlow Signal Generator — CORREGIDO
================================================
Correcciones aplicadas:
  1. pnl en Trade es NETO (incluye fee entrada + fee salida)
  2. profit_factor, win_rate calculados sobre PnL NETO
  3. capital se actualiza con pnl_net directamente (sin doble-descuento de fees)
  4. Nuevas métricas: profit_factor_net, avg_profit_per_trade_net, breakeven_winrate
  5. Test D añadido: holding time mínimo de 5 min para reducir rotación de fees
  6. Cobertura: usa TODOS los CSV disponibles, no solo el último

Ejecutar: python backtesting/backtest_orderflow_signal.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import json

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).parent.parent
DATA_DIR  = REPO_ROOT / "backtesting" / "data" / "orderflow"
RESULTS_DIR = REPO_ROOT / "backtesting" / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ── Dataclasses ───────────────────────────────────────────────────────────────
@dataclass
class Trade:
    """Representa un trade individual con PnL NETO (fees incluidas)."""
    symbol: str
    entry_time: datetime
    exit_time: Optional[datetime] = None
    entry_price: float = 0.0
    exit_price: float = 0.0
    size: float = 0.0          # unidades del activo
    position_value: float = 0.0  # USDT invertidos en la entrada
    side: str = "long"
    pnl_gross: float = 0.0     # sin fees
    fee_entry: float = 0.0
    fee_exit: float = 0.0
    pnl_net: float = 0.0       # gross - fee_entry - fee_exit  ← métrica correcta
    pnl_pct: float = 0.0       # basado en pnl_net / position_value
    exit_reason: str = ""
    entry_imbalance: float = 0.0
    max_pnl_pct: float = 0.0
    min_pnl_pct: float = 0.0

    @property
    def hold_time_minutes(self) -> float:
        if self.exit_time is None:
            return 0.0
        return (self.exit_time - self.entry_time).total_seconds() / 60.0

    @property
    def is_win(self) -> bool:
        return self.pnl_net > 0  # ganador solo si net > 0


@dataclass
class BacktestResult:
    """Resultados completos del backtest — métricas NETAS."""
    # Configuración
    threshold_strong: float = 0.0
    threshold_weak: float = 0.0
    required_consecutive: int = 0
    min_hold_minutes: float = 0.0

    # Trades
    trades: List[Trade] = field(default_factory=list)

    # Capital
    initial_capital: float = 3000.0
    final_capital: float = 3000.0

    # Métricas netas
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    profit_factor_net: float = 0.0   # ← la métrica real
    return_pct: float = 0.0
    avg_profit_per_trade_net: float = 0.0
    breakeven_winrate: float = 0.0   # win rate mínimo para cubrir fees

    # Nuevas métricas: movimiento de ganadores/perdedores y fees
    avg_winner_pct: float = 0.0      # movimiento medio de trades ganadores
    avg_loser_pct: float = 0.0       # movimiento medio de trades perdedores
    fee_pct_of_avg_move: float = 0.0 # fees como % del movimiento medio

    # Otras
    avg_hold_time_min: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    directional_accuracy: float = 0.0
    total_fees: float = 0.0

    def calculate_metrics(self):
        self.total_trades = len(self.trades)
        if self.total_trades == 0:
            return

        # ── Win / Loss sobre PnL NETO ──────────────────────────────────────
        self.winning_trades = sum(1 for t in self.trades if t.is_win)
        self.losing_trades  = self.total_trades - self.winning_trades
        self.win_rate = self.winning_trades / self.total_trades * 100

        total_profit_net = sum(t.pnl_net for t in self.trades if t.is_win)
        total_loss_net   = abs(sum(t.pnl_net for t in self.trades if not t.is_win))
        self.profit_factor_net = (
            total_profit_net / total_loss_net if total_loss_net > 0 else float('inf')
        )

        # ── Capital final (pnl_net ya incluye fees, no sumar nada más) ──────
        self.final_capital = self.initial_capital + sum(t.pnl_net for t in self.trades)
        self.return_pct    = (self.final_capital - self.initial_capital) / self.initial_capital * 100

        # ── PnL medio por trade ───────────────────────────────────────────
        self.avg_profit_per_trade_net = np.mean([t.pnl_net for t in self.trades])

        # ── Fees totales ──────────────────────────────────────────────────
        self.total_fees = sum(t.fee_entry + t.fee_exit for t in self.trades)

        # ── Breakeven win rate ────────────────────────────────────────────
        # win rate mínimo para que la estrategia sea break-even con las fees actuales
        avg_fee_per_trade = self.total_fees / self.total_trades if self.total_trades else 0
        avg_winner_gross  = (
            np.mean([t.pnl_gross for t in self.trades if t.pnl_gross > 0])
            if any(t.pnl_gross > 0 for t in self.trades) else 1
        )
        avg_loser_gross = abs(
            np.mean([t.pnl_gross for t in self.trades if t.pnl_gross <= 0])
            if any(t.pnl_gross <= 0 for t in self.trades) else 1
        )
        # breakeven: W * avg_winner - (1-W) * avg_loser = avg_fee_per_trade * 2
        # W = (avg_loser + avg_fee*2) / (avg_winner + avg_loser)
        self.breakeven_winrate = (
            (avg_loser_gross + avg_fee_per_trade * 2) / (avg_winner_gross + avg_loser_gross) * 100
            if (avg_winner_gross + avg_loser_gross) > 0 else 50.0
        )

        # ── Métricas de movimiento ganadores/perdedores ───────────────────
        winning_trades_list = [t for t in self.trades if t.pnl_net > 0]
        losing_trades_list = [t for t in self.trades if t.pnl_net <= 0]

        # Calcular % de movimiento relativo al capital invertido
        self.avg_winner_pct = (
            np.mean([(t.exit_price - t.entry_price) / t.entry_price * 100
                     for t in winning_trades_list])
            if winning_trades_list else 0.0
        )
        self.avg_loser_pct = (
            np.mean([abs((t.exit_price - t.entry_price) / t.entry_price * 100)
                     for t in losing_trades_list])
            if losing_trades_list else 0.0
        )

        # Fees como % del movimiento medio total
        avg_move = (self.avg_winner_pct * self.winning_trades +
                    self.avg_loser_pct * self.losing_trades) / self.total_trades
        # Fee round-trip = 0.2% (0.1% entrada + 0.1% salida)
        fee_round_trip = 0.2  # %
        self.fee_pct_of_avg_move = (
            (fee_round_trip / avg_move * 100) if avg_move > 0 else 0.0
        )

        # ── Hold time ────────────────────────────────────────────────────
        self.avg_hold_time_min = np.mean([t.hold_time_minutes for t in self.trades])

        # ── Max Drawdown ─────────────────────────────────────────────────
        self.max_drawdown_pct = self._calculate_max_drawdown()

        # ── Sharpe (por trade, anualizado aproximado) ─────────────────────
        returns = [t.pnl_pct for t in self.trades]
        if len(returns) > 1:
            self.sharpe_ratio = (
                np.mean(returns) / (np.std(returns) + 1e-9) * np.sqrt(252 * 24 * 4)
            )

    def _calculate_max_drawdown(self) -> float:
        capital = self.initial_capital
        peak    = capital
        max_dd  = 0.0
        for trade in self.trades:
            capital += trade.pnl_net   # pnl_net ya es correcto
            if capital > peak:
                peak = capital
            dd = (peak - capital) / peak * 100
            max_dd = max(max_dd, dd)
        return max_dd

    def print_summary(self, test_name: str = ""):
        viable = self.profit_factor_net > 1.0 and self.return_pct > 0
        verdict = "✅ VIABLE" if viable else "❌ NO VIABLE"

        # Alerta si fees consumen más del 50% del movimiento
        fee_alert = ""
        if self.fee_pct_of_avg_move > 50:
            fee_alert = " ⚠️ FEES ALTOS"
        elif self.fee_pct_of_avg_move > 30:
            fee_alert = " ⚡ Fees significativos"

        print(f"\n{'='*70}")
        print(f"📊 BACKTEST RESULTS {test_name}  {verdict}")
        print(f"{'='*70}")
        print(f"Config: umbral={self.threshold_strong}, persistencia={self.required_consecutive}, "
              f"min_hold={self.min_hold_minutes}min")
        print(f"{'-'*70}")
        print(f"Capital:               ${self.initial_capital:.2f} → ${self.final_capital:.2f}  "
              f"({self.return_pct:+.2f}%)")
        print(f"Trades:                {self.total_trades} ({self.winning_trades}W / {self.losing_trades}L)")
        print(f"Win Rate:              {self.win_rate:.1f}%  "
              f"(breakeven necesario: {self.breakeven_winrate:.1f}%)")
        print(f"Profit Factor NETO:    {self.profit_factor_net:.2f}  ← métrica real")
        print(f"Avg PnL/trade NETO:    ${self.avg_profit_per_trade_net:.4f}")
        print(f"Avg Hold Time:         {self.avg_hold_time_min:.1f} min")
        print(f"Max Drawdown:          {self.max_drawdown_pct:.2f}%")
        print(f"Sharpe Ratio:          {self.sharpe_ratio:.2f}")
        print(f"Directional Acc(5min): {self.directional_accuracy:.1f}%")
        print(f"Total Fees:            ${self.total_fees:.2f}  "
              f"({self.total_fees/self.initial_capital*100:.2f}% del capital)")
        print(f"{'-'*70}")
        print(f"📈 Movimiento ganadores: {self.avg_winner_pct:+.3f}%")
        print(f"📉 Movimiento perdedores: -{self.avg_loser_pct:.3f}%")
        print(f"💸 Fee % del movimiento:  {self.fee_pct_of_avg_move:.1f}%{fee_alert}")
        print(f"{'='*70}\n")


# ── Backtester ─────────────────────────────────────────────────────────────────
class OrderFlowBacktester:
    """
    Backtester corregido: PnL neto, profit_factor neto, sin doble-conteo de fees.
    """

    def __init__(
        self,
        threshold_strong: float = 0.30,
        threshold_weak: float   = 0.20,
        required_consecutive: int = 3,
        min_hold_minutes: float = 0.0,   # Test D: tiempo mínimo en posición
        initial_capital: float = 3000.0,
        position_size_pct: float = 0.10,
        fee_pct: float = 0.001,           # 0.1% por lado
        stop_loss_pct: float = 1.5
    ):
        self.threshold_strong     = threshold_strong
        self.threshold_weak       = threshold_weak
        self.required_consecutive = required_consecutive
        self.min_hold_minutes     = min_hold_minutes
        self.initial_capital      = initial_capital
        self.position_size_pct    = position_size_pct
        self.fee_pct              = fee_pct
        self.stop_loss_pct        = stop_loss_pct

        self.result = BacktestResult(
            threshold_strong=threshold_strong,
            threshold_weak=threshold_weak,
            required_consecutive=required_consecutive,
            min_hold_minutes=min_hold_minutes,
            initial_capital=initial_capital
        )

        self.current_capital   = initial_capital
        self.open_trade: Optional[Trade] = None
        self.imbalance_buffer: List[float] = []

    # ── Carga de datos ────────────────────────────────────────────────────────
    def load_data(self, symbol: str = "BTCUSDT") -> Optional[pd.DataFrame]:
        """Carga TODOS los CSV disponibles para el símbolo (no solo el último)."""
        pattern = f"orderflow_{symbol}_*.csv"
        files   = sorted(DATA_DIR.glob(pattern))

        if not files:
            print(f"❌ No se encontraron archivos para {symbol} en {DATA_DIR}")
            return None

        dfs = []
        for f in files:
            df_tmp = pd.read_csv(f)
            dfs.append(df_tmp)

        df = pd.concat(dfs, ignore_index=True)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').drop_duplicates('timestamp').reset_index(drop=True)

        print(f"📁 Cargados {len(files)} archivos para {symbol}: "
              f"{len(df):,} registros  "
              f"({df['timestamp'].iloc[0].date()} → {df['timestamp'].iloc[-1].date()})")
        return df

    # ── Helpers buffer ────────────────────────────────────────────────────────
    def count_consecutive_above(self, threshold: float) -> int:
        count = 0
        for val in reversed(self.imbalance_buffer):
            if val > threshold:
                count += 1
            else:
                break
        return count

    def count_consecutive_below(self, threshold: float) -> int:
        count = 0
        for val in reversed(self.imbalance_buffer):
            if val < threshold:
                count += 1
            else:
                break
        return count

    def calculate_confidence(self, imbalance: float) -> float:
        return min(abs(imbalance) / 0.50, 1.0) * 0.80

    # ── Backtest principal ────────────────────────────────────────────────────
    def run_backtest(self, symbol: str = "BTCUSDT") -> BacktestResult:
        df = self.load_data(symbol)
        if df is None:
            return self.result

        print(f"🔄 Ejecutando backtest: umbral={self.threshold_strong}, "
              f"persistencia={self.required_consecutive}, "
              f"min_hold={self.min_hold_minutes}min ...")

        buy_signals: List[Tuple[int, float]] = []

        for i, row in df.iterrows():
            timestamp = row['timestamp']
            imbalance = row['bid_ask_imbalance']
            price     = row['weighted_mid_price']

            if pd.isna(price) or pd.isna(imbalance):
                continue

            # Actualizar buffer (máx 10 lecturas)
            self.imbalance_buffer.append(imbalance)
            if len(self.imbalance_buffer) > 10:
                self.imbalance_buffer.pop(0)

            # ── Gestión de trade abierto ──────────────────────────────────
            if self.open_trade is not None:
                current_pnl_pct = (
                    (price - self.open_trade.entry_price) / self.open_trade.entry_price * 100
                )
                self.open_trade.max_pnl_pct = max(self.open_trade.max_pnl_pct, current_pnl_pct)
                self.open_trade.min_pnl_pct = min(self.open_trade.min_pnl_pct, current_pnl_pct)

                hold_elapsed = (timestamp - self.open_trade.entry_time).total_seconds() / 60.0
                min_hold_ok  = hold_elapsed >= self.min_hold_minutes

                # Stop loss (siempre activo)
                if current_pnl_pct <= -self.stop_loss_pct:
                    self._close_trade(timestamp, price, "stop_loss")
                    continue

                # Señal SELL solo si se cumple min_hold
                if min_hold_ok and imbalance < -self.threshold_strong:
                    consecutive = self.count_consecutive_below(-self.threshold_weak)
                    if consecutive >= self.required_consecutive:
                        self._close_trade(timestamp, price, "signal_sell")
                        continue

            # ── Señal BUY ─────────────────────────────────────────────────
            else:
                if imbalance > self.threshold_strong:
                    consecutive = self.count_consecutive_above(self.threshold_weak)
                    if consecutive >= self.required_consecutive:
                        self._open_trade(timestamp, price, imbalance)
                        buy_signals.append((i, price))

        # Cerrar trade abierto al final
        if self.open_trade is not None:
            last = df.iloc[-1]
            self._close_trade(last['timestamp'], last['weighted_mid_price'], "end_of_data")

        # Métricas
        self.result.directional_accuracy = self._calculate_directional_accuracy(df, buy_signals)
        self.result.calculate_metrics()
        return self.result

    # ── Abrir / cerrar trades ─────────────────────────────────────────────────
    def _open_trade(self, timestamp: datetime, price: float, imbalance: float):
        position_value = self.current_capital * self.position_size_pct
        size           = position_value / price
        fee_entry      = position_value * self.fee_pct

        # Descontar fee de entrada del capital AHORA
        self.current_capital -= fee_entry

        self.open_trade = Trade(
            symbol="BTCUSDT",
            entry_time=timestamp,
            entry_price=price,
            size=size,
            position_value=position_value,
            side="long",
            entry_imbalance=imbalance,
            fee_entry=fee_entry
        )

    def _close_trade(self, timestamp: datetime, price: float, reason: str):
        if self.open_trade is None:
            return

        t = self.open_trade

        t.exit_time   = timestamp
        t.exit_price  = price
        t.exit_reason = reason

        # PnL bruto
        t.pnl_gross = (price - t.entry_price) * t.size

        # Fee de salida sobre el valor de salida
        exit_value = t.size * price
        t.fee_exit = exit_value * self.fee_pct

        # PnL NETO = bruto - fee_salida  (fee_entrada ya fue descontada al abrir)
        t.pnl_net = t.pnl_gross - t.fee_exit

        # pnl_pct basado en el capital arriesgado (position_value)
        t.pnl_pct = t.pnl_net / t.position_value * 100

        # Actualizar capital con pnl_gross - fee_exit
        # (fee_entry ya fue descontada en _open_trade, no restar de nuevo)
        self.current_capital += t.pnl_gross - t.fee_exit

        self.result.trades.append(t)
        self.open_trade = None

    # ── Directional accuracy ──────────────────────────────────────────────────
    def _calculate_directional_accuracy(
        self, df: pd.DataFrame, buy_signals: List[Tuple[int, float]]
    ) -> float:
        if not buy_signals:
            return 0.0

        correct = 0
        total   = 0

        for idx, entry_price in buy_signals:
            entry_time  = df.iloc[idx]['timestamp']
            target_time = entry_time + timedelta(minutes=5)
            future_rows = df[df['timestamp'] >= target_time]
            if len(future_rows) == 0:
                continue
            future_price = future_rows.iloc[0]['weighted_mid_price']
            if future_price > entry_price:
                correct += 1
            total += 1

        return (correct / total * 100) if total > 0 else 0.0


# ── Tests ──────────────────────────────────────────────────────────────────────
def run_robustness_tests():
    print("\n" + "="*80)
    print("🧪 TEST DE ROBUSTEZ — OrderFlow Signal Generator (PnL NETO)")
    print("="*80)

    configs = [
        # (nombre,          strong, weak, consec, min_hold_min, stop_loss)
        ("Test A (Base)",       0.30, 0.20, 3, 0.0,  1.5),
        ("Test B (Sensible)",   0.25, 0.15, 3, 0.0,  1.5),
        ("Test C (Estricto)",   0.35, 0.25, 4, 0.0,  1.5),
        ("Test D (5min hold)",  0.30, 0.20, 3, 5.0,  1.5),   # reduce rotación / fees
        ("Test E (30min hold)", 0.30, 0.20, 3, 30.0, 1.5),   # hipótesis: esperar 30min supera fees
        ("Test F (60min hold)", 0.30, 0.20, 3, 60.0, 2.0),   # hipótesis: esperar 60min con SL 2%
    ]

    results = []

    for test_name, strong, weak, consecutive, min_hold, sl in configs:
        print(f"\n{'='*70}")
        print(f"🚀 {test_name}: umbral={strong}, persistencia={consecutive}, "
              f"min_hold={min_hold}min, SL={sl}%")
        print(f"{'='*70}")

        backtester = OrderFlowBacktester(
            threshold_strong=strong,
            threshold_weak=weak,
            required_consecutive=consecutive,
            min_hold_minutes=min_hold,
            stop_loss_pct=sl
        )

        result = backtester.run_backtest("BTCUSDT")
        result.print_summary(test_name)
        results.append((test_name, result))

    # ── Tabla comparativa ─────────────────────────────────────────────────────
    print("\n" + "="*80)
    print("📊 TABLA COMPARATIVA (métricas NETAS — fees incluidas)")
    print("="*80)
    print(f"{'Test':<22} {'Return%':>9} {'PF_net':>8} {'Win%':>7} "
          f"{'Trades':>7} {'AvgWin%':>8} {'AvgLoss%':>9} {'Fee/Mov%':>9} {'Viable?':>9}")
    print("-"*95)

    viable_count = 0
    for test_name, r in results:
        is_viable = r.profit_factor_net > 1.0 and r.return_pct > 0
        if is_viable:
            viable_count += 1
        tag = "✅ SÍ" if is_viable else "❌ NO"

        print(
            f"{test_name:<22} "
            f"{r.return_pct:>+9.2f} "
            f"{r.profit_factor_net:>8.2f} "
            f"{r.win_rate:>7.1f} "
            f"{r.total_trades:>7} "
            f"{r.avg_winner_pct:>+8.3f} "
            f"{r.avg_loser_pct:>9.3f} "
            f"{r.fee_pct_of_avg_move:>9.1f} "
            f"{tag:>9}"
        )

    print("-"*95)

    print(f"\n🎯 VEREDICTO FINAL:")
    if viable_count == 0:
        print("❌ NINGÚN test es viable después de fees.")
        print("   La señal tiene edge direccional pero no supera el coste de transacción.")
        print("   → Opciones: aumentar min_hold, reducir fees (cuenta VIP), o señal más selectiva.")
    elif viable_count <= 2:
        print(f"⚠️  {viable_count}/6 tests viables. Edge marginal, muy sensible a fees.")
        print("   → Priorizar Tests E/F (min_hold extendido) si son viables.")
    else:
        print(f"✅ {viable_count}/6 tests viables. La señal supera fees consistentemente.")
        print("   → Usar configuración con mejor return_pct.")

    print("="*80)

    # ── Guardar JSON ──────────────────────────────────────────────────────────
    results_dict = {
        "timestamp": datetime.now().isoformat(),
        "note": "Todas las métricas son NETAS (fees incluidas en pnl)",
        "tests": [
            {
                "name": name,
                "threshold_strong": float(r.threshold_strong),
                "threshold_weak": float(r.threshold_weak),
                "required_consecutive": int(r.required_consecutive),
                "min_hold_minutes": float(r.min_hold_minutes),
                "return_pct": float(r.return_pct),
                "profit_factor_net": float(r.profit_factor_net),
                "win_rate": float(r.win_rate),
                "breakeven_winrate": float(r.breakeven_winrate),
                "total_trades": int(r.total_trades),
                "avg_profit_per_trade_net": float(r.avg_profit_per_trade_net),
                "avg_winner_pct": float(r.avg_winner_pct),
                "avg_loser_pct": float(r.avg_loser_pct),
                "fee_pct_of_avg_move": float(r.fee_pct_of_avg_move),
                "total_fees": float(r.total_fees),
                "max_drawdown_pct": float(r.max_drawdown_pct),
                "sharpe_ratio": float(r.sharpe_ratio),
                "directional_accuracy": float(r.directional_accuracy),
                "is_viable": bool(r.profit_factor_net > 1.0 and r.return_pct > 0)
            }
            for name, r in results
        ],
        "verdict": (
            "VIABLE" if viable_count >= 3
            else "MARGINAL" if viable_count >= 1
            else "NOT_VIABLE"
        )
    }

    output_file = RESULTS_DIR / f"orderflow_backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(results_dict, f, indent=2)

    print(f"\n💾 Resultados guardados en: {output_file}")
    return results


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "="*80)
    print("🌊 ORDERFLOW SIGNAL BACKTEST — versión corregida (PnL NETO)")
    print("="*80)
    print(f"📁 Data directory:    {DATA_DIR}")
    print(f"📁 Results directory: {RESULTS_DIR}")

    if not DATA_DIR.exists():
        print(f"\n❌ ERROR: Directorio de datos no existe: {DATA_DIR}")
        print("   Ejecuta primero el sistema para generar datos de orderflow.")
        return

    run_robustness_tests()
    print("\n✅ Backtest completado!")


if __name__ == "__main__":
    main()
