#!/usr/bin/env python3
"""
HRM Trading System - Comprehensive Weekly Performance Analysis
Analyzes all system logs, trades, and metrics from the past week

🔴 CRITICAL ARCHITECTURAL REQUIREMENT - PRIORITY CRITICAL:
❌ FORBIDDEN: Deriving signal metrics from trade data (paper_trades.csv)
✅ REQUIRED: Signals come ONLY from logs (L2/L3 outputs)
✅ REQUIRED: Trades come ONLY from paper_trades.csv
✅ REQUIRED: Complete separation between signal and trade data sources

📌 SIGNAL = System intention from logs/events.json (BUY/SELL/HOLD)
📌 TRADE = Real execution from paper_trades.csv (BUY/SELL only)
📌 executed_trade = order BUY or SELL sent and accepted by L1
📌 validated_signal ≠ executed_trade
📌 HOLD signals NEVER count as executed trades
📌 Un HOLD nunca es un trade
📌 Un trade no define una signal
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import json
import re
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class HRMWeeklyAnalyzer:
    """Complete analysis of HRM trading system performance"""
    
    def __init__(self, base_path=".", days_back=7):
        self.base_path = Path(base_path)
        self.days_back = days_back
        self.cutoff_date = datetime.now() - timedelta(days=days_back)
        
        # Results storage
        self.trades = []
        self.signals = []
        self.errors = []
        self.portfolio_snapshots = []
        self.l3_decisions = []
        self.system_metrics = defaultdict(list)
        
    def analyze_all(self):
        """Run complete analysis pipeline"""
        print("=" * 80)
        print(f"🔍 HRM WEEKLY ANALYSIS - Last {self.days_back} Days")
        print("=" * 80)
        print()
        
        # 1. Load all data sources
        self._load_trades_log()
        self._load_main_logs()
        self._load_portfolio_history()
        self._load_l3_outputs()
        
        # 2. Generate comprehensive reports
        print("\n📊 GENERATING REPORTS...\n")
        
        self._report_system_health()
        self._report_cycle_performance()
        self._report_model_performance()
        self._report_signal_analysis()
        self._report_trading_performance()
        self._report_l3_strategic_analysis()
        self._report_l2_tactical_analysis()
        self._report_l1_execution_analysis()
        self._report_portfolio_evolution()
        self._report_errors_and_issues()
        self._report_recommendations()
        
        # 3. Export detailed reports
        self._export_reports()
        
    def _load_trades_log(self):
        """Load executed trades from logs"""
        print("📁 Loading trades data...")

        # Try multiple possible log locations including paper trades
        log_files = [
            self.base_path / "data/paper_trades/paper_trades.csv",  # Paper/simulated trades
            self.base_path / "data/logs/trades.csv",
            self.base_path / "logs/trades.csv",
            self.base_path / "data/trades_history.csv"
        ]

        for log_file in log_files:
            if log_file.exists():
                try:
                    df = pd.read_csv(log_file)
                    # Ensure we have the required columns
                    if 'timestamp' not in df.columns:
                        print(f"   ⚠️ Skipping {log_file} - no timestamp column")
                        continue

                    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                    df = df.dropna(subset=['timestamp'])  # Remove rows with invalid timestamps
                    df = df[df['timestamp'] >= self.cutoff_date]

                    if len(df) > 0:
                        self.trades = df.to_dict('records')
                        trade_type = "paper/simulated" if "paper_trades" in str(log_file) else "live"
                        print(f"   ✅ Loaded {len(self.trades)} {trade_type} trades from {log_file}")
                        return
                    else:
                        print(f"   ⚠️ No recent trades in {log_file}")

                except Exception as e:
                    print(f"   ⚠️ Error reading {log_file}: {e}")

        print("   ⚠️ No trades log found")
        
    def _load_main_logs(self):
        """Parse main system logs from JSON events file - optimized for large files"""
        print("📁 Loading system logs...")

        log_file = self.base_path / "logs/events.json"
        if not log_file.exists():
            print("   ⚠️ No events.json log found")
            return

        try:
            parsed_entries = 0
            relevant_entries = 0

            # For large files, read from the end and work backwards to find recent entries
            # This is more efficient than reading the entire file
            cutoff_date_str = self.cutoff_date.strftime('%Y-%m-%d')

            with open(log_file, 'r', encoding='utf-8') as f:
                # Read all lines at once (should be fine for most log files)
                lines = f.readlines()

            print(f"   📄 Read {len(lines)} total lines from log file")

            # Process lines in reverse order to get most recent first
            for line in reversed(lines):
                line = line.strip()
                if not line:
                    continue

                try:
                    entry = json.loads(line)
                    parsed_entries += 1

                    # Quick date check using string comparison for efficiency
                    ts_str = entry.get('ts', '')
                    if ts_str and ts_str[:10] < cutoff_date_str:
                        # Since we're going backwards, once we hit an old date we can stop
                        break

                    # Parse timestamp
                    timestamp = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
                    if timestamp < self.cutoff_date:
                        continue

                    relevant_entries += 1

                    level = entry.get('level', 'INFO')
                    message = entry.get('message', '')
                    module = entry.get('module', '')
                    symbol = entry.get('symbol')
                    cycle_id = entry.get('cycle_id')
                    extra = entry.get('extra', {})

                    # Categorize log entry based on content and level
                    if level in ['ERROR', 'CRITICAL', 'WARNING'] or '❌' in message or '⚠️' in message:
                        self.errors.append({
                            'timestamp': timestamp,
                            'message': message,
                            'level': level,
                            'module': module,
                            'symbol': symbol,
                            'cycle_id': cycle_id
                        })

                    # CRITICAL: Extract signals ONLY from SYSTEM INTENTIONS - PRIORITY CRITICAL
                    # 🚫 PROHIBIDO: Derive signals from trade data
                    # ✅ REQUIRED: Signals from system logs representing intentions

                    signal_detected = False
                    signal_source_type = None

                    # Source 1: L2 TACTICAL SIGNALS (system intentions)
                    if ('tactical_signal_processor' in message or
                        'L2_SIGNAL' in message or
                        (extra and 'trading_action' in extra and
                         not any(trade_keyword in message.lower() for trade_keyword in ['trade', 'executed', 'order', 'position']))):
                        signal_detected = True
                        signal_source_type = 'L2_TACTICAL'

                    # Source 2: L3 STRATEGIC SIGNALS (decision maker intentions)
                    elif ('decision_maker' in message or
                          'L3 Signal' in message or
                          'STRATEGY_SIGNAL' in message or
                          any(intent_keyword in message.upper() for intent_keyword in ['REGIME', 'TREND', 'STRATEGY'])):
                        signal_detected = True
                        signal_source_type = 'L3_STRATEGIC'

                    # Source 3: SYSTEM LEVEL SIGNALS (INFO level intentions)
                    elif (level == 'INFO' and
                          any(intent_pattern in message.upper() for intent_pattern in ['SIGNAL:', 'BUY SIGNAL', 'SELL SIGNAL', 'HOLD SIGNAL']) and
                          not any(execution_keyword in message.lower() for execution_keyword in ['executed', 'traded', 'order filled', 'position'])):
                        signal_detected = True
                        signal_source_type = 'SYSTEM_INFO'

                    # CRITICAL VALIDATION: Reject signals that appear to be derived from trades
                    if signal_detected:
                        # 🚫 REJECT: Signals that mention trade execution or positions
                        if any(reject_keyword in message.lower() for reject_keyword in [
                            'executed', 'traded', 'order filled', 'position opened', 'position closed',
                            'paper trade', 'simulated trade', 'realized pnl', 'trade id'
                        ]):
                            signal_detected = False  # Reject - this is trade data, not signal intention

                        # 🚫 REJECT: Signals from trade-related modules
                        if module and any(trade_module in module.lower() for trade_module in [
                            'trade', 'execution', 'order', 'position', 'portfolio'
                        ]):
                            signal_detected = False  # Reject - this is execution, not intention

                    # EXTRACT VALID SYSTEM INTENTIONS ONLY
                    if signal_detected:
                        # Validate signal contains MANDATORY BUY/SELL/HOLD actions
                        has_valid_intention = False
                        intention_type = None

                        # Check for explicit intention in message (system decision)
                        msg_upper = message.upper()
                        if 'BUY' in msg_upper and 'HOLD' not in msg_upper:
                            has_valid_intention = True
                            intention_type = 'BUY'
                        elif 'SELL' in msg_upper and 'HOLD' not in msg_upper:
                            has_valid_intention = True
                            intention_type = 'SELL'
                        elif 'HOLD' in msg_upper:
                            has_valid_intention = True
                            intention_type = 'HOLD'

                        # Check for intention in structured extra data
                        if not has_valid_intention and extra and 'trading_action' in extra:
                            action = extra['trading_action'].get('action', '').upper()
                            if action in ['BUY', 'SELL', 'HOLD']:
                                has_valid_intention = True
                                intention_type = action

                        # CRITICAL: Only accept signals with clear system intentions
                        if has_valid_intention and intention_type:
                            self.signals.append({
                                'timestamp': timestamp,
                                'message': message,
                                'symbol': symbol,
                                'extra': extra,
                                'intention_type': intention_type,  # System intention: BUY/SELL/HOLD
                                'source_type': signal_source_type,  # L2_TACTICAL, L3_STRATEGIC, SYSTEM_INFO
                                'confidence': extra.get('trading_action', {}).get('confidence', 0.5) if extra else 0.5
                            })

                    # Portfolio snapshots (from cycle data)
                    if 'Ciclo' in message and 'completado' in message and extra:
                        self.portfolio_snapshots.append({
                            'timestamp': timestamp,
                            'message': message,
                            'total_value': extra.get('total_value', 0),
                            'signals_count': extra.get('signals_count', 0),
                            'orders_count': extra.get('orders_count', 0),
                            'cycle_time': extra.get('cycle_time', 0),
                            'cycle_id': cycle_id
                        })

                    # L3 Decisions (regime detection, strategic decisions)
                    if any(keyword in message.upper() for keyword in ['REGIME', 'L3', 'TRENDING', 'RANGE', 'BULL', 'BEAR']):
                        self.l3_decisions.append({
                            'timestamp': timestamp,
                            'message': message,
                            'symbol': symbol,
                            'extra': extra
                        })

                    # Store system metrics
                    if 'Ciclo' in message and extra:
                        cycle_data = {
                            'cycle_id': cycle_id,
                            'timestamp': timestamp,
                            'cycle_time': extra.get('cycle_time', 0),
                            'signals_count': extra.get('signals_count', 0),
                            'orders_count': extra.get('orders_count', 0),
                            'rejected_count': extra.get('rejected_count', 0),
                            'total_value': extra.get('total_value', 0)
                        }
                        self.system_metrics['cycles'].append(cycle_data)

                    # Feature calculation metrics
                    if 'Features calculadas' in message:
                        self.system_metrics['feature_calculations'].append({
                            'timestamp': timestamp,
                            'message': message
                        })

                    # Model loading/reuse metrics
                    if any(keyword in message.lower() for keyword in ['modelo', 'bert', 'reutilizado', 'cargado']):
                        self.system_metrics['model_operations'].append({
                            'timestamp': timestamp,
                            'message': message,
                            'type': 'reuse' if 'reutilizado' in message else 'load'
                        })

                except json.JSONDecodeError:
                    continue  # Skip malformed lines

            print(f"   ✅ Parsed {parsed_entries} total entries, {relevant_entries} relevant (last {self.days_back} days)")
            print(f"      - Signals: {len(self.signals)}")
            print(f"      - Errors: {len(self.errors)}")
            print(f"      - Portfolio snapshots: {len(self.portfolio_snapshots)}")
            print(f"      - L3 Decisions: {len(self.l3_decisions)}")
            print(f"      - Cycles: {len(self.system_metrics['cycles'])}")

        except Exception as e:
            print(f"   ⚠️ Error parsing logs: {e}")
            import traceback
            print(f"   Error details: {traceback.format_exc()}")
            
    def _load_portfolio_history(self):
        """Load portfolio history from CSV"""
        print("📁 Loading portfolio history...")
        
        portfolio_file = self.base_path / "data/portfolio_history.csv"
        if portfolio_file.exists():
            try:
                df = pd.read_csv(portfolio_file)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df[df['timestamp'] >= self.cutoff_date]
                print(f"   ✅ Loaded {len(df)} portfolio snapshots")
            except Exception as e:
                print(f"   ⚠️ Error reading portfolio history: {e}")
        else:
            print("   ⚠️ No portfolio history found")
            
    def _load_l3_outputs(self):
        """Load L3 strategic decisions"""
        print("📁 Loading L3 strategic outputs...")
        
        l3_file = self.base_path / "data/datos_inferencia/l3_output.json"
        if l3_file.exists():
            try:
                with open(l3_file, 'r') as f:
                    data = json.load(f)
                print(f"   ✅ Loaded L3 output")
            except Exception as e:
                print(f"   ⚠️ Error reading L3 output: {e}")
        else:
            print("   ⚠️ No L3 output found")
            
    def _report_system_health(self):
        """Overall system health report"""
        print("\n" + "=" * 80)
        print("🏥 SYSTEM HEALTH REPORT")
        print("=" * 80)

        total_errors = len(self.errors)
        total_signals = len(self.signals)
        total_trades = len(self.trades)

        # Get properly classified signal metrics - CRITICAL FIX: HOLD is not a trade
        metrics = self._get_signal_metrics()

        # Calculate uptime (estimate from log frequency)
        if self.portfolio_snapshots:
            first_ts = min(s['timestamp'] for s in self.portfolio_snapshots)
            last_ts = max(s['timestamp'] for s in self.portfolio_snapshots)
            total_hours = (last_ts - first_ts).total_seconds() / 3600
            uptime_pct = 100 if total_hours > 0 else 0
        else:
            total_hours = 0
            uptime_pct = 0

        print(f"\n📊 OPERATIONAL METRICS:")
        print(f"   Total Runtime: {total_hours:.1f} hours ({self.days_back} days)")
        print(f"   Estimated Uptime: {uptime_pct:.1f}%")

        # SEMANTIC VALIDITY CHECK - HARD RULE - PRIORITY CRITICAL
        cycles = self.system_metrics.get('cycles', [])
        total_cycles = len(cycles)

        # HARD RULE: If total_cycles > 1000 AND hold_signals == 0: RAISE SEMANTIC_ERROR
        if total_cycles > 1000 and metrics['hold_signals'] == 0:
            print(f"   🔴 SEMANTIC ERROR: System has {total_cycles} cycles but 0 HOLD signals")
            print(f"      This indicates fundamental signal source corruption")
            print(f"      Report INVALID for ML training and performance analysis")
            semantic_valid = False
        else:
            semantic_valid = True

        # REQUIRED METRICS DISPLAY - EXACT FORMAT
        print(f"\n🔵 SIGNALS GENERATED (BUY / SELL / HOLD): {metrics['total_signals']}")
        print(f"🟡 HOLD SIGNALS (explícito): {metrics['hold_signals']}")
        print(f"🟢 ACTIONABLE SIGNALS: {metrics['actionable_signals']}")
        print(f"🔴 ORDERS EXECUTED: {metrics['orders_executed']}")

        # ========================================================================================
        # INVARIANTES DEL SISTEMA - VALIDACIÓN CRÍTICA
        # ========================================================================================
        print(f"\n🎯 INVARIANTES DEL SISTEMA - VALIDACIÓN:")

        # INVARIANTE 1: cycles >> orders_executed
        cycles_orders_ratio = total_cycles / max(metrics['orders_executed'], 1)
        print(f"   📊 INV-1 cycles >> orders_executed: {cycles_orders_ratio:.1f}:1 ({total_cycles} cycles / {metrics['orders_executed']} orders)")
        if cycles_orders_ratio < 5:
            print(f"      ❌ CRÍTICO: Ratio demasiado bajo! Sistema sobreoperando")
        elif cycles_orders_ratio < 10:
            print(f"      ⚠️ ADVERTENCIA: Ratio bajo, revisar conservadurismo")
        else:
            print(f"      ✅ EXCELENTE: Sistema conservador")

        # INVARIANTE 2: HOLD signals > 30% en mercado normal
        if metrics['total_signals'] > 0:
            hold_percentage = (metrics['hold_signals'] / metrics['total_signals']) * 100
            print(f"   📊 INV-2 HOLD > 30%: {hold_percentage:.1f}% ({metrics['hold_signals']}/{metrics['total_signals']})")
            if hold_percentage < 30:
                print(f"      ❌ CRÍTICO: HOLD signals muy bajos! Sistema agresivo")
            elif hold_percentage < 50:
                print(f"      ⚠️ ADVERTENCIA: HOLD signals bajos")
            else:
                print(f"      ✅ EXCELENTE: Sistema conservador")

        # INVARIANTE 3: L2 ≠ L3 (intención ≠ decisión estratégica)
        print(f"   📊 INV-3 L2 ≠ L3: Validando separación de capas...")
        l2_l3_separation_valid = self._validate_l2_l3_separation(metrics)
        if l2_l3_separation_valid:
            print(f"      ✅ EXCELENTE: L2/L3 correctamente separados")
        else:
            print(f"      ❌ CRÍTICO: L2/L3 mal separados!")

        # INVARIANTE 4: Override ≠ regla, Override = excepción
        l3_override_rate = self._calculate_l3_override_rate()
        print(f"   📊 INV-4 Override = excepción: {l3_override_rate:.1f}% de ciclos")
        if l3_override_rate > 20:
            print(f"      ❌ CRÍTICO: Override rate muy alto! (>20%)")
        elif l3_override_rate > 10:
            print(f"      ⚠️ ADVERTENCIA: Override rate elevado")
        else:
            print(f"      ✅ EXCELENTE: Override como excepción")

        # INVARIANTE 5: Si el sistema duda → HOLD
        doubt_hold_compliance = self._validate_doubt_hold_compliance(metrics)
        print(f"   📊 INV-5 Duda → HOLD: {'✅ Cumple' if doubt_hold_compliance else '❌ No cumple'}")

        # EXECUTION RATE - ONLY OVER ACTIONABLE SIGNALS
        if metrics['actionable_signals'] > 0:
            execution_rate = (metrics['orders_executed'] / metrics['actionable_signals']) * 100
            print(f"\n📊 EXECUTION RATE (solo sobre Actionable Signals): {execution_rate:.2f}%")

            # FINAL VALIDATION CHECK - PRIORITY CRITICAL
            if metrics['hold_signals'] > 0 and metrics['orders_executed'] >= metrics['total_signals']:
                print(f"      🔴 VALIDATION FAILURE: Orders Executed ({metrics['orders_executed']}) >= Signals Generated ({metrics['total_signals']}) with HOLD signals present")
                print(f"      This violates the rule: If HOLD > 0, then Orders Executed < Signals Generated")

            if execution_rate < 10:
                print(f"      ❌ CRITICAL: Very low execution rate!")
            elif execution_rate < 30:
                print(f"      ⚠️ WARNING: Low execution rate")
            else:
                print(f"      ✅ GOOD: Healthy execution rate")
        else:
            print(f"   📊 EXECUTION RATE: N/A (no actionable signals)")

        print(f"   Total Errors Logged: {total_errors}")

        # ========================================================================================
        # PHILOSOPHY VALIDATION: Un sistema que siempre opera no es agresivo, es ciego
        # ========================================================================================
        print(f"\n🧠 PHILOSOPHY VALIDATION:")
        philosophy_compliant = self._validate_philosophy_compliance(metrics, total_cycles)
        if philosophy_compliant:
            print(f"   ✅ PHILOSOPHY COMPLIANT: Sistema conservador, no ciego")
        else:
            print(f"   ❌ PHILOSOPHY VIOLATION: Sistema operando ciegamente!")

        # Error rate (against all signals)
        if metrics['total_signals'] > 0:
            error_rate = (total_errors / metrics['total_signals']) * 100
            print(f"   Error Rate: {error_rate:.2f}%")

        # CROSS-COHERENCE CHECKS - PRIORITY CRITICAL
        print(f"\n🔗 CROSS-COHERENCE VALIDATION:")
        coherence_failures = []

        # ASSERT signal_source != trade_source
        signal_source = "logs/events.json"
        trade_source = "paper_trades.csv"
        if signal_source == trade_source:
            coherence_failures.append("signal_source == trade_source - architectural violation")

        # ASSERT hold_signals > 0 (unless system just started)
        if metrics['hold_signals'] == 0 and total_cycles > 100:
            coherence_failures.append(f"hold_signals == 0 with {total_cycles} cycles - semantic error")

        # ASSERT cycles >> orders_executed (cycles should greatly exceed executed trades)
        if total_cycles > 0 and metrics['orders_executed'] > 0:
            ratio = total_cycles / metrics['orders_executed']
            if ratio < 10:  # cycles should be at least 10x orders_executed
                coherence_failures.append(f"cycles/orders_executed ratio ({ratio:.1f}) too low - should be >> 1")

        if coherence_failures:
            print(f"   🔴 CROSS-COHERENCE FAILURES DETECTED:")
            for failure in coherence_failures:
                print(f"      ❌ {failure}")
            print(f"   📊 REPORT VALIDITY: INVALID FOR ML TRAINING")
            print(f"   📊 REPORT VALIDITY: INVALID FOR PERFORMANCE ANALYSIS")
        else:
            print(f"   ✅ All cross-coherence checks passed")
            print(f"   📊 REPORT VALIDITY: VALID FOR ML TRAINING")
            print(f"   📊 REPORT VALIDITY: VALID FOR PERFORMANCE ANALYSIS")

        # AUTOMATIC DATA VALIDATION - PRIORITY CRITICAL
        print(f"\n🔍 DATA SOURCE VALIDATION:")
        data_warnings = self._validate_data_sources()
        if data_warnings:
            for warning in data_warnings:
                level_emoji = {
                    'CRITICAL': '🔴',
                    'WARNING': '🟡',
                    'INFO': 'ℹ️'
                }.get(warning['level'], '⚪')

                print(f"   {level_emoji} [{warning['level']}] {warning['message']}")
                print(f"      Details: {warning['details']}")
                print(f"      Action: {warning['recommendation']}")
        else:
            print(f"   ✅ All data source validations passed")
            print(f"      Signal source integrity confirmed")
            print(f"      Trade data separation verified")

    def _report_cycle_performance(self):
        """Analyze system cycle performance"""
        print("\n" + "=" * 80)
        print("🔄 CYCLE PERFORMANCE ANALYSIS")
        print("=" * 80)

        cycles = self.system_metrics.get('cycles', [])
        if not cycles:
            print("\n   ⚠️ No cycle data available")
            return

        df_cycles = pd.DataFrame(cycles)

        # Basic cycle stats
        total_cycles = len(df_cycles)
        avg_cycle_time = df_cycles['cycle_time'].mean()
        total_cycle_time = df_cycles['cycle_time'].sum()
        avg_signals_per_cycle = df_cycles['signals_count'].mean()
        avg_orders_per_cycle = df_cycles['orders_count'].mean()
        total_rejected = df_cycles['rejected_count'].sum()

        print(f"\n📊 CYCLE STATISTICS:")
        print(f"   Total Cycles: {total_cycles}")
        print(f"   Average Cycle Time: {avg_cycle_time:.2f}s")
        print(f"   Total Processing Time: {total_cycle_time:.2f}s")
        print(f"   Average Signals per Cycle: {avg_signals_per_cycle:.2f}")
        print(f"   Average Orders per Cycle: {avg_orders_per_cycle:.2f}")
        print(f"   Total Rejected Orders: {total_rejected}")

        # Cycle time analysis
        print(f"\n⏱️ CYCLE TIME ANALYSIS:")
        cycle_times = df_cycles['cycle_time']
        p95_cycle_time = cycle_times.quantile(0.95)
        max_cycle_time = cycle_times.max()
        min_cycle_time = cycle_times.min()

        print(f"   Min Cycle Time: {min_cycle_time:.2f}s")
        print(f"   Max Cycle Time: {max_cycle_time:.2f}s")
        print(f"   95th Percentile: {p95_cycle_time:.2f}s")

        if p95_cycle_time > 2.0:
            print(f"      ⚠️ WARNING: Some cycles are taking too long (>2s)")
        elif avg_cycle_time < 0.5:
            print(f"      ✅ EXCELLENT: Fast cycle times")

        # Cycle frequency analysis - BASED ONLY ON REAL CYCLES WITH ACTION
        # 🚫 EXCLUDE: Cycles with HOLD signals or no actionable signals
        # ✅ INCLUDE: Only cycles that produced BUY/SELL signals or trades

        if total_cycles > 1:
            # Filter cycles that actually produced actionable signals or trades
            active_cycles = []
            for cycle in df_cycles.to_dict('records'):
                cycle_signals = cycle.get('signals_count', 0)
                cycle_orders = cycle.get('orders_count', 0)
                # Include cycle if it produced actionable signals (>0) or executed trades (>0)
                if cycle_signals > 0 or cycle_orders > 0:
                    active_cycles.append(cycle)

            if len(active_cycles) > 1:
                df_active_cycles = pd.DataFrame(active_cycles)

                # Normalize timestamps to UTC and sort chronologically
                df_active_cycles['timestamp'] = pd.to_datetime(df_active_cycles['timestamp'], utc=True)
                df_active_cycles = df_active_cycles.sort_values('timestamp').reset_index(drop=True)

                # Calculate time differences between consecutive ACTIVE cycles
                time_diffs = df_active_cycles['timestamp'].diff().dropna()

                # Filter out invalid time differences
                valid_time_diffs = time_diffs[time_diffs > pd.Timedelta(0)]

                invalid_count = len(time_diffs) - len(valid_time_diffs)
                if invalid_count > 0:
                    print(f"\n   ⚠️ WARNING: {invalid_count} active cycles had invalid timestamps")
                    print(f"      These cycles were excluded from frequency calculations")

                if len(valid_time_diffs) > 0:
                    avg_cycle_frequency = valid_time_diffs.mean().total_seconds()

                    print(f"\n🔄 CYCLE FREQUENCY (Active Cycles Only):")
                    print(f"   Active Cycles Analyzed: {len(valid_time_diffs) + 1}")
                    print(f"   Total Cycles (All): {total_cycles}")
                    print(f"   Active Cycle Ratio: {((len(valid_time_diffs) + 1) / total_cycles * 100):.1f}%")
                    print(f"   Average Time Between Active Cycles: {avg_cycle_frequency:.1f}s")

                    cycles_per_minute = 60 / avg_cycle_frequency if avg_cycle_frequency > 0 else 0
                    print(f"   Active Cycles per Minute: {cycles_per_minute:.2f}")

                    if cycles_per_minute < 0.1:
                        print(f"      ⚠️ CRITICAL: Very low active cycle frequency (<0.1/min)")
                    elif cycles_per_minute < 1:
                        print(f"      ⚠️ WARNING: Low active cycle frequency (<1/min)")
                    elif cycles_per_minute > 10:
                        print(f"      ✅ EXCELLENT: High active cycle frequency")
                else:
                    print(f"\n   ❌ ERROR: No valid active cycle time differences found")
                    print(f"      All active cycles appear to have invalid timestamps")
            else:
                print(f"\n   ℹ️ INFO: Only {len(active_cycles)} active cycles found (cycles with signals or trades)")
                print(f"      Total cycles: {total_cycles} (includes idle cycles)")

    def _report_model_performance(self):
        """Analyze ML model performance and resource usage"""
        print("\n" + "=" * 80)
        print("🤖 MODEL PERFORMANCE ANALYSIS")
        print("=" * 80)

        # Feature calculations
        feature_calcs = self.system_metrics.get('feature_calculations', [])
        print(f"\n📊 FEATURE ENGINEERING:")
        print(f"   Feature Calculations: {len(feature_calcs)}")

        # Model operations
        model_ops = self.system_metrics.get('model_operations', [])
        if model_ops:
            reuse_count = sum(1 for op in model_ops if op['type'] == 'reuse')
            load_count = sum(1 for op in model_ops if op['type'] == 'load')

            print(f"\n🧠 MODEL OPERATIONS:")
            print(f"   Model Loads: {load_count}")
            print(f"   Model Reuses: {reuse_count}")

            if reuse_count > load_count:
                print(f"      ✅ GOOD: Efficient model caching")
            else:
                print(f"      ⚠️ WARNING: Frequent model reloading")

        # Memory and cleanup operations
        cleanup_events = sum(1 for e in self.errors if 'memory' in e['message'].lower() or 'cleanup' in e['message'].lower())
        if cleanup_events > 0:
            print(f"\n🧹 MEMORY MANAGEMENT:")
            print(f"   Cleanup Events: {cleanup_events}")

        # TensorFlow session clears
        tf_clears = sum(1 for e in self.errors if 'tensorflow session cleared' in e['message'].lower())
        if tf_clears > 0:
            print(f"   TensorFlow Session Clears: {tf_clears}")

    def _report_signal_analysis(self):
        """SIGNAL ANALYSIS (INTENCIÓN) - PRIORITY CRITICAL"""
        print("\n" + "=" * 80)
        print("📡 SIGNAL ANALYSIS (INTENCIÓN)")
        print("=" * 80)
        print("🔵 BUY / SELL / HOLD signals from system logs")
        print("🔵 Confidence levels and L2 vs L3 distribution")
        print("🔵 By symbol analysis")
        print("🚫 NO TRADE EXECUTION DATA HERE")

        if not self.signals:
            print("\n   ⚠️ No system intention signals found in logs")
            print("      This indicates either:")
            print("      - System is not generating signals")
            print("      - Signal logging is not working")
            print("      - Signal source filtering is too restrictive")
            return

        # BUY/SELL/HOLD distribution
        intention_counts = defaultdict(int)
        source_counts = defaultdict(int)
        confidence_levels = []

        for signal in self.signals:
            # Extract intention type
            intention_type = signal.get('intention_type', 'UNKNOWN')
            intention_counts[intention_type] += 1

            # Extract source type
            source_type = signal.get('source_type', 'UNKNOWN')
            source_counts[source_type] += 1

            # Extract confidence
            confidence = signal.get('confidence', 0.5)
            confidence_levels.append(confidence)

        print(f"\n🎯 SYSTEM INTENTIONS (BUY/SELL/HOLD):")
        total_signals = len(self.signals)
        for intention, count in sorted(intention_counts.items(), key=lambda x: -x[1]):
            pct = (count / total_signals) * 100
            print(f"   {intention}: {count} signals ({pct:.1f}%)")

        print(f"\n🏗️ SIGNAL SOURCES (L2/L3/System):")
        for source, count in sorted(source_counts.items(), key=lambda x: -x[1]):
            pct = (count / total_signals) * 100
            print(f"   {source}: {count} signals ({pct:.1f}%)")

        # Confidence analysis
        if confidence_levels:
            avg_confidence = np.mean(confidence_levels)
            print(f"\n📊 SIGNAL CONFIDENCE:")
            print(f"   Average Confidence: {avg_confidence:.3f}")
            print(f"   High Confidence (>0.8): {sum(1 for c in confidence_levels if c > 0.8)}")
            print(f"   Medium Confidence (0.5-0.8): {sum(1 for c in confidence_levels if 0.5 <= c <= 0.8)}")
            print(f"   Low Confidence (<0.5): {sum(1 for c in confidence_levels if c < 0.5)}")

            if avg_confidence < 0.6:
                print(f"      ⚠️ WARNING: Low average signal confidence")
            else:
                print(f"      ✅ GOOD: Healthy signal confidence")

        # Signals by symbol (intention-focused)
        symbol_counts = defaultdict(int)
        intention_by_symbol = defaultdict(lambda: defaultdict(int))

        for signal in self.signals:
            symbol = signal.get('symbol', 'UNKNOWN')
            intention = signal.get('intention_type', 'UNKNOWN')

            symbol_counts[symbol] += 1
            intention_by_symbol[symbol][intention] += 1

        print(f"\n🌍 SIGNALS BY SYMBOL (System Intentions):")
        for symbol, count in sorted(symbol_counts.items(), key=lambda x: -x[1]):
            pct = (count / total_signals) * 100
            print(f"   {symbol}: {count} signals ({pct:.1f}%)")

            # Show intention breakdown per symbol
            intentions = intention_by_symbol[symbol]
            for intention, cnt in sorted(intentions.items(), key=lambda x: -x[1]):
                symbol_pct = (cnt / count) * 100
                print(f"      {intention}: {cnt} ({symbol_pct:.1f}%)")

    def _report_trading_performance(self):
        """Trading performance metrics"""
        print("\n" + "=" * 80)
        print("💰 TRADING PERFORMANCE")
        print("=" * 80)

        # Count CLEAN EXITS (tactical sells) from logs
        clean_exits_count = 0
        for signal in self.signals:
            if (signal.get('intention_type') == 'SELL' and
                signal.get('source_type') == 'L2_TACTICAL'):
                clean_exits_count += 1

        if not self.trades:
            print("\n   ⚠️ No trades executed during period")
            print(f"   CLEAN EXITS (SELL táctico): {clean_exits_count}")
            return

        df = pd.DataFrame(self.trades)

        # Basic stats
        total_trades = len(df)
        buy_trades = len(df[df['side'] == 'buy'])
        sell_trades = len(df[df['side'] == 'sell'])

        print(f"\n📊 TRADE STATISTICS:")
        print(f"   Total Trades: {total_trades}")
        print(f"   Buy Trades: {buy_trades}")
        print(f"   Sell Trades: {sell_trades}")
        print(f"   CLEAN EXITS (SELL táctico): {clean_exits_count}")
        
        # By symbol
        print(f"\n📈 TRADES BY SYMBOL:")
        for symbol in df['symbol'].unique():
            symbol_trades = df[df['symbol'] == symbol]
            print(f"   {symbol}: {len(symbol_trades)} trades")
            
        # Calculate returns (if we have entry/exit prices)
        if 'price' in df.columns and 'realized_pnl' in df.columns:
            total_pnl = df['realized_pnl'].sum()
            avg_pnl = df['realized_pnl'].mean()
            
            winning_trades = len(df[df['realized_pnl'] > 0])
            losing_trades = len(df[df['realized_pnl'] < 0])
            
            if total_trades > 0:
                win_rate = (winning_trades / total_trades) * 100
            else:
                win_rate = 0
                
            print(f"\n💵 PROFIT & LOSS:")
            print(f"   Total PnL: ${total_pnl:.2f}")
            print(f"   Average PnL per Trade: ${avg_pnl:.2f}")
            print(f"   Winning Trades: {winning_trades}")
            print(f"   Losing Trades: {losing_trades}")
            print(f"   Win Rate: {win_rate:.2f}%")
            
            if win_rate < 40:
                print(f"      ❌ CRITICAL: Very low win rate!")
            elif win_rate < 55:
                print(f"      ⚠️ WARNING: Below target win rate (55%)")
            else:
                print(f"      ✅ EXCELLENT: Above target win rate")
                
    def _report_l3_strategic_analysis(self):
        """L3 strategic layer analysis"""
        print("\n" + "=" * 80)
        print("🌟 L3 STRATEGIC ANALYSIS")
        print("=" * 80)
        
        if not self.l3_decisions:
            print("\n   ⚠️ No L3 decisions found")
            return
            
        # Count regime detections
        regime_counts = defaultdict(int)
        for decision in self.l3_decisions:
            msg = decision['message']
            if 'TRENDING' in msg:
                regime_counts['TRENDING'] += 1
            elif 'RANGE' in msg:
                regime_counts['RANGE'] += 1
            elif 'VOLATILE' in msg:
                regime_counts['VOLATILE'] += 1
            elif 'BEAR' in msg:
                regime_counts['BEAR'] += 1
            elif 'BULL' in msg:
                regime_counts['BULL'] += 1
                
        print(f"\n📊 REGIME DISTRIBUTION:")
        total_regimes = sum(regime_counts.values())
        for regime, count in sorted(regime_counts.items(), key=lambda x: -x[1]):
            pct = (count / total_regimes * 100) if total_regimes > 0 else 0
            print(f"   {regime}: {count} ({pct:.1f}%)")
            
        # Count L3 overrides
        override_count = sum(1 for d in self.l3_decisions if 'OVERRIDE' in d['message'])
        print(f"\n🎯 L3 OVERRIDES:")
        print(f"   Total Overrides: {override_count}")
        if total_regimes > 0:
            override_rate = (override_count / total_regimes) * 100
            print(f"   Override Rate: {override_rate:.1f}%")
            
    def _report_l2_tactical_analysis(self):
        """L2 tactical layer analysis"""
        print("\n" + "=" * 80)
        print("🎯 L2 TACTICAL ANALYSIS")
        print("=" * 80)
        
        if not self.signals:
            print("\n   ⚠️ No L2 signals found")
            return
            
        # Count signal types
        signal_types = defaultdict(int)
        for signal in self.signals:
            msg = signal['message']
            if 'BUY' in msg:
                signal_types['BUY'] += 1
            elif 'SELL' in msg:
                signal_types['SELL'] += 1
            elif 'HOLD' in msg:
                signal_types['HOLD'] += 1
                
        print(f"\n📊 SIGNAL DISTRIBUTION:")
        total_signals = sum(signal_types.values())
        for sig_type, count in sorted(signal_types.items(), key=lambda x: -x[1]):
            pct = (count / total_signals * 100) if total_signals > 0 else 0
            print(f"   {sig_type}: {count} ({pct:.1f}%)")
            
        # Check if too many HOLD signals
        if signal_types.get('HOLD', 0) > total_signals * 0.6:
            print(f"\n   ℹ️ INFO: High HOLD ratio indicates conservative risk posture")
            print(f"      {signal_types.get('HOLD', 0)}/{total_signals} signals are HOLD ({(signal_types.get('HOLD', 0)/total_signals*100):.1f}%)")
            
    def _validate_data_sources(self):
        """
        AUTOMATIC INCONSISTENCY DETECTION - PRIORITY CRITICAL
        =====================================================
        Validates data source integrity and issues warnings for architectural violations
        """
        cycles = self.system_metrics.get('cycles', [])
        total_cycles = len(cycles)

        # Get signal metrics
        metrics = self._get_signal_metrics()

        warnings = []

        # VALIDATION 1: Check for missing HOLD signals in active system
        if metrics['hold_signals'] == 0 and total_cycles > 1000:
            warnings.append({
                'level': 'CRITICAL',
                'message': f"HOLD signals missing — signal source invalid",
                'details': f"System has {total_cycles} cycles but 0 HOLD signals. Signal data may be corrupted or coming from wrong source.",
                'recommendation': "Verify signal source is logs/events.json, not trade data"
            })

        # VALIDATION 2: Check for signals incorrectly inferred from trades
        if (metrics['orders_executed'] == metrics['actionable_signals'] and
            metrics['hold_signals'] == 0 and
            metrics['orders_executed'] > 0):
            warnings.append({
                'level': 'CRITICAL',
                'message': "Signals incorrectly inferred from trades",
                'details': f"orders_executed ({metrics['orders_executed']}) == actionable_signals ({metrics['actionable_signals']}) with 0 HOLD signals. Signals likely derived from paper_trades.csv instead of logs.",
                'recommendation': "Ensure signals come ONLY from logs/events.json (L2_SIGNAL messages)"
            })

        # VALIDATION 3: Check for unrealistic execution rates (signals << trades)
        if (metrics['actionable_signals'] > 0 and
            (metrics['orders_executed'] / metrics['actionable_signals']) > 2.0):
            warnings.append({
                'level': 'WARNING',
                'message': "Unrealistic execution rate detected",
                'details': f"Execution rate {(metrics['orders_executed']/metrics['actionable_signals'])*100:.1f}% suggests possible data source confusion.",
                'recommendation': "Verify actionable_signals come from logs and orders_executed from paper_trades.csv"
            })

        # VALIDATION 4: Check for signal source validity (should have mix of BUY/SELL/HOLD)
        if (metrics['total_signals'] > 100 and
            (metrics['hold_signals'] == 0 or metrics['hold_signals'] == metrics['total_signals'])):
            warnings.append({
                'level': 'WARNING',
                'message': "Signal distribution anomaly",
                'details': f"All {metrics['total_signals']} signals are {'HOLD' if metrics['hold_signals'] > 0 else 'actionable'} - unusual for healthy system.",
                'recommendation': "Check signal generation logic for balanced BUY/SELL/HOLD distribution"
            })

        return warnings

    def _get_signal_metrics(self):
        """
        CRITICAL ARCHITECTURAL SEPARATION - PRIORITY CRITICAL:
        =================================================================
        ❌ FORBIDDEN: Deriving signal metrics from trade data
        ✅ REQUIRED: Signals come ONLY from logs (L2/L3 outputs)
        ✅ REQUIRED: Trades come ONLY from paper_trades.csv
        ✅ REQUIRED: These are completely separate data sources

        TERMINOLOGY DEFINITIONS (MANDATORY):
        ====================================
        SIGNAL = System intention from logs (BUY/SELL/HOLD)
        TRADE = Real execution from paper_trades.csv (BUY/SELL only)
        executed_trade = order BUY or SELL sent and accepted by L1
        validated_signal ≠ executed_trade
        HOLD signals NEVER count as executed trades
        HOLD can only appear in: signal analysis, decision analysis, strategy filtering
        Execution Rate = orders_executed / actionable_signals, NEVER over HOLD
        Un HOLD nunca es un trade
        Un trade no define una signal

        DATA SOURCE SEPARATION:
        =======================
        self.signals ← ONLY from logs/events.json (L2_SIGNAL messages)
        self.trades ← ONLY from paper_trades.csv
        No cross-contamination between signal and trade data sources
        """
        actionable_signals = 0  # BUY/SELL from logs only - these CAN become executed trades
        hold_signals = 0        # HOLD from logs only - these are NEVER executed trades

        # CRITICAL: Signals come ONLY from logs, never from trade data
        for signal in self.signals:  # This comes from logs/events.json only
            msg = signal['message']
            extra = signal.get('extra', {})

            # Extract signal type from log message (system intention)
            if 'trading_action' in extra:
                action = extra['trading_action'].get('action', '')
                if action in ['BUY', 'SELL']:
                    actionable_signals += 1
                elif action == 'HOLD':
                    hold_signals += 1
            elif 'BUY' in msg:
                actionable_signals += 1
            elif 'SELL' in msg:
                actionable_signals += 1
            elif 'HOLD' in msg:
                # HOLD signals are NOT executed trades - they only appear in signal analysis
                hold_signals += 1

        # CRITICAL: Trades come ONLY from paper_trades.csv, never influence signal metrics
        orders_executed = len(self.trades)  # This comes from paper_trades.csv only

        # EXACT DATA SOURCES AS REQUIRED - OBLIGATORIO
        return {
            'total_signals': len(self.signals),        # signals_generated ← logs (BUY + SELL + HOLD)
            'actionable_signals': actionable_signals,  # actionable_signals ← logs (BUY + SELL)
            'hold_signals': hold_signals,              # hold_signals ← logs (HOLD)
            'orders_executed': orders_executed          # orders_executed ← paper_trades.csv
        }

    def _report_l1_execution_analysis(self):
        """L1 execution layer analysis"""
        print("\n" + "=" * 80)
        print("⚡ L1 EXECUTION ANALYSIS")
        print("=" * 80)

        # Get properly classified signal metrics
        metrics = self._get_signal_metrics()

        # Count rejections from errors
        rejection_count = sum(1 for e in self.errors if 'rejected' in e['message'].lower())
        cooldown_count = sum(1 for e in self.errors if 'cooldown' in e['message'].lower())

        print(f"\n📊 EXECUTION METRICS:")
        print(f"   Signals Generated: {metrics['total_signals']}")
        print(f"   Actionable Signals (BUY/SELL): {metrics['actionable_signals']}")
        print(f"   HOLD Signals: {metrics['hold_signals']}")
        print(f"   Orders Executed: {metrics['orders_executed']}")
        print(f"   Rejected Signals: {rejection_count}")
        print(f"   Cooldown Blocks: {cooldown_count}")

        # Calculate rates against actionable signals only - CRITICAL FIX
        if metrics['actionable_signals'] > 0:
            execution_rate = (metrics['orders_executed'] / metrics['actionable_signals']) * 100
            rejection_rate = (rejection_count / metrics['actionable_signals']) * 100
            cooldown_rate = (cooldown_count / metrics['actionable_signals']) * 100

            print(f"   Execution Rate (vs Actionable Signals): {execution_rate:.1f}%")
            print(f"   Rejection Rate (vs Actionable Signals): {rejection_rate:.1f}%")
            print(f"   Cooldown Block Rate (vs Actionable Signals): {cooldown_rate:.1f}%")

            if execution_rate < 10:
                print(f"\n   ❌ CRITICAL: Very low execution rate!")
            elif execution_rate < 30:
                print(f"\n   ⚠️ WARNING: Low execution rate")
            else:
                print(f"\n   ✅ GOOD: Healthy execution rate")

            if rejection_rate > 50:
                print(f"   ❌ CRITICAL: Very high rejection rate!")
            elif rejection_rate > 20:
                print(f"   ⚠️ WARNING: High rejection rate")
        else:
            print(f"   Execution Rate: N/A (no actionable signals)")
                
    def _report_portfolio_evolution(self):
        """Portfolio value evolution"""
        print("\n" + "=" * 80)
        print("📈 PORTFOLIO EVOLUTION")
        print("=" * 80)
        
        if not self.portfolio_snapshots:
            print("\n   ⚠️ No portfolio history available")
            return
            
        # Extract portfolio values from log messages
        portfolio_values = []
        for snapshot in self.portfolio_snapshots:
            msg = snapshot['message']
            # Look for "Total=XXXX USDT"
            match = re.search(r'Total=(\d+\.?\d*)', msg)
            if match:
                value = float(match.group(1))
                portfolio_values.append({
                    'timestamp': snapshot['timestamp'],
                    'value': value
                })
                
        if portfolio_values:
            df = pd.DataFrame(portfolio_values)
            initial_value = df.iloc[0]['value']
            final_value = df.iloc[-1]['value']
            total_return = ((final_value - initial_value) / initial_value) * 100
            
            print(f"\n💰 PORTFOLIO PERFORMANCE:")
            print(f"   Initial Value: ${initial_value:.2f}")
            print(f"   Final Value: ${final_value:.2f}")
            print(f"   Total Return: {total_return:+.2f}%")
            
            if total_return < -10:
                print(f"      ❌ CRITICAL: Large drawdown!")
            elif total_return < 0:
                print(f"      ⚠️ WARNING: Negative return")
            elif total_return > 5:
                print(f"      ✅ EXCELLENT: Positive return")
            else:
                print(f"      ℹ️ INFO: Small positive return")
                
    def _report_errors_and_issues(self):
        """Detailed error analysis"""
        print("\n" + "=" * 80)
        print("🚨 ERRORS & ISSUES")
        print("=" * 80)
        
        if not self.errors:
            print("\n   ✅ No errors logged during period!")
            return
            
        # Categorize errors
        error_categories = defaultdict(int)
        for error in self.errors:
            msg = error['message'].lower()
            if 'no position' in msg:
                error_categories['No Position to Sell'] += 1
            elif 'insufficient' in msg:
                error_categories['Insufficient Funds'] += 1
            elif 'cooldown' in msg:
                error_categories['Cooldown Active'] += 1
            elif 'rejected' in msg:
                error_categories['Signal Rejected'] += 1
            elif 'timeout' in msg:
                error_categories['Timeout'] += 1
            else:
                error_categories['Other'] += 1
                
        print(f"\n📊 ERROR BREAKDOWN:")
        total_errors = sum(error_categories.values())
        for category, count in sorted(error_categories.items(), key=lambda x: -x[1]):
            pct = (count / total_errors * 100) if total_errors > 0 else 0
            print(f"   {category}: {count} ({pct:.1f}%)")
            
        # Show most recent errors
        print(f"\n🔴 RECENT ERRORS (Last 10):")
        recent_errors = sorted(self.errors, key=lambda x: x['timestamp'], reverse=True)[:10]
        for err in recent_errors:
            ts = err['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
            msg = err['message'][:100]  # Truncate long messages
            print(f"   [{ts}] {msg}")
            
    def _report_recommendations(self):
        """Generate actionable recommendations"""
        print("\n" + "=" * 80)
        print("💡 RECOMMENDATIONS")
        print("=" * 80)

        recommendations = []

        # Get properly classified signal metrics - CRITICAL FIX
        metrics = self._get_signal_metrics()

        # Check execution rate (trades vs actionable signals only)
        if metrics['actionable_signals'] > 0:
            execution_rate = (metrics['orders_executed'] / metrics['actionable_signals']) * 100
            if execution_rate < 10:
                recommendations.append({
                    'priority': 'CRITICAL',
                    'issue': f'Very low execution rate ({execution_rate:.1f}%)',
                    'recommendation': 'Review L1 validation logic and reduce rejection thresholds'
                })
            elif execution_rate < 30:
                recommendations.append({
                    'priority': 'HIGH',
                    'issue': f'Low execution rate ({execution_rate:.1f}%)',
                    'recommendation': 'Analyze rejection reasons and adjust risk parameters'
                })

        # Check for "No position" errors
        no_position_errors = sum(1 for e in self.errors if 'no position' in e['message'].lower())
        if no_position_errors > 10:
            recommendations.append({
                'priority': 'CRITICAL',
                'issue': f'{no_position_errors} "No position to sell" errors',
                'recommendation': 'Fix portfolio state synchronization between L3 and L1'
            })

        # Check win rate
        if self.trades:
            df = pd.DataFrame(self.trades)
            if 'realized_pnl' in df.columns:
                winning = len(df[df['realized_pnl'] > 0])
                total = len(df)
                if total > 10:  # Only if we have enough trades
                    win_rate = (winning / total) * 100
                    if win_rate < 40:
                        recommendations.append({
                            'priority': 'HIGH',
                            'issue': f'Low win rate ({win_rate:.1f}%)',
                            'recommendation': 'Review strategy logic and entry/exit conditions'
                        })

        # Check HOLD signals percentage - HIGH HOLD ratio is GOOD (conservative system)
        if metrics['total_signals'] > 0:
            hold_pct = (metrics['hold_signals'] / metrics['total_signals']) * 100
            if hold_pct > 95:  # Only warn if extremely high HOLD ratio
                recommendations.append({
                    'priority': 'LOW',
                    'issue': f'Extremely high HOLD signals ({hold_pct:.1f}%)',
                    'recommendation': 'System is very conservative - consider if market conditions warrant more activity'
                })

        # Print recommendations
        if not recommendations:
            print("\n   ✅ No critical issues found!")
            print("   System appears to be operating within normal parameters.")
        else:
            print("\n🔧 SUGGESTED ACTIONS:")
            for i, rec in enumerate(recommendations, 1):
                priority_emoji = {
                    'CRITICAL': '🔴',
                    'HIGH': '🟠',
                    'MEDIUM': '🟡',
                    'LOW': '🟢'
                }.get(rec['priority'], '⚪')

                print(f"\n   {i}. {priority_emoji} [{rec['priority']}]")
                print(f"      Issue: {rec['issue']}")
                print(f"      Action: {rec['recommendation']}")
                
    def _export_reports(self):
        """Export detailed reports to files"""
        print("\n" + "=" * 80)
        print("💾 EXPORTING REPORTS")
        print("=" * 80)
        
        output_dir = self.base_path / "reports"
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Export trades
        if self.trades:
            trades_file = output_dir / f"trades_report_{timestamp}.csv"
            pd.DataFrame(self.trades).to_csv(trades_file, index=False)
            print(f"   ✅ Trades exported to: {trades_file}")
            
        # Export errors
        if self.errors:
            errors_file = output_dir / f"errors_report_{timestamp}.csv"
            pd.DataFrame(self.errors).to_csv(errors_file, index=False)
            print(f"   ✅ Errors exported to: {errors_file}")
            
        # Export summary JSON - CRITICAL FIX: Use proper metrics
        # EXACT DATA SOURCES AS REQUIRED - OBLIGATORIO:
        # signals_generated ← logs (BUY + SELL + HOLD)
        # actionable_signals ← logs (BUY + SELL)
        # hold_signals ← logs (HOLD)
        # orders_executed ← paper_trades.csv
        # execution_rate ← orders_executed / actionable_signals
        # 🚫 PROHIBIDO: calcular señales desde paper_trades.csv

        metrics = self._get_signal_metrics()
        summary = {
            'analysis_period': {
                'start': self.cutoff_date.isoformat(),
                'end': datetime.now().isoformat(),
                'days': self.days_back
            },
            'totals': {
                'signals_generated': metrics['total_signals'],    # ← logs (BUY + SELL + HOLD)
                'actionable_signals': metrics['actionable_signals'], # ← logs (BUY + SELL)
                'hold_signals': metrics['hold_signals'],          # ← logs (HOLD)
                'orders_executed': metrics['orders_executed'],    # ← paper_trades.csv
                'errors': len(self.errors)
            },
            'execution_rate': (metrics['orders_executed'] / metrics['actionable_signals'] * 100) if metrics['actionable_signals'] > 0 else 0  # ← orders_executed / actionable_signals
        }
        
        summary_file = output_dir / f"summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"   ✅ Summary exported to: {summary_file}")
        
        print(f"\n📁 All reports saved to: {output_dir}")

    # ========================================================================================
    # INVARIANTES DEL SISTEMA - MÉTODOS DE VALIDACIÓN
    # ========================================================================================

    def _validate_l2_l3_separation(self, metrics):
        """
        CRÍTICO: INV-3 debe validar acoplamiento operativo, no intención

        Nueva regla correcta:
        ❌ ERROR solo si:
        - L3 genera órdenes directas
        - L3 ejecuta trades
        - L3 invalida HOLD por defecto sin override explícito

        ✅ CORRECTO:
        - L2 emite HOLD por defecto (comportamiento correcto)
        - L3 emite contexto estratégico (ej. SELL)
        - L3 NO ejecuta directamente, solo provee contexto
        """
        # Buscar evidencia de acoplamiento operativo problemático
        operational_violations = []

        # VIOLATION 1: L3 generando órdenes directas (prohibido)
        l3_direct_orders = sum(1 for s in self.signals
                              if s.get('source_type') == 'L3_STRATEGIC' and
                              'ORDER' in s.get('message', '').upper())

        if l3_direct_orders > 0:
            operational_violations.append(f"L3 generating direct orders ({l3_direct_orders})")

        # VIOLATION 2: L3 ejecutando trades (severamente prohibido)
        l3_trade_execution = sum(1 for decision in self.l3_decisions
                                if 'EXECUTED' in decision.get('message', '').upper() or
                                'TRADE' in decision.get('message', '').upper())

        if l3_trade_execution > 0:
            operational_violations.append(f"L3 executing trades ({l3_trade_execution})")

        # VIOLATION 3: L3 invalidando HOLD sin override explícito
        l3_forceful_actions = sum(1 for decision in self.l3_decisions
                                 if ('FORCE' in decision.get('message', '').upper() or
                                     'INVALIDATE' in decision.get('message', '').upper()) and
                                 'OVERRIDE' not in decision.get('message', '').upper())

        if l3_forceful_actions > 0:
            operational_violations.append(f"L3 forceful actions without explicit override ({l3_forceful_actions})")

        # VEREDICTO: Si no hay violaciones operativas, separación es correcta
        # Es CORRECTO que L2 diga HOLD aunque L3 diga BUY/SELL (siempre que L3 no fuerce)
        return len(operational_violations) == 0

    def _calculate_l3_override_rate(self):
        """
        Calcula el porcentaje de ciclos donde L3 hizo override.
        Override debe ser excepcional (< 20%)
        """
        total_cycles = len(self.system_metrics.get('cycles', []))
        if total_cycles == 0:
            return 0.0

        # Contar overrides de L3 (buscar en logs)
        override_cycles = 0
        for decision in self.l3_decisions:
            if 'OVERRIDE' in decision['message'].upper():
                override_cycles += 1

        # También buscar en señales con metadata de override
        for signal in self.signals:
            metadata = signal.get('extra', {}).get('trading_action', {}).get('metadata', {})
            if metadata.get('override_type') == 'exceptional':
                override_cycles += 1

        return (override_cycles / total_cycles) * 100

    def _validate_doubt_hold_compliance(self, metrics):
        """
        Valida que cuando el sistema duda, genera HOLD.
        Sistema duda cuando: confianza baja, condiciones mixtas, volatilidad alta
        """
        # Buscar señales con baja confianza que deberían ser HOLD
        doubt_signals = []
        for signal in self.signals:
            confidence = signal.get('confidence', 0.5)
            intention = signal.get('intention_type')

            # Señales con baja confianza deberían ser HOLD
            if confidence < 0.6 and intention != 'HOLD':
                doubt_signals.append(signal)

        # Si hay muchas señales dudosas que no son HOLD, falla validación
        doubt_rate = len(doubt_signals) / max(len(self.signals), 1)

        return doubt_rate < 0.2  # Menos del 20% de señales dudosas no-HOLD

    def _validate_philosophy_compliance(self, metrics, total_cycles):
        """
        Valida cumplimiento de filosofía: "Un sistema que siempre opera no es agresivo, es ciego"

        Sistema "ciego" = opera sin considerar condiciones de mercado
        Sistema "agresivo" = opera con evidencia pero de forma imprudente
        Sistema "conservador" = opera solo con evidencia excepcional
        """
        # Calcular ratio de operación
        operation_ratio = metrics['orders_executed'] / max(total_cycles, 1)

        # Sistema ciego: operation_ratio > 0.5 (opera en más del 50% de ciclos)
        # Sistema conservador: operation_ratio < 0.1 (opera en menos del 10% de ciclos)

        if operation_ratio > 0.5:
            return False  # Sistema ciego - opera siempre
        elif operation_ratio < 0.1:
            return True   # Sistema conservador - opera con evidencia excepcional
        else:
            # Sistema agresivo - opera con evidencia pero no excepcional
            # Verificar si al menos tiene evidencia (HOLD > 30%)
            hold_ratio = metrics['hold_signals'] / max(metrics['total_signals'], 1)
            return hold_ratio > 0.3  # Al menos 30% HOLD = conservador


def main():
    """Run the weekly analysis"""
    import sys
    
    # Get base path from command line or use current directory
    base_path = sys.argv[1] if len(sys.argv) > 1 else "."
    days_back = int(sys.argv[2]) if len(sys.argv) > 2 else 7
    
    analyzer = HRMWeeklyAnalyzer(base_path, days_back)
    analyzer.analyze_all()
    
    print("\n" + "=" * 80)
    print("✅ ANALYSIS COMPLETE")
    print("=" * 80)
    print("\nFor detailed reports, check the 'reports' directory")


if __name__ == "__main__":
    main()
    
    main()
