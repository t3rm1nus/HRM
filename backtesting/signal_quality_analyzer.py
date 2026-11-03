#!/usr/bin/env python3
"""
Signal Quality Analyzer - Analizador de calidad de señales de trading

Este script analiza el CSV de trades exportado por backtest_l1_standalone.py
y responde tres preguntas fundamentales:

1. ¿El modelo tiene dirección correcta?
   - Calcula directional_accuracy comparando side='BUY' con movimiento real del precio

2. ¿El problema es de timing o de dirección?
   - Analiza precios en 20min, 60min, 240min después de la entrada
   - Si accuracy mejora con más tiempo → problema de timing (SL muy corto)
   - Si accuracy no mejora → problema de dirección (señal incorrecta)

3. ¿Hay patrones en los trades ganadores?
   - Agrupa trades por hora del día, día de semana, símbolo
   - Detecta overfitting temporal (80/20 rule)

Usage:
    python backtesting/signal_quality_analyzer.py \\
        --trades-csv backtesting/results/l1_backtest_trades_YYYYMMDD_HHMMSS.csv \\
        --data-csv backtesting/data/btcusdt_1m.csv \\
        --output reporte.json
"""

import os
import sys
import argparse
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class SignalQualityAnalyzer:
    """Analizador de calidad de señales de trading"""
    
    def __init__(self, trades_csv: str, data_csv: str, signals_csv: Optional[str] = None):
        """
        Args:
            trades_csv: Path al CSV de trades exportado por backtest_l1_standalone
            data_csv: Path al CSV de datos originales (OHLCV)
            signals_csv: Path opcional al CSV de señales generadas
        """
        self.trades_csv = trades_csv
        self.data_csv = data_csv
        self.signals_csv = signals_csv
        
        self.trades_df: Optional[pd.DataFrame] = None
        self.data_df: Optional[pd.DataFrame] = None
        self.signals_df: Optional[pd.DataFrame] = None
        self.avg_hold_time: float = 0.0
        
        self.results: Dict[str, Any] = {}
        
    def load_data(self) -> bool:
        """Carga los CSVs necesarios"""
        try:
            # Cargar trades
            logger.info(f"📁 Cargando trades desde: {self.trades_csv}")
            self.trades_df = pd.read_csv(self.trades_csv)
            self.trades_df['timestamp'] = pd.to_datetime(self.trades_df['timestamp'])
            if 'exit_timestamp' in self.trades_df.columns:
                self.trades_df['exit_timestamp'] = pd.to_datetime(self.trades_df['exit_timestamp'])
            
            logger.info(f"   ✓ {len(self.trades_df)} trades cargados")
            
            # Calcular avg_hold_time del backtest
            if 'hold_time_minutes' in self.trades_df.columns:
                self.avg_hold_time = self.trades_df['hold_time_minutes'].mean()
                logger.info(f"   ✓ Tiempo promedio de holding: {self.avg_hold_time:.1f} minutos")
            
            # Cargar datos de precios
            logger.info(f"📁 Cargando datos de precios desde: {self.data_csv}")
            self.data_df = pd.read_csv(self.data_csv)
            
            # Normalizar columnas
            self.data_df.columns = [c.lower() for c in self.data_df.columns]
            if 'timestamp' in self.data_df.columns:
                self.data_df['timestamp'] = pd.to_datetime(self.data_df['timestamp'])
                self.data_df = self.data_df.set_index('timestamp')
            elif 'datetime' in self.data_df.columns:
                self.data_df['datetime'] = pd.to_datetime(self.data_df['datetime'])
                self.data_df = self.data_df.set_index('datetime')
            elif 'date' in self.data_df.columns:
                self.data_df['date'] = pd.to_datetime(self.data_df['date'])
                self.data_df = self.data_df.set_index('date')
            
            # Asegurar que tenemos columna 'close'
            if 'close' not in self.data_df.columns:
                logger.error("❌ No se encontró columna 'close' en los datos de precios")
                return False
            
            logger.info(f"   ✓ {len(self.data_df)} velas cargadas")
            logger.info(f"   ✓ Rango: {self.data_df.index.min()} - {self.data_df.index.max()}")
            
            # Cargar señales si se proporcionaron
            if self.signals_csv and os.path.exists(self.signals_csv):
                logger.info(f"📁 Cargando señales desde: {self.signals_csv}")
                self.signals_df = pd.read_csv(self.signals_csv)
                self.signals_df['timestamp'] = pd.to_datetime(self.signals_df['timestamp'])
                logger.info(f"   ✓ {len(self.signals_df)} señales cargadas")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error cargando datos: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def analyze_directional_accuracy(self) -> Dict[str, Any]:
        """
        PREGUNTA 1: ¿El modelo tiene dirección correcta?
        
        Para cada trade BUY, compara si el precio subió realmente en los
        siguientes N minutos (N = avg_hold_time del backtest).
        """
        logger.info("\n" + "=" * 80)
        logger.info("📊 ANÁLISIS 1: DIRECCIÓN CORRECTA DEL MODELO")
        logger.info("=" * 80)
        
        if self.trades_df is None or self.data_df is None:
            logger.error("❌ Datos no cargados")
            return {}
        
        # Filtrar solo trades BUY (entradas largas)
        buy_trades = self.trades_df[self.trades_df['side'] == 'BUY'].copy()
        
        if len(buy_trades) == 0:
            logger.warning("⚠️ No hay trades BUY para analizar")
            return {'directional_accuracy': 0.0, 'total_trades': 0}
        
        logger.info(f"Analizando {len(buy_trades)} trades BUY...")
        logger.info(f"Ventana de tiempo: {self.avg_hold_time:.1f} minutos (promedio de hold time)")
        
        correct_directions = 0
        total_analyzed = 0
        analysis_details = []
        
        for idx, trade in buy_trades.iterrows():
            entry_time = trade['timestamp']
            entry_price = trade['entry_price']
            symbol = trade.get('symbol', 'UNKNOWN')
            
            # Calcular tiempo de salida esperado
            exit_time = entry_time + timedelta(minutes=self.avg_hold_time)
            
            # Buscar precio en el data_df
            try:
                # Encontrar el precio más cercano después del tiempo de salida esperado
                future_prices = self.data_df[self.data_df.index >= exit_time]
                
                if len(future_prices) == 0:
                    # Si no hay datos futuros, usar el último precio disponible
                    future_price = self.data_df['close'].iloc[-1]
                    actual_hold_time = (self.data_df.index[-1] - entry_time).total_seconds() / 60
                else:
                    future_price = future_prices['close'].iloc[0]
                    actual_hold_time = self.avg_hold_time
                
                # Determinar si la dirección fue correcta
                price_went_up = future_price > entry_price
                direction_correct = price_went_up  # Para BUY, queremos que suba
                
                if direction_correct:
                    correct_directions += 1
                
                total_analyzed += 1
                
                analysis_details.append({
                    'trade_id': trade.get('trade_id', idx),
                    'symbol': symbol,
                    'entry_time': entry_time.isoformat(),
                    'entry_price': float(entry_price),
                    'future_time': exit_time.isoformat(),
                    'future_price': float(future_price),
                    'price_change_pct': float((future_price - entry_price) / entry_price * 100),
                    'direction_correct': bool(direction_correct),
                    'actual_hold_time_min': float(actual_hold_time)
                })
                
            except Exception as e:
                logger.warning(f"⚠️ Error analizando trade {idx}: {e}")
                continue
        
        # Calcular accuracy
        directional_accuracy = correct_directions / total_analyzed if total_analyzed > 0 else 0.0
        
        results = {
            'directional_accuracy': float(directional_accuracy),
            'correct_predictions': int(correct_directions),
            'total_trades': int(total_analyzed),
            'avg_hold_time_used': float(self.avg_hold_time),
            'details': analysis_details[:50]  # Limitar a 50 para el JSON
        }
        
        # Reporte en consola
        logger.info(f"\n📈 RESULTADOS:")
        logger.info(f"   Trades analizados: {total_analyzed}")
        logger.info(f"   Predicciones correctas: {correct_directions}")
        logger.info(f"   Dirección correcta: {directional_accuracy:.2%}")
        
        if directional_accuracy > 0.55:
            logger.info(f"   ✅ El modelo tiene BUENA dirección (>55%)")
        elif directional_accuracy > 0.45:
            logger.info(f"   ⚠️ El modelo tiene dirección NEUTRA (~50%)")
        else:
            logger.info(f"   ❌ El modelo tiene MALA dirección (<45%)")
        
        self.results['directional_accuracy'] = results
        return results
    
    def analyze_timing_vs_direction(self) -> Dict[str, Any]:
        """
        PREGUNTA 2: ¿El problema es de timing o de dirección?
        
        Para cada BUY, calcula:
        - precio_20min_despues vs entry_price → ¿subió en 20min?
        - precio_60min_despues vs entry_price → ¿subió en 60min?
        - precio_240min_despues vs entry_price → ¿subió en 240min?
        
        Si la accuracy mejora con más tiempo → problema de timing (SL muy corto)
        Si la accuracy no mejora → problema de dirección (señal incorrecta)
        """
        logger.info("\n" + "=" * 80)
        logger.info("📊 ANÁLISIS 2: TIMING vs DIRECCIÓN")
        logger.info("=" * 80)
        
        if self.trades_df is None or self.data_df is None:
            logger.error("❌ Datos no cargados")
            return {}
        
        # Filtrar solo trades BUY
        buy_trades = self.trades_df[self.trades_df['side'] == 'BUY'].copy()
        
        if len(buy_trades) == 0:
            logger.warning("⚠️ No hay trades BUY para analizar")
            return {}
        
        time_windows = [20, 60, 240]  # minutos
        results_by_window = {}
        
        logger.info(f"Analizando {len(buy_trades)} trades BUY en ventanas de tiempo...")
        
        for window_minutes in time_windows:
            correct_predictions = 0
            total_analyzed = 0
            price_changes = []
            
            for idx, trade in buy_trades.iterrows():
                entry_time = trade['timestamp']
                entry_price = trade['entry_price']
                
                # Calcular tiempo de salida
                exit_time = entry_time + timedelta(minutes=window_minutes)
                
                try:
                    # Buscar precio en esa ventana
                    future_prices = self.data_df[self.data_df.index >= exit_time]
                    
                    if len(future_prices) == 0:
                        continue
                    
                    future_price = future_prices['close'].iloc[0]
                    price_change_pct = (future_price - entry_price) / entry_price * 100
                    price_went_up = future_price > entry_price
                    
                    if price_went_up:
                        correct_predictions += 1
                    
                    total_analyzed += 1
                    price_changes.append(price_change_pct)
                    
                except Exception as e:
                    continue
            
            accuracy = correct_predictions / total_analyzed if total_analyzed > 0 else 0.0
            avg_change = np.mean(price_changes) if price_changes else 0.0
            median_change = np.median(price_changes) if price_changes else 0.0
            
            results_by_window[window_minutes] = {
                'accuracy': float(accuracy),
                'correct_predictions': int(correct_predictions),
                'total_analyzed': int(total_analyzed),
                'avg_price_change_pct': float(avg_change),
                'median_price_change_pct': float(median_change),
                'min_change_pct': float(min(price_changes)) if price_changes else 0.0,
                'max_change_pct': float(max(price_changes)) if price_changes else 0.0
            }
        
        # Analizar tendencia de accuracy
        accuracies = [results_by_window[w]['accuracy'] for w in time_windows]
        accuracy_trend = "stable"
        
        if accuracies[2] > accuracies[0] + 0.1:  # Mejora significativa en 240min vs 20min
            accuracy_trend = "improving"
        elif accuracies[2] < accuracies[0] - 0.1:  # Empeora significativamente
            accuracy_trend = "worsening"
        
        # Determinar diagnóstico
        diagnosis = {}
        
        if accuracy_trend == "improving":
            diagnosis['primary_issue'] = "TIMING"
            diagnosis['description'] = "La accuracy mejora con más tiempo → problema de timing (SL muy corto)"
            diagnosis['recommendation'] = "Aumentar el stop-loss o reducir el take-profit para dar más tiempo a las operaciones"
        elif accuracies[0] < 0.45:
            diagnosis['primary_issue'] = "DIRECTION"
            diagnosis['description'] = "La accuracy es baja incluso a corto plazo → problema de dirección"
            diagnosis['recommendation'] = "Revisar la lógica de generación de señales, el modelo entra en dirección contraria"
        else:
            diagnosis['primary_issue'] = "MIXED"
            diagnosis['description'] = "La accuracy es neutral, no hay patrón claro de timing ni dirección"
            diagnosis['recommendation'] = "Recolectar más datos o revisar los filtros de confianza"
        
        results = {
            'time_windows': results_by_window,
            'accuracy_trend': accuracy_trend,
            'diagnosis': diagnosis
        }
        
        # Reporte en consola
        logger.info(f"\n📈 RESULTADOS POR VENTANA DE TIEMPO:")
        for window in time_windows:
            r = results_by_window[window]
            logger.info(f"   {window:3d} min: Accuracy={r['accuracy']:.2%} "
                       f"({r['correct_predictions']}/{r['total_analyzed']}) | "
                       f"Avg Change={r['avg_price_change_pct']:+.2f}%")
        
        logger.info(f"\n🔍 DIAGNÓSTICO:")
        logger.info(f"   Tendencia: {accuracy_trend.upper()}")
        logger.info(f"   Problema principal: {diagnosis['primary_issue']}")
        logger.info(f"   Descripción: {diagnosis['description']}")
        logger.info(f"   Recomendación: {diagnosis['recommendation']}")
        
        self.results['timing_vs_direction'] = results
        return results
    
    def analyze_winning_patterns(self) -> Dict[str, Any]:
        """
        PREGUNTA 3: ¿Hay patrones en los trades ganadores?
        
        Agrupa trades por:
        - Hora del día
        - Día de la semana
        - Símbolo
        
        Detecta overfitting temporal (si el 80% de las ganancias vienen del 20% del tiempo)
        """
        logger.info("\n" + "=" * 80)
        logger.info("📊 ANÁLISIS 3: PATRONES EN TRADES GANADORES")
        logger.info("=" * 80)
        
        if self.trades_df is None:
            logger.error("❌ Datos no cargados")
            return {}
        
        df = self.trades_df.copy()
        
        # Asegurar que tenemos columna PnL
        if 'pnl' not in df.columns:
            logger.error("❌ No se encontró columna 'pnl' en los trades")
            return {}
        
        # Añadir columnas de tiempo
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.day_name()
        df['day_of_week_num'] = df['timestamp'].dt.dayofweek
        df['is_winner'] = df['pnl'] > 0
        df['month'] = df['timestamp'].dt.month
        df['date'] = df['timestamp'].dt.date
        
        results = {}
        
        # 1. Análisis por hora del día
        logger.info("\n📅 Análisis por HORA DEL DÍA:")
        hour_analysis = df.groupby('hour').agg({
            'pnl': ['count', 'sum', 'mean'],
            'is_winner': 'sum'
        }).round(2)
        hour_analysis.columns = ['total_trades', 'total_pnl', 'avg_pnl', 'winners']
        hour_analysis['win_rate'] = (hour_analysis['winners'] / hour_analysis['total_trades'] * 100).round(1)
        hour_analysis = hour_analysis.sort_values('total_pnl', ascending=False)
        
        logger.info(hour_analysis.to_string())
        results['by_hour'] = hour_analysis.to_dict()
        
        # 2. Análisis por día de la semana
        logger.info("\n📅 Análisis por DÍA DE LA SEMANA:")
        day_analysis = df.groupby('day_of_week').agg({
            'pnl': ['count', 'sum', 'mean'],
            'is_winner': 'sum'
        }).round(2)
        day_analysis.columns = ['total_trades', 'total_pnl', 'avg_pnl', 'winners']
        day_analysis['win_rate'] = (day_analysis['winners'] / day_analysis['total_trades'] * 100).round(1)
        
        # Ordenar por día de semana
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_analysis = day_analysis.reindex([d for d in day_order if d in day_analysis.index])
        
        logger.info(day_analysis.to_string())
        results['by_day_of_week'] = day_analysis.to_dict()
        
        # 3. Análisis por símbolo
        if 'symbol' in df.columns:
            logger.info("\n📅 Análisis por SÍMBOLO:")
            symbol_analysis = df.groupby('symbol').agg({
                'pnl': ['count', 'sum', 'mean'],
                'is_winner': 'sum'
            }).round(2)
            symbol_analysis.columns = ['total_trades', 'total_pnl', 'avg_pnl', 'winners']
            symbol_analysis['win_rate'] = (symbol_analysis['winners'] / symbol_analysis['total_trades'] * 100).round(1)
            symbol_analysis = symbol_analysis.sort_values('total_pnl', ascending=False)
            
            logger.info(symbol_analysis.to_string())
            results['by_symbol'] = symbol_analysis.to_dict()
        
        # 4. Análisis de concentración temporal (Pareto / 80-20 rule)
        logger.info("\n📅 Análisis DE CONCENTRACIÓN TEMPORAL (80/20):")
        
        # Agrupar PnL por fecha
        pnl_by_date = df.groupby('date')['pnl'].sum().sort_values(ascending=False)
        total_pnl = pnl_by_date.sum()
        
        if total_pnl > 0 and len(pnl_by_date) > 1:
            # Calcular contribución acumulada
            pnl_cumsum = pnl_by_date.cumsum()
            pnl_cumsum_pct = pnl_cumsum / total_pnl * 100
            
            # Encontrar cuántos días generan el 80% de las ganancias
            days_for_80_pct = (pnl_cumsum_pct <= 80).sum() + 1
            total_days = len(pnl_by_date)
            pct_days = days_for_80_pct / total_days * 100
            
            # Calcular métrica de concentración (Gini-like)
            sorted_pnl = np.sort(pnl_by_date.values)
            n = len(sorted_pnl)
            cumsum = np.cumsum(sorted_pnl)
            gini = (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n if cumsum[-1] != 0 else 0
            
            concentration = {
                'total_days_with_trades': int(total_days),
                'days_generating_80pct_profits': int(days_for_80_pct),
                'pct_of_days_for_80pct': float(pct_days),
                'concentration_score': float(gini),
                'interpretation': ''
            }
            
            if pct_days < 25:
                concentration['interpretation'] = "🔴 ALTO RIESGO DE OVERFITTING: El 80% de las ganancias viene de menos del 25% de los días"
            elif pct_days < 40:
                concentration['interpretation'] = "🟡 MODERADO: Posible overfitting temporal, revisar estrategia"
            else:
                concentration['interpretation'] = "🟢 DISTRIBUCIÓN SALUDABLE: Las ganancias están bien distribuidas en el tiempo"
            
            logger.info(f"   Total días con trades: {total_days}")
            logger.info(f"   Días que generan 80% de ganancias: {days_for_80_pct} ({pct_days:.1f}%)")
            logger.info(f"   Score de concentración: {gini:.3f} (0=uniforme, 1=concentrado)")
            logger.info(f"   {concentration['interpretation']}")
            
            # Mostrar top 5 días más rentables
            logger.info(f"\n   Top 5 días más rentables:")
            for i, (date, pnl) in enumerate(pnl_by_date.head(5).items(), 1):
                logger.info(f"      {i}. {date}: ${pnl:.2f}")
            
            results['temporal_concentration'] = concentration
        
        # 5. Análisis de confianza vs resultado
        if 'confidence' in df.columns:
            logger.info("\n📅 Análisis CONFIANZA vs RESULTADO:")
            
            # Crear bins de confianza
            df['confidence_bin'] = pd.cut(df['confidence'], bins=[0, 0.5, 0.6, 0.7, 0.8, 1.0], 
                                          labels=['<0.5', '0.5-0.6', '0.6-0.7', '0.7-0.8', '>0.8'])
            
            conf_analysis = df.groupby('confidence_bin').agg({
                'pnl': ['count', 'sum', 'mean'],
                'is_winner': 'sum'
            }).round(2)
            conf_analysis.columns = ['total_trades', 'total_pnl', 'avg_pnl', 'winners']
            conf_analysis['win_rate'] = (conf_analysis['winners'] / conf_analysis['total_trades'] * 100).round(1)
            
            logger.info(conf_analysis.to_string())
            results['by_confidence'] = conf_analysis.to_dict()
        
        # 6. Análisis por tipo de señal
        if 'signal_type' in df.columns:
            logger.info("\n📅 Análisis por TIPO DE SEÑAL:")
            signal_type_analysis = df.groupby('signal_type').agg({
                'pnl': ['count', 'sum', 'mean'],
                'is_winner': 'sum'
            }).round(2)
            signal_type_analysis.columns = ['total_trades', 'total_pnl', 'avg_pnl', 'winners']
            signal_type_analysis['win_rate'] = (signal_type_analysis['winners'] / signal_type_analysis['total_trades'] * 100).round(1)
            signal_type_analysis = signal_type_analysis.sort_values('total_pnl', ascending=False)
            
            logger.info(signal_type_analysis.to_string())
            results['by_signal_type'] = signal_type_analysis.to_dict()
        
        self.results['winning_patterns'] = results
        return results
    
    def generate_summary(self) -> Dict[str, Any]:
        """Genera un resumen ejecutivo de todos los análisis"""
        logger.info("\n" + "=" * 80)
        logger.info("📋 RESUMEN EJECUTIVO")
        logger.info("=" * 80)
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'files': {
                'trades_csv': self.trades_csv,
                'data_csv': self.data_csv,
                'signals_csv': self.signals_csv
            },
            'executive_summary': {}
        }
        
        # Pregunta 1: Dirección
        if 'directional_accuracy' in self.results:
            da = self.results['directional_accuracy']
            accuracy = da['directional_accuracy']
            
            if accuracy > 0.55:
                verdict1 = "✅ BUENA: El modelo predice correctamente la dirección"
            elif accuracy > 0.45:
                verdict1 = "⚠️ NEUTRA: El modelo no tiene ventaja direccional"
            else:
                verdict1 = "❌ MALA: El modelo predice en dirección contraria"
            
            summary['executive_summary']['question_1_direction'] = {
                'accuracy': f"{accuracy:.2%}",
                'verdict': verdict1
            }
            logger.info(f"\n1. DIRECCIÓN: {accuracy:.2%} - {verdict1}")
        
        # Pregunta 2: Timing vs Dirección
        if 'timing_vs_direction' in self.results:
            tvd = self.results['timing_vs_direction']
            diagnosis = tvd['diagnosis']
            
            summary['executive_summary']['question_2_timing'] = {
                'primary_issue': diagnosis['primary_issue'],
                'description': diagnosis['description'],
                'trend': tvd['accuracy_trend']
            }
            logger.info(f"\n2. TIMING vs DIRECCIÓN: {diagnosis['primary_issue']}")
            logger.info(f"   {diagnosis['description']}")
        
        # Pregunta 3: Patrones
        if 'winning_patterns' in self.results:
            wp = self.results['winning_patterns']
            
            overfitting_risk = "Desconocido"
            if 'temporal_concentration' in wp:
                tc = wp['temporal_concentration']
                pct_days = tc['pct_of_days_for_80pct']
                
                if pct_days < 25:
                    overfitting_risk = "🔴 ALTO"
                elif pct_days < 40:
                    overfitting_risk = "🟡 MODERADO"
                else:
                    overfitting_risk = "🟢 BAJO"
                
                summary['executive_summary']['question_3_patterns'] = {
                    'temporal_concentration': f"{pct_days:.1f}%",
                    'overfitting_risk': overfitting_risk
                }
                logger.info(f"\n3. PATRONES: Concentración en {pct_days:.1f}% de días - Riesgo {overfitting_risk}")
        
        # Recomendaciones finales
        recommendations = []
        
        if 'directional_accuracy' in self.results:
            if self.results['directional_accuracy']['directional_accuracy'] < 0.5:
                recommendations.append("PRIORIDAD 1: Revisar la lógica de generación de señales - el modelo entra en dirección contraria")
        
        if 'timing_vs_direction' in self.results:
            if self.results['timing_vs_direction']['diagnosis']['primary_issue'] == 'TIMING':
                recommendations.append("PRIORIDAD 2: Ajustar parámetros de SL/TP - el modelo necesita más tiempo para desarrollarse")
        
        if 'winning_patterns' in self.results and 'temporal_concentration' in self.results['winning_patterns']:
            tc = self.results['winning_patterns']['temporal_concentration']
            if tc['pct_of_days_for_80pct'] < 30:
                recommendations.append("PRIORIDAD 3: Riesgo de overfitting temporal - diversificar períodos de entrenamiento")
        
        if not recommendations:
            recommendations.append("El modelo parece saludable. Considerar optimizar tamaño de posición.")
        
        summary['executive_summary']['recommendations'] = recommendations
        
        logger.info(f"\n📋 RECOMENDACIONES:")
        for i, rec in enumerate(recommendations, 1):
            logger.info(f"   {i}. {rec}")
        
        self.results['executive_summary'] = summary['executive_summary']
        return summary
    
    def export_results(self, output_path: Optional[str] = None) -> str:
        """Exporta los resultados a JSON"""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"signal_quality_analysis_{timestamp}.json"
        
        # Asegurar que el directorio existe
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        
        # Preparar datos para exportación (convertir tipos no serializables)
        export_data = self._prepare_for_json(self.results)
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logger.info(f"\n💾 Resultados exportados a: {output_path}")
        return output_path
    
    def _prepare_for_json(self, obj):
        """Prepara objetos para serialización JSON"""
        if isinstance(obj, dict):
            return {k: self._prepare_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._prepare_for_json(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        else:
            return obj
    
    def run_full_analysis(self) -> Dict[str, Any]:
        """Ejecuta el análisis completo"""
        logger.info("\n" + "=" * 80)
        logger.info("🔬 SIGNAL QUALITY ANALYZER - ANÁLISIS COMPLETO")
        logger.info("=" * 80)
        
        # Cargar datos
        if not self.load_data():
            logger.error("❌ No se pudieron cargar los datos")
            return {}
        
        # Ejecutar análisis
        self.analyze_directional_accuracy()
        self.analyze_timing_vs_direction()
        self.analyze_winning_patterns()
        
        # Generar resumen
        self.generate_summary()
        
        return self.results


def main():
    """Punto de entrada principal"""
    parser = argparse.ArgumentParser(
        description='Signal Quality Analyzer - Analiza la calidad de señales de trading',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  # Análisis completo
  python backtesting/signal_quality_analyzer.py \\
      --trades-csv backtesting/results/l1_backtest_trades_20260101_120000.csv \\
      --data-csv backtesting/data/btcusdt_1m.csv

  # Con archivo de señales adicional
  python backtesting/signal_quality_analyzer.py \\
      --trades-csv backtesting/results/l1_backtest_trades_20260101_120000.csv \\
      --data-csv backtesting/data/btcusdt_1m.csv \\
      --signals-csv backtesting/results/l1_backtest_signals_20260101_120000.csv \\
      --output mi_analisis.json
        """
    )
    
    parser.add_argument(
        '--trades-csv',
        required=True,
        help='Path al CSV de trades exportado por backtest_l1_standalone.py'
    )
    parser.add_argument(
        '--data-csv',
        required=True,
        help='Path al CSV de datos originales (OHLCV)'
    )
    parser.add_argument(
        '--signals-csv',
        default=None,
        help='Path opcional al CSV de señales generadas'
    )
    parser.add_argument(
        '--output',
        default=None,
        help='Path para guardar el JSON de resultados (default: auto-generado)'
    )
    
    args = parser.parse_args()
    
    # Verificar que los archivos existen
    if not os.path.exists(args.trades_csv):
        logger.error(f"❌ Archivo no encontrado: {args.trades_csv}")
        return 1
    
    if not os.path.exists(args.data_csv):
        logger.error(f"❌ Archivo no encontrado: {args.data_csv}")
        return 1
    
    # Ejecutar análisis
    try:
        analyzer = SignalQualityAnalyzer(
            trades_csv=args.trades_csv,
            data_csv=args.data_csv,
            signals_csv=args.signals_csv
        )
        
        results = analyzer.run_full_analysis()
        
        if results:
            # Exportar resultados
            output_path = analyzer.export_results(args.output)
            logger.info(f"\n✅ Análisis completado exitosamente")
            logger.info(f"   Resultados guardados en: {output_path}")
            return 0
        else:
            logger.error("\n❌ El análisis falló")
            return 1
            
    except KeyboardInterrupt:
        logger.info("\n⚠️ Análisis interrumpido por el usuario")
        return 130
    except Exception as e:
        logger.error(f"\n❌ Error en análisis: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == '__main__':
    sys.exit(main())
