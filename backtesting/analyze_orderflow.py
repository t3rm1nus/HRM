#!/usr/bin/env python3
"""
Script de análisis de datos de Order Flow para HRM.

Analiza todos los CSVs en backtesting/data/orderflow/ y genera:
1. Cobertura temporal (gaps, % de tiempo cubierto)
2. Estadísticas descriptivas por símbolo
3. Correlación con movimientos de precio (1, 5, 15 min)
4. Distribución temporal por regímenes de mercado

Uso: python backtesting/analyze_orderflow.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import glob
import sys
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Configuración
ORDERFLOW_DIR = Path(__file__).parent / "data" / "orderflow"
OUTPUT_DIR = Path(__file__).parent / "results"
OUTPUT_DIR.mkdir(exist_ok=True)

# Columnas de métricas a analizar
METRICS = [
    'bid_ask_imbalance',
    'book_imbalance_5',
    'delta_volume_1m',
    'buy_pressure_1m',
    'spread_bps'
]

PRICE_COL = 'weighted_mid_price'


def load_all_orderflow_data() -> pd.DataFrame:
    """Carga todos los CSVs de orderflow en un solo DataFrame."""
    csv_files = sorted(ORDERFLOW_DIR.glob("orderflow_*.csv"))
    
    if not csv_files:
        print(f"❌ No se encontraron archivos CSV en {ORDERFLOW_DIR}")
        sys.exit(1)
    
    print(f"📁 Encontrados {len(csv_files)} archivos CSV")
    
    dfs = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            df['source_file'] = csv_file.name
            dfs.append(df)
            print(f"   ✓ {csv_file.name}: {len(df):,} registros")
        except Exception as e:
            print(f"   ✗ Error cargando {csv_file.name}: {e}")
    
    if not dfs:
        print("❌ No se pudo cargar ningún archivo")
        sys.exit(1)
    
    combined = pd.concat(dfs, ignore_index=True)
    combined['timestamp'] = pd.to_datetime(combined['timestamp'])
    combined = combined.sort_values('timestamp').reset_index(drop=True)
    
    print(f"\n📊 Total registros combinados: {len(combined):,}")
    print(f"📊 Símbolos: {combined['symbol'].unique().tolist()}")
    
    return combined


def analyze_temporal_coverage(df: pd.DataFrame) -> Dict:
    """Analiza la cobertura temporal de los datos."""
    print("\n" + "="*70)
    print("1. COBERTURA TEMPORAL")
    print("="*70)
    
    results = {}
    
    for symbol in df['symbol'].unique():
        symbol_df = df[df['symbol'] == symbol].copy()
        symbol_df = symbol_df.sort_values('timestamp')
        
        start_time = symbol_df['timestamp'].min()
        end_time = symbol_df['timestamp'].max()
        total_records = len(symbol_df)
        
        # Calcular gaps de más de 60 segundos
        time_diffs = symbol_df['timestamp'].diff().dt.total_seconds()
        gaps = time_diffs[time_diffs > 60]
        
        # Cobertura esperada vs real
        time_span_hours = (end_time - start_time).total_seconds() / 3600
        expected_records = time_span_hours * 60 * 60 / 5  # Asumiendo datos cada ~5s
        coverage_pct = min(100, (total_records / expected_records) * 100) if expected_records > 0 else 0
        
        results[symbol] = {
            'start': start_time,
            'end': end_time,
            'total_records': total_records,
            'gaps_over_60s': len(gaps),
            'max_gap_seconds': gaps.max() if len(gaps) > 0 else 0,
            'coverage_pct': coverage_pct,
            'time_span_hours': time_span_hours
        }
        
        print(f"\n📈 {symbol}:")
        print(f"   Periodo: {start_time} → {end_time}")
        print(f"   Duración: {time_span_hours:.1f} horas")
        print(f"   Registros: {total_records:,}")
        print(f"   Gaps >60s: {len(gaps)}")
        if len(gaps) > 0:
            print(f"   Gap máximo: {gaps.max():.0f} segundos ({gaps.max()/60:.1f} min)")
        print(f"   Cobertura estimada: {coverage_pct:.1f}%")
    
    return results


def analyze_descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula estadísticas descriptivas por símbolo y métrica."""
    print("\n" + "="*70)
    print("2. ESTADÍSTICAS DESCRIPTIVAS POR SÍMBOLO")
    print("="*70)
    
    all_stats = []
    
    for symbol in df['symbol'].unique():
        symbol_df = df[df['symbol'] == symbol]
        
        print(f"\n📈 {symbol}:")
        
        for metric in METRICS:
            if metric not in symbol_df.columns:
                continue
                
            data = symbol_df[metric].dropna()
            
            stats = {
                'symbol': symbol,
                'metric': metric,
                'count': len(data),
                'mean': data.mean(),
                'std': data.std(),
                'min': data.min(),
                'p10': data.quantile(0.10),
                'p25': data.quantile(0.25),
                'p50': data.quantile(0.50),
                'p75': data.quantile(0.75),
                'p90': data.quantile(0.90),
                'max': data.max()
            }
            all_stats.append(stats)
            
            print(f"   {metric:25s}: mean={stats['mean']:10.4f}  std={stats['std']:8.4f}  "
                  f"p10={stats['p10']:8.4f}  p90={stats['p90']:8.4f}")
        
        # Análisis de large_trade_detected
        if 'large_trade_detected' in symbol_df.columns:
            large_pct = symbol_df['large_trade_detected'].mean() * 100
            print(f"   {'large_trade_detected':25s}: {large_pct:.2f}% de veces True")
            all_stats.append({
                'symbol': symbol,
                'metric': 'large_trade_detected_pct',
                'mean': large_pct,
                'count': len(symbol_df)
            })
    
    return pd.DataFrame(all_stats)


def analyze_price_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula correlaciones entre métricas y movimientos de precio futuros."""
    print("\n" + "="*70)
    print("3. CORRELACIÓN CON MOVIMIENTOS DE PRECIO")
    print("="*70)
    print("\nAnalizando poder predictivo de cada métrica...")
    
    horizons = [1, 5, 15]  # minutos
    all_correlations = []
    
    for symbol in df['symbol'].unique():
        symbol_df = df[df['symbol'] == symbol].copy()
        symbol_df = symbol_df.sort_values('timestamp').reset_index(drop=True)
        
        if PRICE_COL not in symbol_df.columns:
            print(f"⚠️  {symbol}: No hay columna de precio {PRICE_COL}")
            continue
        
        print(f"\n📈 {symbol}:")
        
        for horizon in horizons:
            # Calcular retorno futuro
            symbol_df[f'future_return_{horizon}m'] = (
                symbol_df[PRICE_COL].shift(-horizon) - symbol_df[PRICE_COL]
            ) / symbol_df[PRICE_COL] * 100
            
            print(f"   Horizonte {horizon} minutos:")
            
            for metric in METRICS:
                if metric not in symbol_df.columns:
                    continue
                
                # Calcular correlación
                corr = symbol_df[metric].corr(symbol_df[f'future_return_{horizon}m'])
                
                if pd.isna(corr):
                    continue
                
                all_correlations.append({
                    'symbol': symbol,
                    'metric': metric,
                    'horizon_minutes': horizon,
                    'correlation': corr,
                    'abs_correlation': abs(corr),
                    'predictive_power': 'ALTA' if abs(corr) > 0.1 else 'MEDIA' if abs(corr) > 0.05 else 'BAJA'
                })
                
                direction = "📈" if corr > 0 else "📉"
                print(f"      {direction} {metric:25s}: r={corr:7.4f}  ({all_correlations[-1]['predictive_power']})")
    
    return pd.DataFrame(all_correlations)


def analyze_temporal_distribution(df: pd.DataFrame) -> Dict:
    """Analiza la distribución temporal y regímenes de mercado."""
    print("\n" + "="*70)
    print("4. DISTRIBUCIÓN TEMPORAL Y REGÍMENES DE MERCADO")
    print("="*70)
    
    results = {}
    
    for symbol in df['symbol'].unique():
        symbol_df = df[df['symbol'] == symbol].copy()
        symbol_df = symbol_df.sort_values('timestamp')
        
        # Calcular retornos por hora
        symbol_df['hour'] = symbol_df['timestamp'].dt.hour
        symbol_df['returns'] = symbol_df[PRICE_COL].pct_change() * 100
        
        # Calcular tendencia por ventanas de 1 hora
        symbol_df['trend_1h'] = symbol_df[PRICE_COL].rolling(window=720, min_periods=360).apply(
            lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0] * 100 if len(x) > 0 else 0
        )
        
        # Clasificar regímenes
        symbol_df['regime'] = symbol_df['trend_1h'].apply(
            lambda x: 'ALCISTA' if x > 0.5 else 'BAJISTA' if x < -0.5 else 'LATERAL'
        )
        
        regime_counts = symbol_df['regime'].value_counts()
        total = len(symbol_df)
        
        results[symbol] = {
            'alcista_pct': regime_counts.get('ALCISTA', 0) / total * 100,
            'bajista_pct': regime_counts.get('BAJISTA', 0) / total * 100,
            'lateral_pct': regime_counts.get('LATERAL', 0) / total * 100,
            'hourly_activity': symbol_df.groupby('hour').size().to_dict()
        }
        
        print(f"\n📈 {symbol}:")
        print(f"   Régimen ALCISTA:  {results[symbol]['alcista_pct']:.1f}%")
        print(f"   Régimen BAJISTA:  {results[symbol]['bajista_pct']:.1f}%")
        print(f"   Régimen LATERAL:  {results[symbol]['lateral_pct']:.1f}%")
        
        # Horas más activas
        hourly = symbol_df.groupby('hour').size().sort_values(ascending=False)
        print(f"   Horas más activas: {', '.join([f'{h}:00' for h in hourly.head(3).index])}")
    
    return results


def generate_report(coverage: Dict, stats: pd.DataFrame, correlations: pd.DataFrame, 
                   regimes: Dict, output_file: str = None):
    """Genera un reporte en texto plano."""
    
    if output_file is None:
        output_file = OUTPUT_DIR / f"orderflow_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("ANÁLISIS DE ORDER FLOW - HRM\n")
        f.write(f"Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*70 + "\n\n")
        
        # Cobertura temporal
        f.write("1. COBERTURA TEMPORAL\n")
        f.write("-"*70 + "\n")
        for symbol, data in coverage.items():
            f.write(f"\n{symbol}:\n")
            f.write(f"  Periodo: {data['start']} -> {data['end']}\n")
            f.write(f"  Duración: {data['time_span_hours']:.1f} horas\n")
            f.write(f"  Registros: {data['total_records']:,}\n")
            f.write(f"  Gaps >60s: {data['gaps_over_60s']}\n")
            f.write(f"  Cobertura: {data['coverage_pct']:.1f}%\n")
        
        # Estadísticas
        f.write("\n\n2. ESTADÍSTICAS DESCRIPTIVAS\n")
        f.write("-"*70 + "\n")
        for symbol in stats['symbol'].unique():
            f.write(f"\n{symbol}:\n")
            symbol_stats = stats[stats['symbol'] == symbol]
            for _, row in symbol_stats.iterrows():
                if 'large_trade' in row['metric']:
                    f.write(f"  {row['metric']}: {row['mean']:.2f}%\n")
                elif pd.notna(row.get('mean')):
                    f.write(f"  {row['metric']}: mean={row['mean']:.4f}, std={row.get('std', 'N/A')}\n")
        
        # Correlaciones
        f.write("\n\n3. CORRELACIONES CON PRECIO (TOP 10)\n")
        f.write("-"*70 + "\n")
        if len(correlations) > 0:
            top_corr = correlations.nlargest(10, 'abs_correlation')
            for _, row in top_corr.iterrows():
                f.write(f"\n{row['symbol']} - {row['metric']} ({row['horizon_minutes']}min):\n")
                f.write(f"  Correlación: {row['correlation']:.4f}\n")
                f.write(f"  Poder predictivo: {row['predictive_power']}\n")
        
        # Regímenes
        f.write("\n\n4. DISTRIBUCIÓN DE REGÍMENES\n")
        f.write("-"*70 + "\n")
        for symbol, data in regimes.items():
            f.write(f"\n{symbol}:\n")
            f.write(f"  Alcista: {data['alcista_pct']:.1f}%\n")
            f.write(f"  Bajista: {data['bajista_pct']:.1f}%\n")
            f.write(f"  Lateral: {data['lateral_pct']:.1f}%\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("FIN DEL REPORTE\n")
        f.write("="*70 + "\n")
    
    print(f"\n✅ Reporte guardado en: {output_file}")
    return output_file


def main():
    """Función principal."""
    print("="*70)
    print("ANÁLISIS DE ORDER FLOW - HRM")
    print("="*70)
    print(f"Directorio de datos: {ORDERFLOW_DIR}")
    print(f"Directorio de salida: {OUTPUT_DIR}")
    print("="*70)
    
    # 1. Cargar datos
    df = load_all_orderflow_data()
    
    # 2. Análisis de cobertura temporal
    coverage = analyze_temporal_coverage(df)
    
    # 3. Estadísticas descriptivas
    stats = analyze_descriptive_stats(df)
    stats.to_csv(OUTPUT_DIR / "orderflow_descriptive_stats.csv", index=False)
    print(f"\n✅ Estadísticas guardadas en: {OUTPUT_DIR / 'orderflow_descriptive_stats.csv'}")
    
    # 4. Correlaciones con precio
    correlations = analyze_price_correlations(df)
    if len(correlations) > 0:
        correlations.to_csv(OUTPUT_DIR / "orderflow_correlations.csv", index=False)
        print(f"✅ Correlaciones guardadas en: {OUTPUT_DIR / 'orderflow_correlations.csv'}")
    
    # 5. Distribución temporal
    regimes = analyze_temporal_distribution(df)
    
    # 6. Generar reporte
    report_file = generate_report(coverage, stats, correlations, regimes)
    
    print("\n" + "="*70)
    print("RESUMEN EJECUTIVO")
    print("="*70)
    
    # Resumen de calidad de datos
    total_records = len(df)
    total_gaps = sum(c['gaps_over_60s'] for c in coverage.values())
    avg_coverage = np.mean([c['coverage_pct'] for c in coverage.values()])
    
    print(f"\n📊 Calidad de datos:")
    print(f"   Total registros: {total_records:,}")
    print(f"   Cobertura promedio: {avg_coverage:.1f}%")
    print(f"   Total gaps >60s: {total_gaps}")
    
    if len(correlations) > 0:
        top_metric = correlations.loc[correlations['abs_correlation'].idxmax()]
        print(f"\n🔮 Métrica más predictiva:")
        print(f"   {top_metric['symbol']} - {top_metric['metric']} ({top_metric['horizon_minutes']}min)")
        print(f"   Correlación: {top_metric['correlation']:.4f}")
    
    print(f"\n✅ Análisis completado. Resultados en: {OUTPUT_DIR}")
    print("="*70)


if __name__ == "__main__":
    main()
