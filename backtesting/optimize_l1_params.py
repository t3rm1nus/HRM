#!/usr/bin/env python3
"""
Optimizador de parámetros L1 - Grid Search

Ejecuta grid search sobre los parámetros clave del backtest L1
para encontrar la combinación óptima.

Uso:
    python backtesting/optimize_l1_params.py

Output:
    - backtesting/results/l1_optimization_TIMESTAMP.csv
    - Top 10 combinaciones en consola
"""

import os
import sys
import asyncio
import itertools
import time
import csv
from datetime import datetime
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass

# Añadir el directorio raíz al path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from backtesting.backtest_l1_standalone import run_backtest, DEFAULT_SYMBOLS


@dataclass
class OptimizationResult:
    """Resultado de una combinación de parámetros"""
    stop_loss_pct: float
    take_profit_pct: float
    cooldown_minutes: int
    min_confidence: float
    win_rate: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown_pct: float
    total_trades: int
    total_pnl: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'stop_loss_pct': self.stop_loss_pct,
            'take_profit_pct': self.take_profit_pct,
            'cooldown_minutes': self.cooldown_minutes,
            'min_confidence': self.min_confidence,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown_pct': self.max_drawdown_pct,
            'total_trades': self.total_trades,
            'total_pnl': self.total_pnl,
        }
    
    def is_valid(self) -> bool:
        """Verifica si cumple criterios mínimos"""
        return (
            self.profit_factor > 1.0 
            and self.win_rate > 0.45 
            and self.total_trades >= 10
        )


async def run_single_backtest(
    params: Tuple[float, float, int, float],
    semaphore: asyncio.Semaphore
) -> OptimizationResult:
    """
    Ejecuta un backtest con parámetros específicos
    
    Args:
        params: (stop_loss_pct, take_profit_pct, cooldown_minutes, min_confidence)
        semaphore: Semáforo para limitar concurrencia
    """
    stop_loss_pct, take_profit_pct, cooldown_minutes, min_confidence = params
    
    async with semaphore:
        print(f"  → Probando: SL={stop_loss_pct:.3f}, TP={take_profit_pct:.3f}, "
              f"CD={cooldown_minutes}m, CONF={min_confidence:.2f}")
        
        try:
            result = await run_backtest(
                symbols=DEFAULT_SYMBOLS,
                timeframe='1m',
                days=7,
                initial_balance=3000.0,
                position_size_pct=0.1,
                min_confidence=min_confidence,
                use_cache=True,
                output_dir=None,  # No exportar CSVs individuales
                sample_every=10,
                stop_loss_pct=stop_loss_pct,
                take_profit_pct=take_profit_pct,
                cooldown_minutes=cooldown_minutes,
            )
            
            if result and 'metrics' in result:
                metrics = result['metrics']
                
                # Parsear métricas (vienen como strings formateados)
                def parse_pct(value):
                    if isinstance(value, str):
                        return float(value.replace('%', '')) / 100
                    return float(value)
                
                def parse_float(value):
                    if isinstance(value, str):
                        return float(value.replace(',', ''))
                    return float(value)
                
                return OptimizationResult(
                    stop_loss_pct=stop_loss_pct,
                    take_profit_pct=take_profit_pct,
                    cooldown_minutes=cooldown_minutes,
                    min_confidence=min_confidence,
                    win_rate=parse_pct(metrics.get('win_rate', '0%')),
                    profit_factor=parse_float(metrics.get('profit_factor', '0')),
                    sharpe_ratio=parse_float(metrics.get('sharpe_ratio', '0')),
                    max_drawdown_pct=parse_pct(metrics.get('max_drawdown_pct', '0%')),
                    total_trades=int(metrics.get('total_trades', 0)),
                    total_pnl=parse_float(metrics.get('total_pnl', '0')),
                )
            else:
                # Resultado vacío
                return OptimizationResult(
                    stop_loss_pct=stop_loss_pct,
                    take_profit_pct=take_profit_pct,
                    cooldown_minutes=cooldown_minutes,
                    min_confidence=min_confidence,
                    win_rate=0.0,
                    profit_factor=0.0,
                    sharpe_ratio=0.0,
                    max_drawdown_pct=0.0,
                    total_trades=0,
                    total_pnl=0.0,
                )
                
        except Exception as e:
            print(f"    ⚠️ Error en backtest: {e}")
            return OptimizationResult(
                stop_loss_pct=stop_loss_pct,
                take_profit_pct=take_profit_pct,
                cooldown_minutes=cooldown_minutes,
                min_confidence=min_confidence,
                win_rate=0.0,
                profit_factor=0.0,
                sharpe_ratio=0.0,
                max_drawdown_pct=0.0,
                total_trades=0,
                total_pnl=0.0,
            )


async def run_optimization():
    """Ejecuta el grid search completo"""
    
    print("=" * 80)
    print("🔬 OPTIMIZACIÓN DE PARÁMETROS L1 - GRID SEARCH")
    print("=" * 80)
    
    # Definir rangos de parámetros
    param_grid = {
        'stop_loss_pct': [0.010, 0.015, 0.020],
        'take_profit_pct': [0.020, 0.030, 0.040],
        'cooldown_minutes': [120, 240, 480],
        'min_confidence': [0.55, 0.60, 0.65, 0.70],
    }
    
    # Generar todas las combinaciones
    param_combinations = list(itertools.product(
        param_grid['stop_loss_pct'],
        param_grid['take_profit_pct'],
        param_grid['cooldown_minutes'],
        param_grid['min_confidence']
    ))
    
    total_combinations = len(param_combinations)
    print(f"\n📊 Total de combinaciones a probar: {total_combinations}")
    print(f"   SL: {param_grid['stop_loss_pct']}")
    print(f"   TP: {param_grid['take_profit_pct']}")
    print(f"   CD: {param_grid['cooldown_minutes']}")
    print(f"   CF: {param_grid['min_confidence']}")
    print()
    
    # Semáforo para limitar a 4 backtests simultáneos
    semaphore = asyncio.Semaphore(4)
    
    # Ejecutar todos los backtests
    start_time = time.time()
    print("⏳ Ejecutando backtests...")
    
    tasks = [run_single_backtest(params, semaphore) for params in param_combinations]
    results = await asyncio.gather(*tasks)
    
    elapsed = time.time() - start_time
    print(f"\n✅ Grid search completado en {elapsed:.1f} segundos")
    
    # Filtrar resultados con menos de 10 trades
    valid_results = [r for r in results if r.total_trades >= 10]
    print(f"   Combinaciones válidas (≥10 trades): {len(valid_results)}/{total_combinations}")
    
    # Ordenar por profit_factor descendente
    sorted_results = sorted(valid_results, key=lambda x: x.profit_factor, reverse=True)
    
    return sorted_results


def save_results(results: List[OptimizationResult]):
    """Guarda los resultados en CSV"""
    
    # Crear directorio si no existe
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"l1_optimization_{timestamp}.csv"
    filepath = os.path.join(results_dir, filename)
    
    # Escribir CSV
    with open(filepath, 'w', newline='') as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=results[0].to_dict().keys())
            writer.writeheader()
            for result in results:
                writer.writerow(result.to_dict())
    
    print(f"\n💾 Resultados guardados en: {filepath}")
    return filepath


def print_top_results(results: List[OptimizationResult], top_n: int = 10):
    """Muestra los mejores resultados en formato tabla"""
    
    print(f"\n🏆 TOP {top_n} COMBINACIONES (ordenadas por profit_factor)")
    print("=" * 120)
    
    # Headers
    print(f"{'Rank':<6} {'SL%':<6} {'TP%':<6} {'CD':<5} {'Conf':<6} "
          f"{'Win%':<8} {'PF':<8} {'Sharpe':<8} {'DD%':<8} {'Trades':<8} "
          f"{'PnL':<10} {'✓':<3}")
    print("-" * 120)
    
    for i, result in enumerate(results[:top_n], 1):
        valid_marker = "✅" if result.is_valid() else ""
        
        print(f"{i:<6} "
              f"{result.stop_loss_pct*100:<6.1f} "
              f"{result.take_profit_pct*100:<6.1f} "
              f"{result.cooldown_minutes:<5} "
              f"{result.min_confidence:<6.2f} "
              f"{result.win_rate*100:<8.1f} "
              f"{result.profit_factor:<8.2f} "
              f"{result.sharpe_ratio:<8.2f} "
              f"{result.max_drawdown_pct*100:<8.1f} "
              f"{result.total_trades:<8} "
              f"{result.total_pnl:<10.2f} "
              f"{valid_marker:<3}")
    
    print("=" * 120)
    print("\n✅ = Cumple criterios: PF > 1.0 AND Win% > 45% AND Trades >= 10")


def print_summary(results: List[OptimizationResult]):
    """Muestra resumen estadístico"""
    
    if not results:
        print("\n⚠️ No hay resultados suficientes para análisis")
        return
    
    print("\n📈 RESUMEN ESTADÍSTICO")
    print("=" * 60)
    
    # Contar válidos
    valid_count = sum(1 for r in results if r.is_valid())
    print(f"Combinaciones válidas: {valid_count}/{len(results)}")
    
    if valid_count > 0:
        valid_results = [r for r in results if r.is_valid()]
        
        # Mejores métricas
        best_pf = max(valid_results, key=lambda x: x.profit_factor)
        best_wr = max(valid_results, key=lambda x: x.win_rate)
        best_sharpe = max(valid_results, key=lambda x: x.sharpe_ratio)
        best_pnl = max(valid_results, key=lambda x: x.total_pnl)
        
        print(f"\nMejor Profit Factor:  {best_pf.profit_factor:.2f} "
              f"(SL={best_pf.stop_loss_pct*100:.1f}%, TP={best_pf.take_profit_pct*100:.1f}%)")
        print(f"Mejor Win Rate:       {best_wr.win_rate*100:.1f}% "
              f"(SL={best_wr.stop_loss_pct*100:.1f}%, TP={best_wr.take_profit_pct*100:.1f}%)")
        print(f"Mejor Sharpe:         {best_sharpe.sharpe_ratio:.2f} "
              f"(SL={best_sharpe.stop_loss_pct*100:.1f}%, TP={best_sharpe.take_profit_pct*100:.1f}%)")
        print(f"Mejor PnL Total:      ${best_pnl.total_pnl:.2f} "
              f"(SL={best_pnl.stop_loss_pct*100:.1f}%, TP={best_pnl.take_profit_pct*100:.1f}%)")
    
    print("=" * 60)


async def main():
    """Función principal"""
    
    print("\n🚀 Iniciando optimización de parámetros L1...")
    print("   Presiona Ctrl+C para cancelar\n")
    
    try:
        # Ejecutar optimización
        results = await run_optimization()
        
        if not results:
            print("\n❌ No se encontraron resultados válidos")
            return 1
        
        # Guardar resultados
        filepath = save_results(results)
        
        # Mostrar mejores resultados
        print_top_results(results, top_n=10)
        
        # Mostrar resumen
        print_summary(results)
        
        print(f"\n✅ Optimización completada exitosamente")
        print(f"   Archivo: {filepath}")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Optimización cancelada por el usuario")
        return 130
    except Exception as e:
        print(f"\n❌ Error en optimización: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
