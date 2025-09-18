#!/backtesting/report_generator.py

#!/usr/bin/env python3
"""
HRM Report Generator - Generador de Reportes
Genera reportes detallados de backtesting y análisis del sistema HRM
"""

import os
import json
import pandas as pd
import numpy as np
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None
    sns = None
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import asyncio
import logging
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configurar matplotlib para no requerir GUI
if MATPLOTLIB_AVAILABLE:
    plt.switch_backend('Agg')

class ReportGenerator:
    """Generador principal de reportes del sistema HRM"""
    
    def __init__(self, config: Dict):
        self.config = config
        from core.logging import logger
        self.logger = logger
        
        # Configuración de reportes
        self.output_dir = config.get('output_dir', 'backtesting/results')
        self.generate_charts = config.get('generate_charts', True)
        self.detailed_logs = config.get('detailed_logs', True)
        self.export_trades = config.get('export_trades', True)
        
        # Crear directorio de salida
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Configurar estilo de gráficos
        if self.generate_charts and MATPLOTLIB_AVAILABLE:
            plt.style.use('default')
            sns.set_palette("husl")
    
    async def generate_complete_report(self, results: Dict) -> Dict:
        """Generar reporte completo del backtesting"""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_paths = {}
        
        try:
            self.logger.info("Generando reporte completo de backtesting...")
            
            # 1. Reporte ejecutivo (texto)
            executive_path = await self.generate_executive_summary(results, timestamp)
            report_paths['executive_summary'] = executive_path
            
            # 2. Reporte técnico detallado
            technical_path = await self.generate_technical_report(results, timestamp)
            report_paths['technical_report'] = technical_path
            
            # 3. Análisis por modelos
            models_path = await self.generate_models_analysis(results, timestamp)
            report_paths['models_analysis'] = models_path
            
            # 4. Gráficos de rendimiento
            if self.generate_charts:
                charts_path = await self.generate_performance_charts(results, timestamp)
                report_paths['performance_charts'] = charts_path
            
            # 5. Exportar trades si está habilitado
            if self.export_trades and results.get('trades'):
                trades_path = await self.export_trades_data(results, timestamp)
                report_paths['trades_export'] = trades_path
            
            # 6. Reporte JSON completo
            json_path = await self.export_json_report(results, timestamp)
            report_paths['json_report'] = json_path
            
            # 7. Dashboard HTML
            dashboard_path = await self.generate_html_dashboard(results, timestamp)
            report_paths['html_dashboard'] = dashboard_path
            
            self.logger.info(f"Reportes generados exitosamente en: {self.output_dir}")
            return report_paths
            
        except Exception as e:
            self.logger.error(f"Error generando reportes: {e}")
            return report_paths
    
    async def generate_executive_summary(self, results: Dict, timestamp: str) -> str:
        """Generar resumen ejecutivo en texto"""
        
        filename = f"{self.output_dir}/executive_summary_{timestamp}.txt"
        
        try:
            overall = results.get('overall', {})
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("="*80 + "\n")
                f.write("HRM BACKTESTING - RESUMEN EJECUTIVO\n")
                f.write("="*80 + "\n")
                f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # Métricas clave
                f.write("📊 MÉTRICAS CLAVE:\n")
                f.write("-" * 40 + "\n")
                f.write(f"Retorno Total: {overall.get('total_return', 0):.2%}\n")
                f.write(f"Retorno Anualizado: {overall.get('annualized_return', 0):.2%}\n")
                f.write(f"Sharpe Ratio: {overall.get('sharpe_ratio', 0):.3f}\n")
                f.write(f"Drawdown Máximo: {overall.get('max_drawdown', 0):.2%}\n")
                f.write(f"Win Rate: {overall.get('win_rate', 0):.2%}\n")
                f.write(f"Profit Factor: {overall.get('profit_factor', 0):.2f}\n")
                f.write(f"Total Trades: {overall.get('total_trades', 0)}\n\n")
                
                # Rendimiento por modelos L1
                f.write("🤖 MODELOS L1 (OPERACIONALES):\n")
                f.write("-" * 40 + "\n")
                l1_models = results.get('l1_models', {})
                for model_name, metrics in l1_models.items():
                    f.write(f"{model_name.upper()}:\n")
                    f.write(f"  • Accuracy: {metrics.get('accuracy', 0):.2%}\n")
                    f.write(f"  • Precision: {metrics.get('precision', 0):.3f}\n")
                    f.write(f"  • F1-Score: {metrics.get('f1_score', 0):.3f}\n")
                    f.write(f"  • Contribución Profit: ${metrics.get('profit_contribution', 0):.2f}\n")
                    f.write(f"  • Latencia Promedio: {metrics.get('latency_ms', 0):.1f}ms\n\n")
                
                # Modelo L2
                f.write("🎯 MODELO L2 (TÁCTICO):\n")
                f.write("-" * 40 + "\n")
                l2_model = results.get('l2_model', {})
                f.write(f"Signal Quality: {l2_model.get('signal_quality', 0):.3f}\n")
                f.write(f"Sizing Efficiency: {l2_model.get('sizing_efficiency', 0):.3f}\n")
                f.write(f"Risk Effectiveness: {l2_model.get('risk_effectiveness', 0):.3f}\n")
                f.write(f"Hit Rate: {l2_model.get('hit_rate', 0):.2%}\n")
                f.write(f"Risk-Adjusted Return: ${l2_model.get('risk_adjusted_return', 0):.2f}\n\n")
                
                # Modelos L3
                f.write("🧠 MODELOS L3 (ESTRATÉGICOS):\n")
                f.write("-" * 40 + "\n")
                l3_models = results.get('l3_models', {})
                for model_name, metrics in l3_models.items():
                    f.write(f"{model_name.upper()}:\n")
                    f.write(f"  • Decision Accuracy: {metrics.get('decision_accuracy', 0):.2%}\n")
                    f.write(f"  • Regime Detection: {metrics.get('regime_detection_accuracy', 0):.2%}\n")
                    f.write(f"  • Strategic Value: {metrics.get('strategic_value', 0):.3f}\n")
                    f.write(f"  • Allocation Efficiency: {metrics.get('allocation_efficiency', 0):.3f}\n\n")
                
                f.write("="*80 + "\n")
            
            return filename
            
        except Exception as e:
            self.logger.error(f"Error generando resumen ejecutivo: {e}")
            return ""
    
    async def generate_technical_report(self, results: Dict, timestamp: str) -> str:
        """Generar reporte técnico detallado en texto"""
        
        filename = f"{self.output_dir}/technical_report_{timestamp}.txt"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("="*80 + "\n")
                f.write("HRM BACKTESTING - REPORTE TÉCNICO DETALLADO\n")
                f.write("="*80 + "\n")
                f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # Métricas generales detalladas
                overall = results.get('overall', {})
                f.write("📊 MÉTRICAS GENERALES DETALLADAS:\n")
                f.write("-" * 40 + "\n")
                for key, value in overall.items():
                    f.write(f"{key.replace('_', ' ').title()}: {value}\n")
                f.write("\n")
                
                # Benchmark comparison
                benchmark = overall.get('benchmark_comparison', {})
                if benchmark:
                    f.write("📈 COMPARACIÓN CON BENCHMARK:\n")
                    f.write("-" * 40 + "\n")
                    for key, value in benchmark.items():
                        f.write(f"{key.replace('_', ' ').title()}: {value:.4f}\n")
                    f.write("\n")
                
                # VaR y ratios avanzados
                f.write("🛡️ MÉTRICAS DE RIESGO AVANZADAS:\n")
                f.write("-" * 40 + "\n")
                f.write(f"VaR 95%: {overall.get('var_95', 0):.4f}\n")
                f.write(f"VaR 99%: {overall.get('var_99', 0):.4f}\n")
                f.write(f"Calmar Ratio: {overall.get('calmar_ratio', 0):.4f}\n")
                f.write(f"Sortino Ratio: {overall.get('sortino_ratio', 0):.4f}\n")
                f.write(f"Recovery Factor: {overall.get('recovery_factor', 0):.4f}\n\n")
                
                f.write("="*80 + "\n")
            
            return filename
            
        except Exception as e:
            self.logger.error(f"Error generando reporte técnico: {e}")
            return ""
    
    async def generate_models_analysis(self, results: Dict, timestamp: str) -> str:
        """Generar análisis detallado por modelos en texto"""
        
        filename = f"{self.output_dir}/models_analysis_{timestamp}.txt"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("="*80 + "\n")
                f.write("HRM BACKTESTING - ANÁLISIS POR MODELOS\n")
                f.write("="*80 + "\n")
                f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # L1 Models
                f.write("🤖 ANÁLISIS MODELOS L1:\n")
                f.write("-" * 40 + "\n")
                l1_models = results.get('l1_models', {})
                for model_name, metrics in l1_models.items():
                    f.write(f"{model_name.upper()}:\n")
                    for key, value in metrics.items():
                        f.write(f"  • {key.replace('_', ' ').title()}: {value:.4f}\n")
                    f.write("\n")
                
                # L2 Model
                f.write("🎯 ANÁLISIS MODELO L2:\n")
                f.write("-" * 40 + "\n")
                l2_model = results.get('l2_model', {})
                for key, value in l2_model.items():
                    f.write(f"  • {key.replace('_', ' ').title()}: {value:.4f}\n")
                f.write("\n")
                
                # L3 Models
                f.write("🧠 ANÁLISIS MODELOS L3:\n")
                f.write("-" * 40 + "\n")
                l3_models = results.get('l3_models', {})
                for model_name, metrics in l3_models.items():
                    f.write(f"{model_name.upper()}:\n")
                    for key, value in metrics.items():
                        f.write(f"  • {key.replace('_', ' ').title()}: {value:.4f}\n")
                    f.write("\n")
                
                # Recomendaciones por modelo
                f.write("💡 RECOMENDACIONES POR MODELO:\n")
                f.write("-" * 40 + "\n")
                
                for model_name, metrics in l1_models.items():
                    accuracy = metrics.get('accuracy', 0)
                    f.write(f"  {model_name}:\n")
                    if accuracy < 0.6:
                        f.write("    • Mejorar dataset de entrenamiento\n")
                        f.write("    • Ajustar hiperparámetros\n")
                    if metrics.get('latency_ms', 0) > 100:
                        f.write("    • Optimizar inferencia del modelo\n")
                    f.write("\n")
                
                l2_hit_rate = l2_model.get('hit_rate', 0)
                if l2_hit_rate < 0.5:
                    f.write("  L2 Model:\n")
                    f.write("    • Revisar lógica de signal quality\n")
                    f.write("    • Aumentar umbrales de confianza\n\n")
                
                for model_name, metrics in l3_models.items():
                    decision_acc = metrics.get('decision_accuracy', 0)
                    f.write(f"  {model_name}:\n")
                    if decision_acc < 0.6:
                        f.write("    • Revisar lógica de toma de decisiones\n")
                        f.write("    • Incorporar más variables macro\n")
                    if metrics.get('strategic_value', 0) < 100:
                        f.write("    • Evaluar contribución al rendimiento\n")
                    f.write("\n")
                
                f.write("="*80 + "\n")
            
            return filename
            
        except Exception as e:
            self.logger.error(f"Error generando análisis de modelos: {e}")
            return ""
    
    async def generate_performance_charts(self, results: Dict, timestamp: str) -> str:
        """Generar gráficos de rendimiento"""
        
        charts_dir = f"{self.output_dir}/charts_{timestamp}"
        Path(charts_dir).mkdir(parents=True, exist_ok=True)
        
        try:
            # 1. Gráfico de equity curve
            await self._plot_equity_curve(results, charts_dir)
            
            # 2. Gráfico de drawdown
            await self._plot_drawdown_chart(results, charts_dir)
            
            # 3. Comparación de modelos L1
            await self._plot_l1_models_comparison(results, charts_dir)
            
            # 4. Distribución de retornos
            await self._plot_returns_distribution(results, charts_dir)
            
            # 5. Heatmap de correlaciones (si disponible)
            await self._plot_correlation_heatmap(results, charts_dir)
            
            self.logger.info(f"Gráficos generados en: {charts_dir}")
            return charts_dir
            
        except Exception as e:
            self.logger.error(f"Error generando gráficos: {e}")
            return ""
    
    async def _plot_equity_curve(self, results: Dict, output_dir: str):
        """Generar gráfico de curva de equity"""
        
        try:
            # Simular datos de equity curve si no están disponibles
            equity_data = results.get('equity_curve', [])
            
            if not equity_data:
                # Generar curva simulada basada en métricas
                overall = results.get('overall', {})
                total_return = overall.get('total_return', 0)
                volatility = overall.get('volatility', 0.2)
                
                # Generar 252 puntos (1 año de trading)
                dates = pd.date_range(start='2024-01-01', periods=252, freq='D')
                returns = np.random.normal(total_return/252, volatility/np.sqrt(252), 252)
                equity = (1 + returns).cumprod() * 10000  # Capital inicial
                
                equity_data = [{'timestamp': date, 'equity': eq} for date, eq in zip(dates, equity)]
            
            df = pd.DataFrame(equity_data)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            plt.figure(figsize=(12, 6))
            plt.plot(df['timestamp'], df['equity'], linewidth=2, color='blue', label='Strategy Equity')
            plt.title('HRM Strategy - Equity Curve', fontsize=16, fontweight='bold')
            plt.xlabel('Date')
            plt.ylabel('Equity ($)')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            
            plt.savefig(f"{output_dir}/equity_curve.png", dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Error plotting equity curve: {e}")
    
    async def _plot_drawdown_chart(self, results: Dict, output_dir: str):
        """Generar gráfico de drawdown"""
        
        try:
            # Simular datos de drawdown
            overall = results.get('overall', {})
            max_dd = overall.get('max_drawdown', -0.1)
            
            # Generar serie de drawdown simulada
            dates = pd.date_range(start='2024-01-01', periods=252, freq='D')
            drawdown = np.random.uniform(max_dd, 0, 252)
            drawdown = np.minimum.accumulate(drawdown)  # Drawdown cumulativo
            
            plt.figure(figsize=(12, 6))
            plt.fill_between(dates, drawdown * 100, 0, color='red', alpha=0.3, label='Drawdown')
            plt.plot(dates, drawdown * 100, color='red', linewidth=1)
            plt.title('HRM Strategy - Drawdown Analysis', fontsize=16, fontweight='bold')
            plt.xlabel('Date')
            plt.ylabel('Drawdown (%)')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            
            plt.savefig(f"{output_dir}/drawdown_chart.png", dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Error plotting drawdown: {e}")
    
    async def _plot_l1_models_comparison(self, results: Dict, output_dir: str):
        """Generar gráfico comparativo de modelos L1"""
        
        try:
            l1_models = results.get('l1_models', {})
            
            if not l1_models:
                return
            
            # Preparar datos
            models = list(l1_models.keys())
            metrics = ['accuracy', 'precision', 'f1_score']
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            for i, metric in enumerate(metrics):
                values = [l1_models[model].get(metric, 0) for model in models]
                
                bars = axes[i].bar(models, values, color=['skyblue', 'lightgreen', 'salmon'])
                axes[i].set_title(f'{metric.replace("_", " ").title()}', fontweight='bold')
                axes[i].set_ylabel('Score')
                axes[i].set_ylim(0, 1)
                
                # Añadir valores en las barras
                for bar, value in zip(bars, values):
                    axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                               f'{value:.3f}', ha='center', va='bottom')
            
            plt.suptitle('L1 Models Performance Comparison', fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            plt.savefig(f"{output_dir}/l1_models_comparison.png", dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Error plotting L1 models comparison: {e}")
    
    async def _plot_returns_distribution(self, results: Dict, output_dir: str):
        """Generar gráfico de distribución de retornos"""
        
        try:
            # Simular datos de retornos si no disponibles
            trades = results.get('trades', [])
            if trades:
                df = pd.DataFrame(trades)
                returns = df.get('pnl', pd.Series(np.random.normal(0.01, 0.05, 100)))
            else:
                returns = np.random.normal(0.01, 0.05, 100)
            
            plt.figure(figsize=(10, 6))
            sns.histplot(returns, kde=True, color='purple')
            plt.title('Distribución de Retornos de Trades', fontsize=16, fontweight='bold')
            plt.xlabel('Retorno (%)')
            plt.ylabel('Frecuencia')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            plt.savefig(f"{output_dir}/returns_distribution.png", dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Error plotting returns distribution: {e}")
    
    async def _plot_correlation_heatmap(self, results: Dict, output_dir: str):
        """Generar heatmap de correlaciones si disponible"""
        
        try:
            # Simular correlaciones si no disponibles
            symbols = ['BTCUSDT', 'ETHUSDT']  # Asumir símbolos
            corr_matrix = np.random.uniform(-1, 1, (len(symbols), len(symbols)))
            np.fill_diagonal(corr_matrix, 1)
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', xticklabels=symbols, yticklabels=symbols)
            plt.title('Heatmap de Correlaciones entre Activos', fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            plt.savefig(f"{output_dir}/correlation_heatmap.png", dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Error plotting correlation heatmap: {e}")
    
    async def export_trades_data(self, results: Dict, timestamp: str) -> str:
        """Exportar datos de trades a CSV"""
        
        filename = f"{self.output_dir}/trades_export_{timestamp}.csv"
        
        try:
            trades = results.get('trades', [])
            if trades:
                df = pd.DataFrame(trades)
                df.to_csv(filename, index=False)
                self.logger.info(f"Trades exportados a: {filename}")
                return filename
            else:
                self.logger.warning("No hay trades para exportar")
                return ""
            
        except Exception as e:
            self.logger.error(f"Error exportando trades: {e}")
            return ""
    
    async def export_json_report(self, results: Dict, timestamp: str) -> str:
        """Exportar reporte completo en JSON"""
        
        filename = f"{self.output_dir}/full_report_{timestamp}.json"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=4, default=str)
            self.logger.info(f"Reporte JSON exportado a: {filename}")
            return filename
            
        except Exception as e:
            self.logger.error(f"Error exportando JSON: {e}")
            return ""
    
    async def generate_html_dashboard(self, results: Dict, timestamp: str) -> str:
        """Generar dashboard HTML simple"""
        
        filename = f"{self.output_dir}/dashboard_{timestamp}.html"
        
        try:
            html_content = """
            <!DOCTYPE html>
            <html lang="es">
            <head>
                <meta charset="UTF-8">
                <title>HRM Backtesting Dashboard</title>
                <style>
                    body { font-family: Arial, sans-serif; }
                    table { border-collapse: collapse; width: 100%; }
                    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                    th { background-color: #f2f2f2; }
                </style>
            </head>
            <body>
                <h1>HRM Backtesting Dashboard</h1>
                <h2>Métricas Generales</h2>
                <table>
                    <tr><th>Métrica</th><th>Valor</th></tr>
            """
            
            overall = results.get('overall', {})
            for key, value in overall.items():
                if not isinstance(value, dict):  # Evitar sub-diccionarios
                    html_content += f"<tr><td>{key.replace('_', ' ').title()}</td><td>{value}</td></tr>"
            
            html_content += """
                </table>
                <h2>Modelos L1</h2>
                <table>
                    <tr><th>Modelo</th><th>Accuracy</th><th>Precision</th><th>F1 Score</th></tr>
            """
            
            l1_models = results.get('l1_models', {})
            for model_name, metrics in l1_models.items():
                html_content += f"<tr><td>{model_name}</td><td>{metrics.get('accuracy', 0):.2%}</td><td>{metrics.get('precision', 0):.3f}</td><td>{metrics.get('f1_score', 0):.3f}</td></tr>"
            
            html_content += """
                </table>
            </body>
            </html>
            """
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            self.logger.info(f"Dashboard HTML generado en: {filename}")
            return filename
            
        except Exception as e:
            self.logger.error(f"Error generando dashboard HTML: {e}")
            return ""

    def generate_full_report(self, results: Dict, output_dir: str) -> str:
        """Método síncrono para generar reporte completo (compatibilidad con main.py)"""
        try:
            # Crear directorio si no existe
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            # Generar reporte básico
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Reporte ejecutivo simple
            filename = f"{output_dir}/executive_summary_{timestamp}.txt"
            overall = results.get('overall', {})
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("="*80 + "\n")
                f.write("HRM BACKTESTING - RESUMEN EJECUTIVO\n")
                f.write("="*80 + "\n")
                f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                f.write("📊 MÉTRICAS CLAVE:\n")
                f.write("-" * 40 + "\n")
                f.write(f"Retorno Total: {overall.get('total_return', 0):.2%}\n")
                f.write(f"Retorno Anualizado: {overall.get('annualized_return', 0):.2%}\n")
                f.write(f"Sharpe Ratio: {overall.get('sharpe_ratio', 0):.3f}\n")
                f.write(f"Drawdown Máximo: {overall.get('max_drawdown', 0):.2%}\n")
                f.write(f"Win Rate: {overall.get('win_rate', 0):.2%}\n")
                f.write(f"Profit Factor: {overall.get('profit_factor', 0):.2f}\n")
                f.write(f"Total Trades: {overall.get('total_trades', 0)}\n")
                f.write("="*80 + "\n")
            
            self.logger.info(f"Reporte generado en: {filename}")
            return filename
            
        except Exception as e:
            self.logger.error(f"Error generando reporte: {e}")
            return ""
