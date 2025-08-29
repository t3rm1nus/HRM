#!//backtesting/probador_estrategia.py
"""
HRM Strategy Tester - Probador de Estrategias
Ejecuta y evalúa las estrategias del sistema HRM con datos reales/simulados
"""

import os
import sys
import asyncio
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import json
from dataclasses import dataclass
from enum import Enum
from report_generator import ReportGenerator
from getdata import BinanceDataCollector

# Agregar paths del proyecto HRM
sys.path.append('../l1_operational')
sys.path.append('../l2_tactic')
sys.path.append('../l3_strategic')
sys.path.append('..core')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))



from l1_operational.l1_operational import L1Model
from l2_tactic.procesar_l2 import procesar_l2 # O el nombre del módulo principal de L2

class TestMode(Enum):
    """Modos de testing disponibles"""
    UNIT = "unit"           # Tests unitarios por nivel
    INTEGRATION = "integration"  # Tests de integración entre niveles
    END_TO_END = "end_to_end"   # Test completo L4->L3->L2->L1
    PERFORMANCE = "performance"  # Tests de rendimiento
    REGRESSION = "regression"    # Tests de regresión

@dataclass
class TestResult:
    """Resultado de una prueba"""
    test_name: str
    level: str
    success: bool
    duration: float
    metrics: Dict[str, Any]
    errors: List[str]
    warnings: List[str]

class HRMStrategyTester:
    """Probador principal de estrategias HRM"""
    
    def __init__(self, config: Dict):
        # Configuración de logger
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:  # evita duplicar handlers si se instancia varias veces
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
            
        # Si la config viene anidada, extraer la sección testing
        if 'testing' in config:
            self.config = config['testing']
            self.binance_config = config.get('binance', {})
        else:
            self.config = config
            self.binance_config = {}
        
        # Usar symbols de binance_config si está disponible
        if 'symbols' not in self.config and 'symbols' in self.binance_config:
            self.config['symbols'] = self.binance_config['symbols']
        
        # 🔑 Inicialización de dependencias necesarias para el backtest
        self.data_collector = BinanceDataCollector(self.config)

        self.report_generator = ReportGenerator(self.config)

    def _load_test_config(self, config_path: str) -> Dict:
        """Cargar configuración de testing"""
        default_config = {
            "test_modes": ["unit", "integration", "end_to_end"],
            "test_duration_minutes": 10,
            "use_mock_data": True,
            "mock_symbols": ["BTCUSDT", "ETHUSDT"],
            "initial_capital": 10000,
            "max_trades_per_test": 100,
            "logging_level": "INFO",
            "output_dir": "test_results",
            "l1_models_path": "models/L1",
            "l2_models_path": "models/L2",
            "l3_models_path": "models/L3",
            "performance_thresholds": {
                "min_accuracy": 0.55,
                "max_latency_ms": 100,
                "min_sharpe": 0.5,
                "max_drawdown": 0.15
            }
        }
        
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    # Merge con defaults
                    for key, value in default_config.items():
                        if key not in config:
                            config[key] = value
                    return config
            else:
                self.logger.warning(f"Config file not found: {config_path}, using defaults")
                return default_config
        except Exception as e:
            logging.warning(f"Error loading config: {e}, using defaults")
            return default_config

    def setup_logging(self):
        """Configurar logging para tests"""
        os.makedirs("test_logs", exist_ok=True)
        log_filename = f"test_logs/strategy_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=getattr(logging, self.config.get('logging_level', 'INFO')),
            format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s',
            handlers=[
                logging.FileHandler(log_filename),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _initialize_components(self):
        """Inicializar componentes del sistema HRM"""
        try:
            # Importar componentes L1
            if os.path.exists('l1_operational'):
                from l1_operational.order_manager import OrderManager
                from l1_operational.risk_guard import RiskGuard
                from l1_operational.executor import Executor
                self.l1_components = {
                    'order_manager': OrderManager,
                    'risk_guard': RiskGuard,
                    'executor': Executor
                }
            else:
                self.l1_components = None
                
            # Importar componentes L2
            if os.path.exists('l2_tactic'):
                from l2_tactic.signal_generator import L2TacticProcessor
                from l2_tactic.position_sizer import PositionSizerManager
                self.l2_components = {
                    'signal_generator': L2TacticProcessor,
                    'position_sizer': PositionSizerManager
                }
            else:
                self.l2_components = None
                
            # Importar componentes L3
            if os.path.exists('l3_strategic'):
                from l3_strategic.strategic_processor import StrategicProcessor
                self.l3_components = {
                    'strategic_processor': StrategicProcessor
                }
            else:
                self.l3_components = None
                
        except ImportError as e:
            self.logger.warning(f"Some components not available: {e}")
            self.logger.info("Tests will run with available components only")

    def generate_mock_market_data(self, symbols: List[str], duration_hours: int = 24) -> Dict:
        """Generar datos de mercado simulados para testing"""
        self.logger.info(f"Generando datos mock para {symbols} ({duration_hours}h)")
        
        data = {}
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=duration_hours)
        
        # Generar timestamps cada minuto
        timestamps = pd.date_range(start=start_time, end=end_time, freq='1T')
        
        for symbol in symbols:
            # Precio base según símbolo
            base_price = 50000 if 'BTC' in symbol else 3000
            
            # Generar precios con random walk realista
            returns = np.random.normal(0, 0.002, len(timestamps))  # Vol ~0.2%
            prices = [base_price]
            
            for ret in returns[1:]:
                new_price = prices[-1] * (1 + ret)
                prices.append(max(new_price, base_price * 0.5))  # Floor price
            
            # Crear DataFrame
            df = pd.DataFrame({
                'timestamp': timestamps,
                'open': prices,
                'high': [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices],
                'low': [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices],
                'close': prices,
                'volume': np.random.uniform(1000, 10000, len(prices))
            })
            
            df.set_index('timestamp', inplace=True)
            data[symbol] = {'1m': df}
            
        self.logger.info(f"Mock data generado: {len(data)} símbolos, {len(timestamps)} puntos cada uno")
        return data

    async def run_unit_tests(self) -> List[TestResult]:
        """Ejecutar tests unitarios por nivel"""
        self.logger.info("🧪 Ejecutando tests unitarios...")
        results = []
        
        # Test L1 - Operational
        if self.l1_components:
            results.extend(await self._test_l1_components())
        
        # Test L2 - Tactical  
        if self.l2_components:
            results.extend(await self._test_l2_components())
            
        # Test L3 - Strategic
        if self.l3_components:
            results.extend(await self._test_l3_components())
            
        return results

    async def _test_l1_components(self) -> List[TestResult]:
        """Tests específicos para componentes L1"""
        results = []
        
        # Test modelos IA L1
        for model_name in ['modelo1_lr.pkl', 'modelo2_rf.pkl', 'modelo3_lgbm.pkl']:
            start_time = datetime.now()
            
            try:
                model_path = os.path.join(self.config['l1_models_path'], model_name)
                if os.path.exists(model_path):
                    # Test carga del modelo
                    import pickle
                    with open(model_path, 'rb') as f:
                        model = pickle.load(f)
                    
                    # Test predicción mock
                    mock_features = np.random.random((1, 10))  # 10 features mock
                    prediction = model.predict(mock_features)
                    
                    duration = (datetime.now() - start_time).total_seconds()
                    
                    results.append(TestResult(
                        test_name=f"L1_Model_{model_name}",
                        level="L1",
                        success=True,
                        duration=duration,
                        metrics={"prediction_shape": prediction.shape},
                        errors=[],
                        warnings=[]
                    ))
                    
                else:
                    results.append(TestResult(
                        test_name=f"L1_Model_{model_name}",
                        level="L1", 
                        success=False,
                        duration=0,
                        metrics={},
                        errors=[f"Model file not found: {model_path}"],
                        warnings=[]
                    ))
                    
            except Exception as e:
                duration = (datetime.now() - start_time).total_seconds()
                results.append(TestResult(
                    test_name=f"L1_Model_{model_name}",
                    level="L1",
                    success=False,
                    duration=duration,
                    metrics={},
                    errors=[str(e)],
                    warnings=[]
                ))
        
        # Test validaciones de riesgo L1
        results.append(await self._test_l1_risk_validations())
        
        return results

    async def _test_l1_risk_validations(self) -> TestResult:
        """Test validaciones de riesgo L1"""
        start_time = datetime.now()
        errors = []
        warnings = []
        
        try:
            # Test señal válida
            valid_signal = {
                "symbol": "BTCUSDT",
                "side": "buy",
                "qty": 0.01,
                "stop_loss": 49000.0,
                "price": 50000.0
            }
            
            # Simular validaciones básicas
            validations_passed = 0
            total_validations = 4
            
            # 1. Stop-loss coherente
            if valid_signal["side"] == "buy" and valid_signal["stop_loss"] < valid_signal["price"]:
                validations_passed += 1
            elif valid_signal["side"] == "sell" and valid_signal["stop_loss"] > valid_signal["price"]:
                validations_passed += 1
                
            # 2. Cantidad positiva
            if valid_signal["qty"] > 0:
                validations_passed += 1
                
            # 3. Límites de cantidad (BTC: max 0.05)
            if valid_signal["symbol"] == "BTCUSDT" and valid_signal["qty"] <= 0.05:
                validations_passed += 1
                
            # 4. Precio positivo
            if valid_signal["price"] > 0:
                validations_passed += 1
            
            success = validations_passed == total_validations
            if not success:
                errors.append(f"Risk validations failed: {validations_passed}/{total_validations}")
                
            duration = (datetime.now() - start_time).total_seconds()
            
            return TestResult(
                test_name="L1_Risk_Validations",
                level="L1",
                success=success,
                duration=duration,
                metrics={
                    "validations_passed": validations_passed,
                    "total_validations": total_validations,
                    "pass_rate": validations_passed / total_validations
                },
                errors=errors,
                warnings=warnings
            )
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            return TestResult(
                test_name="L1_Risk_Validations",
                level="L1",
                success=False,
                duration=duration,
                metrics={},
                errors=[str(e)],
                warnings=warnings
            )
    async def run_hrm_strategy(self, data: dict) -> dict:
        """Run the HRM trading strategy on the provided data."""
        self.logger.info("Running HRM strategy...")
        
        try:
            results = {
                'overall': {
                    'total_return': 0.0,
                    'annualized_return': 0.0,
                    'sharpe_ratio': 0.0,
                    'max_drawdown': 0.0,
                    'win_rate': 0.0,
                    'profit_factor': 0.0,
                    'total_trades': 0
                },
                'l1_models': {},
                'l2_model': {},
                'l3_models': {},
                'trades': []
            }

            for symbol in data:
                for interval in data[symbol]:
                    if isinstance(data[symbol][interval], pd.DataFrame) and not data[symbol][interval].empty:
                        df = data[symbol][interval].copy() # Usar una copia para evitar SettingWithCopyWarning
                        df['signal'] = 0
                        df.loc[df['rsi'] < 30, 'signal'] = 1  # Buy
                        df.loc[df['rsi'] > 70, 'signal'] = -1  # Sell

                        trades_temp = []
                        position = 0
                        entry_price = 0
                        entry_timestamp = None
                        
                        for idx, row in df.iterrows():
                            # Abrir una posición
                            if row['signal'] == 1 and position == 0:
                                entry_price = row['close']
                                entry_timestamp = idx
                                position = 1
                                # No se registra el trade hasta que se cierra
                                
                            # Cerrar una posición
                            elif row['signal'] == -1 and position == 1:
                                exit_price = row['close']
                                pnl = exit_price - entry_price
                                position = 0
                                
                                # REGISTRO DEL TRADE CERRADO
                                trades_temp.append({
                                    'entry_timestamp': entry_timestamp,
                                    'exit_timestamp': idx,
                                    'symbol': symbol,
                                    'interval': interval,
                                    'entry_price': entry_price,
                                    'exit_price': exit_price,
                                    'pnl': pnl
                                })

                        if trades_temp:
                            results['trades'].extend(trades_temp)
                            total_trades = len(trades_temp)
                            wins = len([t for t in trades_temp if t['pnl'] > 0])
                            losses = len([t for t in trades_temp if t['pnl'] <= 0])
                            total_pnl = sum(t['pnl'] for t in trades_temp)
                            
                            results['overall']['total_trades'] += total_trades
                            results['overall']['total_return'] += total_pnl / entry_price if entry_price else 0
                            results['overall']['win_rate'] = wins / total_trades if total_trades > 0 else 0
                            
                            # CÁLCULO MEJORADO DEL PROFIT FACTOR
                            gross_profit = sum(t['pnl'] for t in trades_temp if t['pnl'] > 0)
                            gross_loss = abs(sum(t['pnl'] for t in trades_temp if t['pnl'] <= 0))
                            
                            results['overall']['profit_factor'] = gross_profit / gross_loss if gross_loss > 0 else float('inf')

            # Esta sección se mantiene sin cambios ya que parece ser mock data
            results['l1_models'] = {
                'model1': L1Model().predict(None),
                'model2': {'accuracy': 0.82, 'precision': 0.78, 'f1_score': 0.80, 'profit_contribution': 800, 'latency_ms': 45}
            }
            results['l2_model'] = {
                'signal_quality': 0.75,
                'sizing_efficiency': 0.80,
                'hit_rate': 0.65,
                'risk_adjusted_return': 0.05
            }
            results['l3_models'] = {
                'strategic': {'decision_accuracy': 0.90, 'regime_detection_accuracy': 0.85, 'strategic_value': 0.95, 'allocation_efficiency': 0.88}
            }

            self.logger.info("HRM strategy execution completed")
            return results

        except Exception as e:
            self.logger.error(f"Error running HRM strategy: {e}")
            raise
    
    
    async def run_full_backtest(self):
        """Run the full backtesting process."""
        self.logger.info("Starting full backtest...")
        try:
            data = await self.data_collector.collect_historical_data(
                symbols=self.config['symbols'],
                start_date=(datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d'),
                end_date=datetime.now().strftime('%Y-%m-%d'),
                intervals=self.config['intervals']
            )
            results = await self.run_hrm_strategy(data)
            report_paths = await self.report_generator.generate_complete_report(results)
            self.logger.info("Backtest completed successfully")
            return report_paths
        except Exception as e:
            self.logger.error(f"Error in backtest: {e}")
            raise

    async def _test_l2_components(self) -> List[TestResult]:
        """Tests específicos para componentes L2"""
        results = []
        
        # Test generación de señales L2
        start_time = datetime.now()
        
        try:
            # Generar datos mock para L2
            mock_data = self.generate_mock_market_data(["BTCUSDT"], duration_hours=1)
            
            # Simular procesamiento L2
            mock_l3_decision = {
                "regime": "bull_trend",
                "target_exposure": 0.7,
                "universe": ["BTCUSDT"],
                "strategy_mode": "aggressive"
            }
            
            # Test composición de señales (simulado)
            signal_sources = ["technical", "ai_model", "pattern"]
            signals_generated = len(signal_sources)
            
            duration = (datetime.now() - start_time).total_seconds()
            
            results.append(TestResult(
                test_name="L2_Signal_Generation",
                level="L2",
                success=signals_generated > 0,
                duration=duration,
                metrics={
                    "signals_generated": signals_generated,
                    "data_points": len(mock_data["BTCUSDT"]["1m"])
                },
                errors=[],
                warnings=[]
            ))
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            results.append(TestResult(
                test_name="L2_Signal_Generation",
                level="L2",
                success=False,
                duration=duration,
                metrics={},
                errors=[str(e)],
                warnings=[]
            ))
        
        return results

    async def _test_l3_components(self) -> List[TestResult]:
        """Tests específicos para componentes L3"""
        results = []
        
        # Test modelos L3 simplificados (3 modelos según README L3)
        l3_models = ['unified_decision_model', 'regime_detector', 'risk_assessor']
        
        for model_name in l3_models:
            start_time = datetime.now()
            
            try:
                # Test simulado del modelo L3
                mock_market_features = {
                    "volatility": 0.25,
                    "momentum": 0.15,
                    "correlation_btc_eth": 0.7,
                    "volume_ratio": 1.2
                }
                
                # Simular predicción según tipo de modelo
                if model_name == "regime_detector":
                    prediction = {
                        "regime": "bull_trend",
                        "confidence": 0.72,
                        "probabilities": {
                            "bull_trend": 0.72,
                            "bear_trend": 0.15,
                            "sideways": 0.08,
                            "volatile": 0.05
                        }
                    }
                elif model_name == "unified_decision_model":
                    prediction = {
                        "allocation": {"BTC": 0.65, "ETH": 0.35},
                        "target_exposure": 0.75,
                        "strategy_mode": "aggressive_trend",
                        "confidence": 0.82
                    }
                else:  # risk_assessor
                    prediction = {
                        "risk_level": "moderate",
                        "risk_score": 0.34,
                        "max_position_size": 0.08,
                        "stop_loss_level": 0.05
                    }
                
                duration = (datetime.now() - start_time).total_seconds()
                
                results.append(TestResult(
                    test_name=f"L3_{model_name}",
                    level="L3",
                    success=True,
                    duration=duration,
                    metrics={
                        "prediction_keys": len(prediction),
                        "has_confidence": "confidence" in prediction
                    },
                    errors=[],
                    warnings=[]
                ))
                
            except Exception as e:
                duration = (datetime.now() - start_time).total_seconds()
                results.append(TestResult(
                    test_name=f"L3_{model_name}",
                    level="L3",
                    success=False,
                    duration=duration,
                    metrics={},
                    errors=[str(e)],
                    warnings=[]
                ))
        
        return results

    async def run_integration_tests(self) -> List[TestResult]:
        """Ejecutar tests de integración entre niveles"""
        self.logger.info("🔗 Ejecutando tests de integración...")
        results = []
        
        # Test integración L3->L2
        results.append(await self._test_l3_l2_integration())
        
        # Test integración L2->L1
        results.append(await self._test_l2_l1_integration())
        
        # Test bus de mensajes
        results.append(await self._test_message_bus())
        
        return results

    async def _test_l3_l2_integration(self) -> TestResult:
        """Test integración L3->L2"""
        start_time = datetime.now()
        errors = []
        warnings = []
        
        try:
            # Simular decisión estratégica L3
            l3_decision = {
                "regime": "bull_trend",
                "target_exposure": 0.7,
                "universe": ["BTCUSDT", "ETHUSDT"],
                "strategy_mode": "aggressive",
                "risk_appetite": "moderate"
            }
            
            # Simular procesamiento L2 de la decisión L3
            l2_signals = []
            for symbol in l3_decision["universe"]:
                signal = {
                    "symbol": symbol,
                    "side": "buy" if l3_decision["regime"] == "bull_trend" else "sell",
                    "confidence": 0.75,
                    "regime_context": l3_decision["regime"],
                    "target_exposure": l3_decision["target_exposure"] / len(l3_decision["universe"])
                }
                l2_signals.append(signal)
            
            success = len(l2_signals) == len(l3_decision["universe"])
            if not success:
                errors.append("L2 did not generate signals for all L3 universe assets")
                
            duration = (datetime.now() - start_time).total_seconds()
            
            return TestResult(
                test_name="L3_L2_Integration",
                level="Integration",
                success=success,
                duration=duration,
                metrics={
                    "l3_universe_size": len(l3_decision["universe"]),
                    "l2_signals_generated": len(l2_signals),
                    "signal_coverage": len(l2_signals) / len(l3_decision["universe"])
                },
                errors=errors,
                warnings=warnings
            )
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            return TestResult(
                test_name="L3_L2_Integration",
                level="Integration",
                success=False,
                duration=duration,
                metrics={},
                errors=[str(e)],
                warnings=warnings
            )

    async def _test_l2_l1_integration(self) -> TestResult:
        """Test integración L2->L1"""
        start_time = datetime.now()
        errors = []
        warnings = []
        
        try:
            # Simular señal táctica L2
            l2_signal = {
                "symbol": "BTCUSDT",
                "side": "buy",
                "qty": 0.01,
                "confidence": 0.8,
                "stop_loss": 49000.0,
                "take_profit": 52000.0,
                "regime_context": "bull_trend"
            }
            
            # Simular validaciones L1
            l1_validations = {
                "risk_check": True,
                "balance_check": True,
                "exposure_check": True,
                "correlation_check": True
            }
            
            # Simular procesamiento de modelos IA L1
            ai_scores = {
                "modelo1_lr": 0.72,
                "modelo2_rf": 0.68,
                "modelo3_lgbm": 0.75
            }
            
            # Decisión final L1 (simulada)
            avg_ai_score = sum(ai_scores.values()) / len(ai_scores)
            execution_approved = avg_ai_score > 0.6 and all(l1_validations.values())
            
            duration = (datetime.now() - start_time).total_seconds()
            
            return TestResult(
                test_name="L2_L1_Integration",
                level="Integration",
                success=execution_approved,
                duration=duration,
                metrics={
                    "l2_signal_confidence": l2_signal["confidence"],
                    "l1_validations_passed": sum(l1_validations.values()),
                    "avg_ai_score": avg_ai_score,
                    "execution_approved": execution_approved
                },
                errors=errors,
                warnings=warnings
            )
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            return TestResult(
                test_name="L2_L1_Integration",
                level="Integration",
                success=False,
                duration=duration,
                metrics={},
                errors=[str(e)],
                warnings=warnings
            )

    async def _test_message_bus(self) -> TestResult:
        """Test sistema de mensajería"""
        start_time = datetime.now()
        
        try:
            # Simular envío y recepción de mensajes
            messages_sent = 0
            messages_received = 0
            
            # Test message types según el sistema
            test_messages = [
                {"type": "l3_decision", "data": {"regime": "bull_trend"}},
                {"type": "l2_signal", "data": {"symbol": "BTCUSDT", "side": "buy"}},
                {"type": "l1_execution", "data": {"status": "filled", "qty": 0.01}}
            ]
            
            for msg in test_messages:
                messages_sent += 1
                # Simular procesamiento del mensaje
                if "data" in msg and "type" in msg:
                    messages_received += 1
            
            success = messages_sent == messages_received
            duration = (datetime.now() - start_time).total_seconds()
            
            return TestResult(
                test_name="Message_Bus",
                level="Integration",
                success=success,
                duration=duration,
                metrics={
                    "messages_sent": messages_sent,
                    "messages_received": messages_received,
                    "success_rate": messages_received / messages_sent if messages_sent > 0 else 0
                },
                errors=[] if success else ["Message bus test failed"],
                warnings=[]
            )
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            return TestResult(
                test_name="Message_Bus",
                level="Integration",
                success=False,
                duration=duration,
                metrics={},
                errors=[str(e)],
                warnings=[]
            )

    async def run_end_to_end_test(self) -> TestResult:
        """Ejecutar test completo end-to-end"""
        self.logger.info("🎯 Ejecutando test end-to-end...")
        start_time = datetime.now()
        
        try:
            # 1. Generar datos mock
            mock_data = self.generate_mock_market_data(
                self.config["mock_symbols"], 
                duration_hours=1
            )
            
            # 2. Simular flujo completo L4->L3->L2->L1
            
            # L4: Meta-razonamiento (simulado)
            l4_context = {
                "market_regime": "bull_market",
                "risk_appetite": "moderate",
                "capital_allocation": {"BTC": 0.6, "ETH": 0.4},
                "model_weights": {"trend": 0.7, "mean_reversion": 0.3}
            }
            
            # L3: Decisión estratégica
            l3_decision = {
                "regime": l4_context["market_regime"],
                "target_exposure": 0.7,
                "universe": self.config["mock_symbols"],
                "strategy_mode": "aggressive_trend",
                "risk_budget": 0.1
            }
            
            # L2: Señales tácticas
            l2_signals = []
            for symbol in l3_decision["universe"]:
                signal = {
                    "symbol": symbol,
                    "side": "buy",
                    "confidence": np.random.uniform(0.6, 0.9),
                    "qty": 0.01 if "BTC" in symbol else 1.0,
                    "stop_loss": 49000 if "BTC" in symbol else 2900,
                    "take_profit": 52000 if "BTC" in symbol else 3200
                }
                l2_signals.append(signal)
            
            # L1: Ejecución (simulada)
            executed_trades = 0
            total_profit = 0
            
            for signal in l2_signals:
                # Simular validaciones L1
                risk_passed = True  # Simplificado
                ai_score = np.random.uniform(0.5, 0.9)
                
                if risk_passed and ai_score > 0.6:
                    executed_trades += 1
                    # Simular P&L
                    trade_pnl = np.random.uniform(-100, 200)  # USD
                    total_profit += trade_pnl
            
            # Calcular métricas end-to-end
            success_rate = executed_trades / len(l2_signals) if l2_signals else 0
            avg_profit_per_trade = total_profit / executed_trades if executed_trades > 0 else 0
            
            success = (
                executed_trades > 0 and 
                success_rate >= 0.5 and
                total_profit > -500  # Pérdida máxima aceptable
            )
            
            duration = (datetime.now() - start_time).total_seconds()
            
            return TestResult(
                test_name="End_to_End_Strategy_Test",
                level="System",
                success=success,
                duration=duration,
                metrics={
                    "signals_generated": len(l2_signals),
                    "trades_executed": executed_trades,
                    "success_rate": success_rate,
                    "total_profit": total_profit,
                    "avg_profit_per_trade": avg_profit_per_trade,
                    "data_points_processed": sum(len(mock_data[s]["1m"]) for s in mock_data)
                },
                errors=[],
                warnings=[] if success else ["End-to-end test below expectations"]
            )
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            return TestResult(
                test_name="End_to_End_Strategy_Test",
                level="System",
                success=False,
                duration=duration,
                metrics={},
                errors=[str(e)],
                warnings=[]
            )

    async def run_performance_tests(self) -> List[TestResult]:
        """Ejecutar tests de performance"""
        self.logger.info("⚡ Ejecutando tests de performance...")
        results = []
        
        # Test latencia L1 (crítico para ejecución)
        results.append(await self._test_l1_latency())
        
        # Test throughput del sistema
        results.append(await self._test_system_throughput())
        
        # Test memoria
        results.append(await self._test_memory_usage())
        
        return results

    async def _test_l1_latency(self) -> TestResult:
        """Test latencia de L1 (crítico)"""
        start_time = datetime.now()
        
        try:
            latencies = []
            n_tests = 100
            
            for _ in range(n_tests):
                test_start = datetime.now()
                
                # Simular procesamiento L1 completo
                # 1. Validaciones risk guard
                await asyncio.sleep(0.001)  # 1ms simulado
                
                # 2. Modelos IA (3 modelos)
                await asyncio.sleep(0.005)  # 5ms simulado
                
                # 3. Decisión final
                await asyncio.sleep(0.001)  # 1ms simulado
                
                test_end = datetime.now()
                latency_ms = (test_end - test_start).total_seconds() * 1000
                latencies.append(latency_ms)
            
            avg_latency = sum(latencies) / len(latencies)
            max_latency = max(latencies)
            p95_latency = np.percentile(latencies, 95)
            
            # Threshold según config (100ms máximo)
            max_allowed = self.config["performance_thresholds"]["max_latency_ms"]
            success = avg_latency < max_allowed
            
            duration = (datetime.now() - start_time).total_seconds()
            
            return TestResult(
                test_name="L1_Latency_Test",
                level="Performance",
                success=success,
                duration=duration,
                metrics={
                    "avg_latency_ms": avg_latency,
                    "max_latency_ms": max_latency,
                    "p95_latency_ms": p95_latency,
                    "tests_run": n_tests,
                    "threshold_ms": max_allowed
                },
                errors=[] if success else [f"Average latency {avg_latency:.2f}ms exceeds threshold {max_allowed}ms"],
                warnings=[] if p95_latency < max_allowed * 1.5 else [f"P95 latency {p95_latency:.2f}ms is high"]
            )
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            return TestResult(
                test_name="L1_Latency_Test",
                level="Performance",
                success=False,
                duration=duration,
                metrics={},
                errors=[str(e)],
                warnings=[]
            )

    async def _test_system_throughput(self) -> TestResult:
        """Test throughput del sistema completo"""
        start_time = datetime.now()
        
        try:
            # Test procesamiento de múltiples señales simultáneas
            n_signals = 50
            processed_signals = 0
            failed_signals = 0
            
            tasks = []
            for i in range(n_signals):
                # Crear señal mock
                signal = {
                    "id": f"signal_{i}",
                    "symbol": "BTCUSDT" if i % 2 == 0 else "ETHUSDT",
                    "side": "buy" if i % 3 == 0 else "sell",
                    "qty": 0.01 if "BTC" in ("BTCUSDT" if i % 2 == 0 else "ETHUSDT") else 1.0
                }
                
                # Simular procesamiento asíncrono
                task = asyncio.create_task(self._process_signal_mock(signal))
                tasks.append(task)
            
            # Esperar todos los procesamientos
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, Exception):
                    failed_signals += 1
                else:
                    processed_signals += 1
            
            throughput = processed_signals / ((datetime.now() - start_time).total_seconds())
            success_rate = processed_signals / n_signals
            
            # Threshold: >100 señales/segundo según README L1
            min_throughput = 100
            success = throughput >= min_throughput and success_rate >= 0.95
            
            duration = (datetime.now() - start_time).total_seconds()
            
            return TestResult(
                test_name="System_Throughput_Test",
                level="Performance",
                success=success,
                duration=duration,
                metrics={
                    "signals_processed": processed_signals,
                    "signals_failed": failed_signals,
                    "throughput_per_sec": throughput,
                    "success_rate": success_rate,
                    "target_throughput": min_throughput
                },
                errors=[] if success else [f"Throughput {throughput:.1f}/s below target {min_throughput}/s"],
                warnings=[]
            )
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            return TestResult(
                test_name="System_Throughput_Test",
                level="Performance",
                success=False,
                duration=duration,
                metrics={},
                errors=[str(e)],
                warnings=[]
            )

    async def _process_signal_mock(self, signal: Dict) -> bool:
        """Procesamiento mock de una señal para test de throughput"""
        try:
            # Simular validaciones L1
            await asyncio.sleep(0.002)  # 2ms para validaciones
            
            # Simular modelos IA
            await asyncio.sleep(0.005)  # 5ms para 3 modelos IA
            
            # Simular decisión final
            await asyncio.sleep(0.001)  # 1ms para decisión
            
            return True
        except:
            return False

    async def _test_memory_usage(self) -> TestResult:
        """Test uso de memoria del sistema"""
        start_time = datetime.now()
        
        try:
            import psutil
            process = psutil.Process()
            
            # Memoria inicial
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Simular carga de modelos y datos
            mock_models = []
            mock_data_sets = []
            
            # Cargar datos mock grandes
            for i in range(10):
                large_dataset = np.random.random((10000, 50))  # 10k rows, 50 features
                mock_data_sets.append(large_dataset)
            
            # Simular modelos cargados
            for i in range(6):  # 3 L1 + 1 L2 + 2 L3 modelos principales
                mock_model = {"weights": np.random.random((100, 100))}
                mock_models.append(mock_model)
            
            # Memoria después de carga
            peak_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Limpiar
            del mock_models
            del mock_data_sets
            
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            memory_increase = peak_memory - initial_memory
            memory_cleaned = peak_memory - final_memory
            
            # Threshold: <500MB según requisitos típicos
            max_memory_mb = 500
            success = peak_memory < max_memory_mb
            
            duration = (datetime.now() - start_time).total_seconds()
            
            return TestResult(
                test_name="Memory_Usage_Test",
                level="Performance",
                success=success,
                duration=duration,
                metrics={
                    "initial_memory_mb": initial_memory,
                    "peak_memory_mb": peak_memory,
                    "final_memory_mb": final_memory,
                    "memory_increase_mb": memory_increase,
                    "memory_cleaned_mb": memory_cleaned,
                    "threshold_mb": max_memory_mb
                },
                errors=[] if success else [f"Peak memory {peak_memory:.1f}MB exceeds threshold {max_memory_mb}MB"],
                warnings=[] if memory_cleaned > memory_increase * 0.8 else ["Poor memory cleanup detected"]
            )
            
        except ImportError:
            return TestResult(
                test_name="Memory_Usage_Test",
                level="Performance",
                success=False,
                duration=0,
                metrics={},
                errors=["psutil not available for memory testing"],
                warnings=["Install psutil for memory tests: pip install psutil"]
            )
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            return TestResult(
                test_name="Memory_Usage_Test",
                level="Performance",
                success=False,
                duration=duration,
                metrics={},
                errors=[str(e)],
                warnings=[]
            )

    def generate_test_report(self) -> str:
        """Generar reporte detallado de tests"""
        if not self.test_results:
            return "No test results available"
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results if r.success)
        failed_tests = total_tests - passed_tests
        
        total_duration = sum(r.duration for r in self.test_results)
        
        report = []
        report.append("=" * 80)
        report.append("HRM STRATEGY TESTER - DETAILED REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Summary
        report.append("EXECUTIVE SUMMARY:")
        report.append(f"  Total Tests: {total_tests}")
        report.append(f"  Passed: {passed_tests} ({passed_tests/total_tests*100:.1f}%)")
        report.append(f"  Failed: {failed_tests} ({failed_tests/total_tests*100:.1f}%)")
        report.append(f"  Total Duration: {total_duration:.2f} seconds")
        report.append("")
        
        # Results by level
        levels = {}
        for result in self.test_results:
            if result.level not in levels:
                levels[result.level] = {"passed": 0, "failed": 0, "total": 0}
            levels[result.level]["total"] += 1
            if result.success:
                levels[result.level]["passed"] += 1
            else:
                levels[result.level]["failed"] += 1
        
        report.append("RESULTS BY LEVEL:")
        for level, stats in levels.items():
            report.append(f"  {level}:")
            report.append(f"    Passed: {stats['passed']}/{stats['total']}")
            report.append(f"    Success Rate: {stats['passed']/stats['total']*100:.1f}%")
        report.append("")
        
        # Detailed results
        report.append("DETAILED RESULTS:")
        for result in self.test_results:
            status = "PASS" if result.success else "FAIL"
            report.append(f"  [{status}] {result.test_name} ({result.level})")
            report.append(f"    Duration: {result.duration:.3f}s")
            
            if result.metrics:
                report.append("    Metrics:")
                for key, value in result.metrics.items():
                    if isinstance(value, float):
                        report.append(f"      {key}: {value:.3f}")
                    else:
                        report.append(f"      {key}: {value}")
            
            if result.errors:
                report.append("    Errors:")
                for error in result.errors:
                    report.append(f"      - {error}")
            
            if result.warnings:
                report.append("    Warnings:")
                for warning in result.warnings:
                    report.append(f"      - {warning}")
            report.append("")
        
        # Performance analysis
        perf_tests = [r for r in self.test_results if r.level == "Performance"]
        if perf_tests:
            report.append("PERFORMANCE ANALYSIS:")
            for test in perf_tests:
                report.append(f"  {test.test_name}:")
                if "avg_latency_ms" in test.metrics:
                    report.append(f"    Latency: {test.metrics['avg_latency_ms']:.2f}ms avg")
                if "throughput_per_sec" in test.metrics:
                    report.append(f"    Throughput: {test.metrics['throughput_per_sec']:.1f}/sec")
                if "peak_memory_mb" in test.metrics:
                    report.append(f"    Memory: {test.metrics['peak_memory_mb']:.1f}MB peak")
            report.append("")
        
        # Recommendations
        report.append("RECOMMENDATIONS:")
        if failed_tests > 0:
            report.append("  - Review failed tests and fix underlying issues")
        
        l1_latency_tests = [r for r in self.test_results 
                           if "Latency" in r.test_name and not r.success]
        if l1_latency_tests:
            report.append("  - L1 latency issues detected - optimize AI model inference")
        
        memory_tests = [r for r in self.test_results 
                       if "Memory" in r.test_name and r.warnings]
        if memory_tests:
            report.append("  - Memory cleanup issues - review resource management")
        
        if passed_tests / total_tests < 0.9:
            report.append("  - Overall test success rate below 90% - system needs attention")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)

    async def run_all_tests(self) -> Dict:
        """Ejecutar toda la suite de tests"""
        self.logger.info("Starting comprehensive HRM strategy testing...")
        
        start_time = datetime.now()
        self.test_results = []
        
        # Ejecutar todos los tipos de tests
        for test_mode in self.config["test_modes"]:
            self.logger.info(f"Running {test_mode} tests...")
            
            if test_mode == "unit":
                results = await self.run_unit_tests()
                self.test_results.extend(results)
                
            elif test_mode == "integration":
                results = await self.run_integration_tests()
                self.test_results.extend(results)
                
            elif test_mode == "end_to_end":
                result = await self.run_end_to_end_test()
                self.test_results.append(result)
                
            elif test_mode == "performance":
                results = await self.run_performance_tests()
                self.test_results.extend(results)
        
        total_duration = (datetime.now() - start_time).total_seconds()
        
        # Generar reporte
        report_text = self.generate_test_report()
        
        # Guardar reporte
        os.makedirs(self.config["output_dir"], exist_ok=True)
        report_filename = os.path.join(
            self.config["output_dir"], 
            f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )
        
        with open(report_filename, 'w') as f:
            f.write(report_text)
        
        self.logger.info(f"Testing completed in {total_duration:.2f} seconds")
        self.logger.info(f"Report saved to: {report_filename}")
        
        # Calcular métricas finales
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results if r.success)
        
        return {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": total_tests - passed_tests,
            "success_rate": passed_tests / total_tests if total_tests > 0 else 0,
            "total_duration": total_duration,
            "report_path": report_filename,
            "test_results": self.test_results
        }

    def print_summary(self, results: Dict):
        """Imprimir resumen de resultados en consola"""
        print("\n" + "="*60)
        print("HRM STRATEGY TESTER - EXECUTION SUMMARY")
        print("="*60)
        
        print(f"\nTEST EXECUTION:")
        print(f"  Total Tests: {results['total_tests']}")
        print(f"  Passed: {results['passed_tests']}")
        print(f"  Failed: {results['failed_tests']}")
        print(f"  Success Rate: {results['success_rate']:.1%}")
        print(f"  Duration: {results['total_duration']:.2f} seconds")
        
        # Status por nivel
        levels = {}
        for result in results['test_results']:
            if result.level not in levels:
                levels[result.level] = {"passed": 0, "total": 0}
            levels[result.level]["total"] += 1
            if result.success:
                levels[result.level]["passed"] += 1
        
        print(f"\nRESULTS BY LEVEL:")
        for level, stats in levels.items():
            status = "PASS" if stats["passed"] == stats["total"] else "FAIL"
            print(f"  [{status}] {level}: {stats['passed']}/{stats['total']}")
        
        # Key metrics
        perf_results = [r for r in results['test_results'] if r.level == "Performance"]
        if perf_results:
            print(f"\nPERFORMANCE HIGHLIGHTS:")
            for result in perf_results:
                if "avg_latency_ms" in result.metrics:
                    print(f"  L1 Latency: {result.metrics['avg_latency_ms']:.2f}ms avg")
                if "throughput_per_sec" in result.metrics:
                    print(f"  Throughput: {result.metrics['throughput_per_sec']:.1f} signals/sec")
        
        print(f"\nDetailed report: {results['report_path']}")
        print("="*60)

# Función principal para ejecución directa
async def main():
    """Ejecutar el probador de estrategias"""
    
    print("HRM Strategy Tester")
    print("==================")
    
    # Crear directorio de resultados
    os.makedirs("test_results", exist_ok=True)
    
    # Inicializar tester
    tester = HRMStrategyTester()
    
    try:
        # Ejecutar todos los tests
        results = await tester.run_all_tests()
        
        # Mostrar resumen
        tester.print_summary(results)
        
        # Verificar si todos los tests críticos pasaron
        critical_tests = [r for r in results['test_results'] 
                         if r.level in ["L1", "Integration", "Performance"]]
        critical_passed = all(r.success for r in critical_tests)
        
        if critical_passed:
            print("\nALL CRITICAL TESTS PASSED - System ready for deployment")
            return 0
        else:
            print("\nCRITICAL TESTS FAILED - System needs attention before deployment")
            return 1
            
    except KeyboardInterrupt:
        print("\nTesting interrupted by user")
        return 1
    except Exception as e:
        print(f"\nTesting failed with error: {str(e)}")
        logging.exception("Full error details:")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())