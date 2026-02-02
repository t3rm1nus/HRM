#!/usr/bin/env python3
"""
Validaciones de seguridad para evitar trading real accidental.
Este script implementa múltiples capas de validación para garantizar
que el sistema nunca ejecute operaciones reales de forma accidental.
"""

import os
import sys
import asyncio
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

class SecurityValidator:
    """Validador de seguridad para prevenir trading real accidental."""
    
    def __init__(self):
        self.security_checks = []
        self.critical_failures = []
        self.warnings = []
        
    def add_security_check(self, name: str, check_func, critical: bool = True):
        """Añade una verificación de seguridad."""
        self.security_checks.append({
            'name': name,
            'check_func': check_func,
            'critical': critical
        })
    
    def run_all_checks(self) -> Dict[str, Any]:
        """Ejecuta todas las verificaciones de seguridad."""
        results = {
            'timestamp': datetime.utcnow().isoformat(),
            'total_checks': len(self.security_checks),
            'passed_checks': 0,
            'failed_checks': 0,
            'critical_failures': [],
            'warnings': [],
            'security_status': 'UNKNOWN',
            'safe_to_trade': False
        }
        
        for check in self.security_checks:
            try:
                check_result = check['check_func']()
                if check_result['status'] == 'PASS':
                    results['passed_checks'] += 1
                elif check_result['status'] == 'FAIL':
                    results['failed_checks'] += 1
                    if check['critical']:
                        results['critical_failures'].append({
                            'name': check['name'],
                            'reason': check_result['reason']
                        })
                    else:
                        results['warnings'].append({
                            'name': check['name'],
                            'reason': check_result['reason']
                        })
                        
            except Exception as e:
                results['failed_checks'] += 1
                if check['critical']:
                    results['critical_failures'].append({
                        'name': check['name'],
                        'reason': f"Error en validación: {str(e)}"
                    })
                else:
                    results['warnings'].append({
                        'name': check['name'],
                        'reason': f"Error en validación: {str(e)}"
                    })
        
        # Determinar estado de seguridad
        if results['critical_failures']:
            results['security_status'] = 'CRITICAL_FAILURE'
            results['safe_to_trade'] = False
        elif results['failed_checks'] > 0:
            results['security_status'] = 'WARNING'
            results['safe_to_trade'] = False
        else:
            results['security_status'] = 'SECURE'
            results['safe_to_trade'] = True
        
        return results

class SecurityChecks:
    """Clase con métodos de verificación de seguridad."""
    
    @staticmethod
    def check_binance_mode() -> Dict[str, str]:
        """Verifica que el modo Binance esté en PAPER."""
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
        
        binance_mode = os.getenv('BINANCE_MODE', '').upper()
        
        if binance_mode == 'PAPER':
            return {'status': 'PASS', 'reason': 'Modo PAPER detectado'}
        elif binance_mode == 'LIVE':
            return {'status': 'FAIL', 'reason': 'Modo LIVE detectado - RIESGO DE OPERACIONES REALES'}
        else:
            return {'status': 'FAIL', 'reason': f'Modo desconocido: {binance_mode}'}
    
    @staticmethod
    def check_testnet_enabled() -> Dict[str, str]:
        """Verifica que el testnet esté habilitado."""
        use_testnet = os.getenv('USE_TESTNET', '').lower()
        
        if use_testnet in ['true', '1', 'yes']:
            return {'status': 'PASS', 'reason': 'Testnet habilitado'}
        else:
            return {'status': 'FAIL', 'reason': f'Testnet deshabilitado: {use_testnet}'}
    
    @staticmethod
    def check_credentials_safety() -> Dict[str, str]:
        """Verifica que las credenciales sean seguras (no reales)."""
        api_key = os.getenv('BINANCE_API_KEY', '')
        api_secret = os.getenv('BINANCE_API_SECRET', '')
        
        # Verificar credenciales de ejemplo
        example_indicators = ['your_', 'example', 'test', 'demo']
        
        if any(indicator in api_key.lower() for indicator in example_indicators):
            return {'status': 'PASS', 'reason': 'Credenciales de ejemplo detectadas (seguro)'}
        
        if any(indicator in api_secret.lower() for indicator in example_indicators):
            return {'status': 'PASS', 'reason': 'Credenciales de ejemplo detectadas (seguro)'}
        
        # Verificar credenciales reales
        if api_key and api_secret:
            if len(api_key) >= 32 and len(api_secret) >= 32:
                return {'status': 'FAIL', 'reason': 'Credenciales reales detectadas - RIESGO DE OPERACIONES REALES'}
        
        return {'status': 'PASS', 'reason': 'Credenciales seguras o no configuradas'}
    
    @staticmethod
    def check_testnet_urls() -> Dict[str, str]:
        """Verifica que las URLs sean de testnet."""
        testnet_url = os.getenv('BINANCE_TESTNET_URL', '')
        
        if not testnet_url:
            return {'status': 'FAIL', 'reason': 'URL de testnet no configurada'}
        
        expected_domains = ['testnet.binance.vision', 'testnet.binance.com']
        
        if any(domain in testnet_url for domain in expected_domains):
            return {'status': 'PASS', 'reason': f'URL de testnet válida: {testnet_url}'}
        else:
            return {'status': 'FAIL', 'reason': f'URL no es de testnet: {testnet_url}'}
    
    @staticmethod
    def check_binance_client_mode() -> Dict[str, str]:
        """Verifica que el BinanceClient esté en modo testnet."""
        try:
            from l1_operational.binance_client import BinanceClient
            
            # Crear cliente para verificar configuración
            client = BinanceClient()
            
            if hasattr(client, 'use_testnet') and client.use_testnet:
                return {'status': 'PASS', 'reason': 'BinanceClient en modo testnet'}
            else:
                return {'status': 'FAIL', 'reason': 'BinanceClient no está en modo testnet'}
                
        except Exception as e:
            return {'status': 'FAIL', 'reason': f'Error verificando BinanceClient: {e}'}
    
    @staticmethod
    def check_order_manager_mode() -> Dict[str, str]:
        """Verifica que el OrderManager esté en modo paper."""
        try:
            from l1_operational.order_manager import OrderManager
            from l1_operational.binance_client import BinanceClient
            
            # Crear cliente y manager para verificar
            binance_client = BinanceClient()
            order_manager = OrderManager(binance_client=binance_client)
            
            if hasattr(order_manager, 'paper_mode') and order_manager.paper_mode:
                return {'status': 'PASS', 'reason': 'OrderManager en modo paper'}
            else:
                return {'status': 'FAIL', 'reason': 'OrderManager no está en modo paper'}
                
        except Exception as e:
            return {'status': 'FAIL', 'reason': f'Error verificando OrderManager: {e}'}
    
    @staticmethod
    def check_no_real_trading_enabled() -> Dict[str, str]:
        """Verifica que no haya trading real habilitado en ningún componente."""
        try:
            # Verificar que no haya credenciales reales activas
            api_key = os.getenv('BINANCE_API_KEY', '')
            api_secret = os.getenv('BINANCE_API_SECRET', '')
            
            if api_key and api_secret:
                # Verificar que no sean credenciales reales
                real_indicators = ['your_', 'example', 'test', 'demo']
                
                if not any(indicator in api_key.lower() for indicator in real_indicators):
                    if not any(indicator in api_secret.lower() for indicator in real_indicators):
                        if len(api_key) >= 32 and len(api_secret) >= 32:
                            return {'status': 'FAIL', 'reason': 'Credenciales reales activas detectadas'}
            
            return {'status': 'PASS', 'reason': 'No hay trading real habilitado'}
            
        except Exception as e:
            return {'status': 'FAIL', 'reason': f'Error verificando trading real: {e}'}
    
    @staticmethod
    def check_environment_isolation() -> Dict[str, str]:
        """Verifica que el entorno esté aislado para paper trading."""
        # Verificar variables de entorno críticas
        critical_vars = {
            'BINANCE_MODE': 'PAPER',
            'USE_TESTNET': 'true',
            'BINANCE_STRICT_TESTNET_MODE': 'true'
        }
        
        failed_checks = []
        
        for var, expected_value in critical_vars.items():
            actual_value = os.getenv(var, '').lower()
            if actual_value != expected_value.lower():
                failed_checks.append(f"{var}={actual_value} (esperado: {expected_value})")
        
        if failed_checks:
            return {'status': 'FAIL', 'reason': f'Variables críticas incorrectas: {", ".join(failed_checks)}'}
        else:
            return {'status': 'PASS', 'reason': 'Entorno aislado correctamente'}

def create_emergency_stop_procedure():
    """Crea un procedimiento de parada de emergencia."""
    
    emergency_procedure = """
🚨 PROCEDIMIENTO DE PARADA DE EMERGENCIA 🚨

Si se detecta una operación real accidental:

1. INMEDIATAMENTE:
   - Detener todos los procesos de trading
   - Desconectar del exchange
   - Notificar al equipo de seguridad

2. VERIFICAR:
   - Revisar todas las órdenes ejecutadas
   - Verificar balances afectados
   - Identificar la causa del fallo

3. DOCUMENTAR:
   - Registrar el incidente completo
   - Documentar pasos tomados
   - Crear reporte de lecciones aprendidas

4. CORREGIR:
   - Ajustar configuración de seguridad
   - Implementar parches si es necesario
   - Reforzar validaciones

5. PREVENIR:
   - Revisar procedimientos de seguridad
   - Capacitar al equipo
   - Mejorar sistemas de detección

⚠️  CONTACTO DE EMERGENCIA:
   - Equipo de Seguridad: [contacto]
   - Desarrollo: [contacto]
   - Gerencia: [contacto]
"""
    
    return emergency_procedure

def main():
    """Función principal de validación de seguridad."""
    print("🔒 VALIDACIÓN DE SEGURIDAD - PREVENCIÓN DE TRADING REAL ACCIDENTAL")
    print("=" * 70)
    
    # Crear validador de seguridad
    validator = SecurityValidator()
    
    # Añadir verificaciones críticas
    security_checks = SecurityChecks()
    
    validator.add_security_check(
        "Modo Binance", 
        security_checks.check_binance_mode,
        critical=True
    )
    
    validator.add_security_check(
        "Testnet Habilitado",
        security_checks.check_testnet_enabled,
        critical=True
    )
    
    validator.add_security_check(
        "Seguridad de Credenciales",
        security_checks.check_credentials_safety,
        critical=True
    )
    
    validator.add_security_check(
        "URLs de Testnet",
        security_checks.check_testnet_urls,
        critical=True
    )
    
    validator.add_security_check(
        "Modo BinanceClient",
        security_checks.check_binance_client_mode,
        critical=True
    )
    
    validator.add_security_check(
        "Modo OrderManager",
        security_checks.check_order_manager_mode,
        critical=True
    )
    
    validator.add_security_check(
        "Sin Trading Real",
        security_checks.check_no_real_trading_enabled,
        critical=True
    )
    
    validator.add_security_check(
        "Aislamiento de Entorno",
        security_checks.check_environment_isolation,
        critical=False
    )
    
    # Ejecutar validaciones
    print("\n🔍 EJECUTANDO VERIFICACIONES DE SEGURIDAD...")
    print("-" * 50)
    
    results = validator.run_all_checks()
    
    # Mostrar resultados
    print(f"\n📊 RESULTADOS DE SEGURIDAD:")
    print(f"   - Total de verificaciones: {results['total_checks']}")
    print(f"   - Verificaciones exitosas: {results['passed_checks']}")
    print(f"   - Verificaciones fallidas: {results['failed_checks']}")
    print(f"   - Estado de seguridad: {results['security_status']}")
    print(f"   - Seguro para operar: {'✅ SÍ' if results['safe_to_trade'] else '❌ NO'}")
    
    # Mostrar fallos críticos
    if results['critical_failures']:
        print(f"\n🚨 FALLOS CRÍTICOS DETECTADOS:")
        for failure in results['critical_failures']:
            print(f"   - {failure['name']}: {failure['reason']}")
    
    # Mostrar advertencias
    if results['warnings']:
        print(f"\n⚠️  ADVERTENCIAS:")
        for warning in results['warnings']:
            print(f"   - {warning['name']}: {warning['reason']}")
    
    # Mostrar resumen final
    print("\n" + "=" * 70)
    
    if results['safe_to_trade']:
        print("🎉 VALIDACIÓN DE SEGURIDAD COMPLETADA EXITOSAMENTE")
        print("✅ Sistema seguro para operar en modo paper")
        print("🔒 Protección contra operaciones reales activa")
        print("📊 Paper trading listo para usar")
        
        # Crear procedimiento de emergencia
        emergency_procedure = create_emergency_stop_procedure()
        print("\n" + emergency_procedure)
        
        return 0
    else:
        print("❌ VALIDACIÓN DE SEGURIDAD FALLIDA")
        print("🚨 Sistema NO seguro para operar")
        print("⚠️  Corrija los fallos críticos antes de operar")
        print("🔒 No intente operar hasta que todas las validaciones pasen")
        
        return 1

if __name__ == "__main__":
    sys.exit(main())