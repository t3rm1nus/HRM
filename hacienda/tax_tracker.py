# hacienda/tax_tracker.py
# Sistema de seguimiento fiscal para declaración de impuestos española
# Implementa método FIFO para cálculo de ganancias/pérdidas

import os
import csv
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from collections import defaultdict
import pandas as pd

from core.logging import logger

class FIFOPosition:
    """Representa una posición FIFO para cálculo de ganancias/pérdidas"""

    def __init__(self, symbol: str, quantity: float, price: float, timestamp: str, operation_id: str, remaining_quantity: float = None):
        self.symbol = symbol
        self.quantity = quantity
        self.price = price
        self.timestamp = timestamp
        self.operation_id = operation_id
        self.remaining_quantity = remaining_quantity if remaining_quantity is not None else quantity

    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'quantity': self.quantity,
            'price': self.price,
            'timestamp': self.timestamp,
            'operation_id': self.operation_id,
            'remaining_quantity': self.remaining_quantity
        }

class TaxTracker:
    """
    Sistema de seguimiento fiscal para criptomonedas según normativa española
    Implementa método FIFO para cálculo de ganancias y pérdidas
    """

    def __init__(self, hacienda_dir: str = "hacienda"):
        self.hacienda_dir = hacienda_dir
        self.operations_file = os.path.join(hacienda_dir, "operaciones.csv")
        self.positions_file = os.path.join(hacienda_dir, "posiciones_fifo.json")
        self.tax_report_file = os.path.join(hacienda_dir, "informe_fiscal.csv")

        # Crear directorio si no existe
        os.makedirs(hacienda_dir, exist_ok=True)

        # Estructuras de datos FIFO
        self.positions: Dict[str, List[FIFOPosition]] = defaultdict(list)  # symbol -> lista de posiciones
        self.tax_operations: List[Dict[str, Any]] = []

        # Cargar datos existentes
        self._load_positions()
        self._load_operations()

        logger.info(f"✅ TaxTracker inicializado - Directorio: {hacienda_dir}")

    def _load_positions(self):
        """Carga posiciones FIFO desde archivo"""
        try:
            if os.path.exists(self.positions_file):
                with open(self.positions_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                for symbol, positions_data in data.items():
                    self.positions[symbol] = [
                        FIFOPosition(**pos_data) for pos_data in positions_data
                    ]

                logger.info(f"✅ Posiciones FIFO cargadas: {sum(len(pos) for pos in self.positions.values())} posiciones")
        except Exception as e:
            logger.error(f"❌ Error cargando posiciones FIFO: {e}")

    def _save_positions(self):
        """Guarda posiciones FIFO a archivo"""
        try:
            data = {}
            for symbol, positions in self.positions.items():
                data[symbol] = [pos.to_dict() for pos in positions]

            with open(self.positions_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            logger.debug("💾 Posiciones FIFO guardadas")
        except Exception as e:
            logger.error(f"❌ Error guardando posiciones FIFO: {e}")

    def _load_operations(self):
        """Carga operaciones desde archivo CSV"""
        try:
            if os.path.exists(self.operations_file):
                with open(self.operations_file, 'r', encoding='utf-8', newline='') as f:
                    reader = csv.DictReader(f)
                    self.tax_operations = list(reader)

                logger.info(f"✅ Operaciones cargadas: {len(self.tax_operations)}")
        except Exception as e:
            logger.error(f"❌ Error cargando operaciones: {e}")

    def _save_operations(self):
        """Guarda operaciones a archivo CSV"""
        try:
            if not self.tax_operations:
                return

            fieldnames = [
                'operation_id', 'timestamp', 'symbol', 'side', 'quantity', 'price',
                'total_value', 'commission', 'net_value', 'exchange', 'tax_year'
            ]

            with open(self.operations_file, 'w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for op in self.tax_operations:
                    writer.writerow(op)

            logger.debug("💾 Operaciones guardadas")
        except Exception as e:
            logger.error(f"❌ Error guardando operaciones: {e}")

    def record_operation(self, order: Dict[str, Any], exchange: str = "Binance"):
        """
        Registra una operación para fines fiscales

        Args:
            order: Diccionario con datos de la orden ejecutada
            exchange: Nombre del exchange (default: Binance)
        """
        try:
            operation_id = f"{order.get('symbol', 'UNKNOWN')}_{datetime.utcnow().isoformat()}_{id(order)}"

            # Extraer datos de la orden
            symbol = order.get('symbol', 'UNKNOWN')
            side = order.get('side', 'unknown')
            quantity = abs(float(order.get('filled_quantity', order.get('quantity', 0))))
            price = float(order.get('filled_price', order.get('price', 0)))
            commission = float(order.get('commission', 0))

            # Calcular valores
            total_value = quantity * price
            net_value = total_value - commission

            # Crear registro de operación
            operation = {
                'operation_id': operation_id,
                'timestamp': datetime.utcnow().isoformat(),
                'symbol': symbol,
                'side': side,
                'quantity': quantity,
                'price': price,
                'total_value': round(total_value, 2),
                'commission': round(commission, 2),
                'net_value': round(net_value, 2),
                'exchange': exchange,
                'tax_year': datetime.utcnow().year
            }

            # Agregar a lista de operaciones
            self.tax_operations.append(operation)

            # Actualizar posiciones FIFO
            self._update_fifo_positions(operation)

            # Guardar inmediatamente
            self._save_operations()
            self._save_positions()

            logger.info(f"📝 Operación registrada: {symbol} {side} {quantity:.6f} @ {price:.2f}")

        except Exception as e:
            logger.error(f"❌ Error registrando operación: {e}")

    def _update_fifo_positions(self, operation: Dict[str, Any]):
        """
        Actualiza posiciones FIFO basado en la operación

        Args:
            operation: Diccionario con datos de la operación
        """
        try:
            symbol = operation['symbol']
            side = operation['side']
            quantity = operation['quantity']
            price = operation['price']
            operation_id = operation['operation_id']

            if side.lower() == 'buy':
                # Agregar nueva posición (primera entrada)
                position = FIFOPosition(symbol, quantity, price, operation['timestamp'], operation_id)
                self.positions[symbol].append(position)
                logger.debug(f"📈 BUY: Nueva posición FIFO para {symbol}: {quantity} @ {price}")

            elif side.lower() == 'sell':
                # Procesar venta usando FIFO (primera salida)
                remaining_to_sell = quantity
                realized_gains = []

                while remaining_to_sell > 0 and self.positions[symbol]:
                    # Tomar la posición más antigua (FIFO)
                    oldest_position = self.positions[symbol][0]

                    if oldest_position.remaining_quantity <= remaining_to_sell:
                        # Vender toda la posición
                        sell_quantity = oldest_position.remaining_quantity
                        cost_basis = oldest_position.price * sell_quantity
                        sell_value = price * sell_quantity
                        gain_loss = sell_value - cost_basis

                        realized_gains.append({
                            'operation_id': operation_id,
                            'symbol': symbol,
                            'sell_date': operation['timestamp'],
                            'buy_date': oldest_position.timestamp,
                            'quantity': sell_quantity,
                            'buy_price': oldest_position.price,
                            'sell_price': price,
                            'cost_basis': round(cost_basis, 2),
                            'sell_value': round(sell_value, 2),
                            'gain_loss': round(gain_loss, 2),
                            'holding_period_days': self._calculate_holding_period(
                                oldest_position.timestamp, operation['timestamp']
                            )
                        })

                        # Remover posición completamente vendida
                        self.positions[symbol].pop(0)
                        remaining_to_sell -= sell_quantity

                    else:
                        # Vender parte de la posición
                        sell_quantity = remaining_to_sell
                        cost_basis = oldest_position.price * sell_quantity
                        sell_value = price * sell_quantity
                        gain_loss = sell_value - cost_basis

                        realized_gains.append({
                            'operation_id': operation_id,
                            'symbol': symbol,
                            'sell_date': operation['timestamp'],
                            'buy_date': oldest_position.timestamp,
                            'quantity': sell_quantity,
                            'buy_price': oldest_position.price,
                            'sell_price': price,
                            'cost_basis': round(cost_basis, 2),
                            'sell_value': round(sell_value, 2),
                            'gain_loss': round(gain_loss, 2),
                            'holding_period_days': self._calculate_holding_period(
                                oldest_position.timestamp, operation['timestamp']
                            )
                        })

                        # Reducir cantidad restante en la posición
                        oldest_position.remaining_quantity -= sell_quantity
                        remaining_to_sell = 0

                if realized_gains:
                    self._save_realized_gains(realized_gains)
                    total_gain_loss = sum(g['gain_loss'] for g in realized_gains)
                    logger.info(f"📉 SELL: Ganancia/pérdida realizada {symbol}: {total_gain_loss:+.2f} USDT")

                if remaining_to_sell > 0:
                    logger.warning(f"⚠️ Intento de vender más de lo disponible: {remaining_to_sell} {symbol} restantes")

        except Exception as e:
            logger.error(f"❌ Error actualizando posiciones FIFO: {e}")

    def _calculate_holding_period(self, buy_timestamp: str, sell_timestamp: str) -> int:
        """Calcula el período de tenencia en días"""
        try:
            buy_date = datetime.fromisoformat(buy_timestamp.replace('Z', '+00:00'))
            sell_date = datetime.fromisoformat(sell_timestamp.replace('Z', '+00:00'))
            return (sell_date - buy_date).days
        except:
            return 0

    def _save_realized_gains(self, gains: List[Dict[str, Any]]):
        """Guarda ganancias realizadas en archivo separado"""
        try:
            gains_file = os.path.join(self.hacienda_dir, "ganancias_realizadas.csv")

            fieldnames = [
                'operation_id', 'symbol', 'sell_date', 'buy_date', 'quantity',
                'buy_price', 'sell_price', 'cost_basis', 'sell_value',
                'gain_loss', 'holding_period_days'
            ]

            file_exists = os.path.exists(gains_file)

            with open(gains_file, 'a', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()
                for gain in gains:
                    writer.writerow(gain)

        except Exception as e:
            logger.error(f"❌ Error guardando ganancias realizadas: {e}")

    def generate_tax_report(self, tax_year: Optional[int] = None) -> Dict[str, Any]:
        """
        Genera informe fiscal para declaración de impuestos

        Args:
            tax_year: Año fiscal (default: año actual)

        Returns:
            Diccionario con resumen fiscal
        """
        try:
            if tax_year is None:
                tax_year = datetime.utcnow().year

            # Filtrar operaciones del año fiscal
            year_operations = [
                op for op in self.tax_operations
                if datetime.fromisoformat(op['timestamp']).year == tax_year
            ]

            if not year_operations:
                logger.info(f"ℹ️ No hay operaciones para el año fiscal {tax_year}")
                return {}

            # Cargar ganancias realizadas
            gains_file = os.path.join(self.hacienda_dir, "ganancias_realizadas.csv")
            realized_gains = []

            if os.path.exists(gains_file):
                with open(gains_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if datetime.fromisoformat(row['sell_date']).year == tax_year:
                            realized_gains.append({
                                'gain_loss': float(row['gain_loss']),
                                'symbol': row['symbol'],
                                'sell_date': row['sell_date']
                            })

            # Calcular estadísticas fiscales
            total_operations = len(year_operations)
            buy_operations = [op for op in year_operations if op['side'].lower() == 'buy']
            sell_operations = [op for op in year_operations if op['side'].lower() == 'sell']

            total_buy_value = sum(float(op['net_value']) for op in buy_operations)
            total_sell_value = sum(float(op['net_value']) for op in sell_operations)
            total_commissions = sum(float(op['commission']) for op in year_operations)

            # Ganancias/pérdidas realizadas
            total_realized_gains = sum(g['gain_loss'] for g in realized_gains if g['gain_loss'] > 0)
            total_realized_losses = abs(sum(g['gain_loss'] for g in realized_gains if g['gain_loss'] < 0))

            # Posiciones abiertas al final del año
            open_positions_value = 0.0
            for symbol, positions in self.positions.items():
                for pos in positions:
                    # Usar precio actual aproximado (esto debería mejorarse con precios reales)
                    open_positions_value += pos.remaining_quantity * pos.price

            # Crear resumen fiscal
            tax_summary = {
                'tax_year': tax_year,
                'total_operations': total_operations,
                'buy_operations': len(buy_operations),
                'sell_operations': len(sell_operations),
                'total_buy_value': round(total_buy_value, 2),
                'total_sell_value': round(total_sell_value, 2),
                'total_commissions': round(total_commissions, 2),
                'realized_gains': round(total_realized_gains, 2),
                'realized_losses': round(total_realized_losses, 2),
                'net_realized_gains': round(total_realized_gains - total_realized_losses, 2),
                'open_positions_value': round(open_positions_value, 2),
                'tax_base': round(total_realized_gains - total_realized_losses, 2),  # Base imponible
                'generated_at': datetime.utcnow().isoformat()
            }

            # Guardar resumen
            self._save_tax_summary(tax_summary)

            logger.info(f"📊 Informe fiscal generado para {tax_year}:")
            logger.info(f"   Ganancias realizadas: {tax_summary['realized_gains']:.2f} USDT")
            logger.info(f"   Pérdidas realizadas: {tax_summary['realized_losses']:.2f} USDT")
            logger.info(f"   Base imponible: {tax_summary['tax_base']:.2f} USDT")

            return tax_summary

        except Exception as e:
            logger.error(f"❌ Error generando informe fiscal: {e}")
            return {}

    def _save_tax_summary(self, summary: Dict[str, Any]):
        """Guarda resumen fiscal en archivo JSON"""
        try:
            summary_file = os.path.join(self.hacienda_dir, f"resumen_fiscal_{summary['tax_year']}.json")

            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)

        except Exception as e:
            logger.error(f"❌ Error guardando resumen fiscal: {e}")

    def get_positions_summary(self) -> Dict[str, Any]:
        """Obtiene resumen de posiciones actuales"""
        try:
            summary = {}

            for symbol, positions in self.positions.items():
                total_quantity = sum(pos.remaining_quantity for pos in positions)
                total_cost = sum(pos.remaining_quantity * pos.price for pos in positions)
                avg_price = total_cost / total_quantity if total_quantity > 0 else 0

                summary[symbol] = {
                    'total_quantity': round(total_quantity, 6),
                    'total_cost': round(total_cost, 2),
                    'avg_price': round(avg_price, 2),
                    'num_positions': len(positions)
                }

            return summary

        except Exception as e:
            logger.error(f"❌ Error obteniendo resumen de posiciones: {e}")
            return {}

    def export_for_tax_declaration(self, tax_year: Optional[int] = None) -> str:
        """
        Exporta datos en formato adecuado para declaración de impuestos española

        Returns:
            Ruta del archivo exportado
        """
        try:
            if tax_year is None:
                tax_year = datetime.utcnow().year

            export_file = os.path.join(self.hacienda_dir, f"declaracion_impuestos_{tax_year}.csv")

            # Filtrar operaciones del año
            year_operations = [
                op for op in self.tax_operations
                if datetime.fromisoformat(op['timestamp']).year == tax_year
            ]

            if not year_operations:
                logger.warning(f"⚠️ No hay operaciones para exportar en {tax_year}")
                return ""

            # Crear DataFrame para exportación
            df = pd.DataFrame(year_operations)

            # Renombrar columnas para claridad fiscal
            column_mapping = {
                'timestamp': 'Fecha_Hora',
                'symbol': 'Criptomoneda',
                'side': 'Tipo_Operacion',  # BUY/SELL
                'quantity': 'Cantidad',
                'price': 'Precio_Unitario_USD',
                'total_value': 'Valor_Total_USD',
                'commission': 'Comision_USD',
                'net_value': 'Valor_Neto_USD',
                'exchange': 'Exchange'
            }

            df = df.rename(columns=column_mapping)

            # Agregar columna de fecha solo (sin hora) para agrupación fiscal
            df['Fecha'] = pd.to_datetime(df['Fecha_Hora']).dt.date

            # Convertir tipos BUY/SELL a términos fiscales españoles
            df['Tipo_Operacion'] = df['Tipo_Operacion'].map({
                'buy': 'Compra',
                'sell': 'Venta'
            })

            # Guardar archivo de exportación
            df.to_csv(export_file, index=False, encoding='utf-8-sig')  # utf-8-sig para Excel español

            logger.info(f"📤 Datos exportados para declaración: {export_file}")
            logger.info(f"   Operaciones: {len(df)}")
            logger.info(f"   Compras: {len(df[df['Tipo_Operacion'] == 'Compra'])}")
            logger.info(f"   Ventas: {len(df[df['Tipo_Operacion'] == 'Venta'])}")

            return export_file

        except Exception as e:
            logger.error(f"❌ Error exportando datos fiscales: {e}")
            return ""
