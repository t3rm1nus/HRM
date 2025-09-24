# hacienda/tax_utils.py
# Utilidades para gestión fiscal española de criptomonedas

import os
import json
from datetime import datetime
from typing import Dict, Any, Optional

from core.logging import logger
from .tax_tracker import TaxTracker

class TaxUtils:
    """
    Utilidades para gestión fiscal española de criptomonedas
    Proporciona herramientas para generar informes fiscales y exportar datos
    """

    def __init__(self, hacienda_dir: str = "hacienda"):
        self.hacienda_dir = hacienda_dir
        self.tax_tracker = TaxTracker(hacienda_dir)
        logger.info("✅ TaxUtils inicializado")

    def generate_annual_tax_report(self, tax_year: Optional[int] = None) -> Dict[str, Any]:
        """
        Genera informe fiscal anual completo

        Args:
            tax_year: Año fiscal (default: año actual)

        Returns:
            Diccionario con informe fiscal completo
        """
        try:
            if tax_year is None:
                tax_year = datetime.utcnow().year

            logger.info(f"📊 Generando informe fiscal para {tax_year}")

            # Generar resumen fiscal
            tax_summary = self.tax_tracker.generate_tax_report(tax_year)

            if not tax_summary:
                logger.warning(f"⚠️ No hay datos fiscales para el año {tax_year}")
                return {}

            # Obtener resumen de posiciones
            positions_summary = self.tax_tracker.get_positions_summary()

            # Crear informe completo
            annual_report = {
                'tax_year': tax_year,
                'generated_at': datetime.utcnow().isoformat(),
                'tax_summary': tax_summary,
                'positions_summary': positions_summary,
                'files_generated': []
            }

            # Exportar datos para declaración de impuestos
            export_file = self.tax_tracker.export_for_tax_declaration(tax_year)
            if export_file:
                annual_report['files_generated'].append({
                    'type': 'tax_declaration_export',
                    'file': export_file,
                    'description': f'Datos exportados para declaración de impuestos {tax_year}'
                })

            # Guardar informe anual
            self._save_annual_report(annual_report)

            logger.info("✅ Informe fiscal anual generado exitosamente")
            logger.info(f"   Archivo: hacienda/informe_fiscal_{tax_year}.json")
            if export_file:
                logger.info(f"   Export: {export_file}")

            return annual_report

        except Exception as e:
            logger.error(f"❌ Error generando informe fiscal anual: {e}")
            return {}

    def _save_annual_report(self, report: Dict[str, Any]):
        """Guarda el informe fiscal anual"""
        try:
            tax_year = report['tax_year']
            report_file = os.path.join(self.hacienda_dir, f"informe_fiscal_{tax_year}.json")

            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)

        except Exception as e:
            logger.error(f"❌ Error guardando informe anual: {e}")

    def show_tax_summary(self, tax_year: Optional[int] = None):
        """
        Muestra resumen fiscal en consola

        Args:
            tax_year: Año fiscal (default: año actual)
        """
        try:
            if tax_year is None:
                tax_year = datetime.utcnow().year

            # Obtener resumen fiscal
            tax_summary = self.tax_tracker.generate_tax_report(tax_year)

            if not tax_summary:
                print(f"⚠️ No hay datos fiscales para el año {tax_year}")
                return

            print("\n" + "="*80)
            print(f"📊 RESUMEN FISCAL ESPAÑOL - AÑO {tax_year}")
            print("="*80)

            print("\n💰 OPERACIONES:")
            print(f"   Total operaciones: {tax_summary['total_operations']}")
            print(f"   Compras: {tax_summary['buy_operations']}")
            print(f"   Ventas: {tax_summary['sell_operations']}")

            print("\n💵 VALORES:")
            print(f"   Total comprado: ${tax_summary['total_buy_value']:,.2f}")
            print(f"   Total vendido: ${tax_summary['total_sell_value']:,.2f}")
            print(f"   Comisiones totales: ${tax_summary['total_commissions']:,.2f}")

            print("\n📈 GANANCIAS/PÉRDIDAS:")
            print(f"   Ganancias realizadas: ${tax_summary['realized_gains']:,.2f}")
            print(f"   Pérdidas realizadas: ${tax_summary['realized_losses']:,.2f}")
            print(f"   Resultado neto: ${tax_summary['net_realized_gains']:,.2f}")

            print("\n🏦 POSICIONES ABIERTAS:")
            print(f"   Valor posiciones abiertas: ${tax_summary['open_positions_value']:,.2f}")

            print("\n📋 BASE IMPONIBLE:")
            print(f"   Base imponible: ${tax_summary['tax_base']:,.2f}")

            # Mostrar posiciones actuales
            positions = self.tax_tracker.get_positions_summary()
            if positions:
                print("\n📊 POSICIONES ACTUALES:")
                for symbol, pos_data in positions.items():
                    if pos_data['total_quantity'] > 0:
                        print(f"   {symbol}: {pos_data['total_quantity']:.6f} @ ${pos_data['avg_price']:,.2f}")

            print("\n" + "="*80)
            print("💡 RECUERDA: Consulta con tu asesor fiscal para confirmar el tratamiento correcto")
            print("="*80)

        except Exception as e:
            logger.error(f"❌ Error mostrando resumen fiscal: {e}")

    def export_tax_data(self, tax_year: Optional[int] = None, format: str = "csv") -> str:
        """
        Exporta datos fiscales en diferentes formatos

        Args:
            tax_year: Año fiscal
            format: Formato de exportación ("csv", "json", "excel")

        Returns:
            Ruta del archivo exportado
        """
        try:
            if tax_year is None:
                tax_year = datetime.utcnow().year

            if format.lower() == "csv":
                return self.tax_tracker.export_for_tax_declaration(tax_year)
            elif format.lower() == "json":
                return self._export_tax_data_json(tax_year)
            elif format.lower() == "excel":
                return self._export_tax_data_excel(tax_year)
            else:
                logger.error(f"❌ Formato no soportado: {format}")
                return ""

        except Exception as e:
            logger.error(f"❌ Error exportando datos fiscales: {e}")
            return ""

    def _export_tax_data_json(self, tax_year: int) -> str:
        """Exporta datos fiscales en formato JSON"""
        try:
            export_file = os.path.join(self.hacienda_dir, f"datos_fiscales_{tax_year}.json")

            # Obtener datos del año
            year_operations = [
                op for op in self.tax_tracker.tax_operations
                if datetime.fromisoformat(op['timestamp']).year == tax_year
            ]

            # Obtener ganancias realizadas
            gains_file = os.path.join(self.hacienda_dir, "ganancias_realizadas.csv")
            realized_gains = []

            if os.path.exists(gains_file):
                import csv
                with open(gains_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if datetime.fromisoformat(row['sell_date']).year == tax_year:
                            realized_gains.append(row)

            # Crear estructura de datos
            export_data = {
                'tax_year': tax_year,
                'exported_at': datetime.utcnow().isoformat(),
                'operations': year_operations,
                'realized_gains': realized_gains,
                'positions': self.tax_tracker.get_positions_summary()
            }

            # Guardar archivo
            with open(export_file, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)

            logger.info(f"📤 Datos fiscales exportados: {export_file}")
            return export_file

        except Exception as e:
            logger.error(f"❌ Error exportando datos JSON: {e}")
            return ""

    def _export_tax_data_excel(self, tax_year: int) -> str:
        """Exporta datos fiscales en formato Excel"""
        try:
            import pandas as pd

            export_file = os.path.join(self.hacienda_dir, f"datos_fiscales_{tax_year}.xlsx")

            # Obtener datos del año
            year_operations = [
                op for op in self.tax_tracker.tax_operations
                if datetime.fromisoformat(op['timestamp']).year == tax_year
            ]

            # Crear DataFrames
            operations_df = pd.DataFrame(year_operations)

            # Obtener ganancias realizadas
            gains_file = os.path.join(self.hacienda_dir, "ganancias_realizadas.csv")
            if os.path.exists(gains_file):
                gains_df = pd.read_csv(gains_file)
                gains_df = gains_df[gains_df['sell_date'].str.startswith(str(tax_year))]
            else:
                gains_df = pd.DataFrame()

            # Crear archivo Excel con múltiples hojas
            with pd.ExcelWriter(export_file, engine='openpyxl') as writer:
                if not operations_df.empty:
                    operations_df.to_excel(writer, sheet_name='Operaciones', index=False)
                if not gains_df.empty:
                    gains_df.to_excel(writer, sheet_name='Ganancias_Realizadas', index=False)

                # Hoja de resumen
                summary_data = self.tax_tracker.generate_tax_report(tax_year)
                if summary_data:
                    summary_df = pd.DataFrame([summary_data])
                    summary_df.to_excel(writer, sheet_name='Resumen_Fiscal', index=False)

            logger.info(f"📤 Datos fiscales exportados a Excel: {export_file}")
            return export_file

        except ImportError:
            logger.error("❌ Pandas no disponible para exportación Excel")
            return ""
        except Exception as e:
            logger.error(f"❌ Error exportando datos Excel: {e}")
            return ""

    def get_tax_advice(self) -> str:
        """
        Proporciona consejos fiscales básicos para criptomonedas en España

        Returns:
            Texto con consejos fiscales
        """
        advice = """
💡 CONSEJOS FISCALES PARA CRIPTOMONEDAS EN ESPAÑA

📅 PERÍODO DE TENENCIA:
• Corto plazo: <1 año → IRPF general (19-47%)
• Largo plazo: >1 año → IRPF reducido (19-26%)

💰 GANANCIAS/PÉRDIDAS:
• Se calculan por método FIFO (primera entrada, primera salida)
• Las pérdidas se pueden compensar con ganancias
• Límite de compensación: 25% de la base imponible

📋 DECLARACIÓN:
• Modelo 100 (IRPF) - Base imponible del ahorro
• Modelo 720 - Declaración de bienes en el extranjero (>50,000€)
• Plazo: Abril-Junio del año siguiente

⚠️ IMPORTANTE:
• Este software calcula ganancias/pérdidas según normativa española
• Los datos generados son para ayuda en la declaración
• CONSULTA SIEMPRE con tu asesor fiscal o AEAT
• La interpretación de la normativa puede cambiar

🔗 RECURSOS ÚTILES:
• AEAT: https://www.agenciatributaria.es
• Información sobre criptomonedas: Consulta web de la AEAT

⚖️ DESCARGO DE RESPONSABILIDAD:
Esta herramienta es solo para ayuda en la gestión fiscal.
No constituye asesoramiento fiscal profesional.
        """

        return advice

def main():
    """Función principal para uso desde línea de comandos"""
    import argparse

    parser = argparse.ArgumentParser(description='Utilidades fiscales para criptomonedas - España')
    parser.add_argument('--year', type=int, help='Año fiscal (default: actual)')
    parser.add_argument('--action', choices=['summary', 'export', 'report'],
                       default='summary', help='Acción a realizar')
    parser.add_argument('--format', choices=['csv', 'json', 'excel'],
                       default='csv', help='Formato de exportación')

    args = parser.parse_args()

    # Inicializar utilidades fiscales
    tax_utils = TaxUtils()

    if args.action == 'summary':
        tax_utils.show_tax_summary(args.year)
    elif args.action == 'export':
        export_file = tax_utils.export_tax_data(args.year, args.format)
        if export_file:
            print(f"✅ Datos exportados: {export_file}")
        else:
            print("❌ Error en la exportación")
    elif args.action == 'report':
        report = tax_utils.generate_annual_tax_report(args.year)
        if report:
            print(f"✅ Informe generado: hacienda/informe_fiscal_{args.year}.json")
        else:
            print("❌ Error generando informe")

    # Mostrar consejos fiscales
    print("\n" + tax_utils.get_tax_advice())

if __name__ == "__main__":
    main()
