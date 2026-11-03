import React, { useEffect, useRef } from 'react';
import { LightweightCharts } from 'lightweight-charts';
import { calcularDrawdownSubmarino } from '../../dominio/graficas/funciones';

export interface CurvaEquityProps {
  datosOperaciones: {
    timestamp: string;
    pnl: number;
    regimen: 'PAPEL' | 'TESTNET' | 'LIVE';
    tipo: 'ingreso' | 'egreso' | 'transferencia';
  }[];
  width?: number;
  height?: number;
  theme?: 'light' | 'dark';
}

export const CurvaEquity: React.FC<CurvaEquityProps> = ({
  datosOperaciones,
  width = 800,
  height = 400,
  theme = 'dark',
}) => {
  const chartRef = useRef<LightweightCharts.Widget | null>(null);

  useEffect(() => {
    if (!chartRef.current) return;

    const chart = chartRef.current.renderer();
    const series = chart.addLineSeries({
      color: '#10B981',
      lineWidth: 2,
      title: 'Curva Equity',
    });

    // Calcular drawdown submarino
    const precios = datosOperaciones.map((op) => Math.abs(op.pnl) + 1000);
    const drawdown = calcularDrawdownSubmarino(precios);

    const formattedData = datosOperaciones.map((op) => ({
      time: op.timestamp,
      value: op.pnl,
    }));

    series.setData(formattedData);

    // añadir marcas de operaciones
    datosOperaciones.forEach((op, i) => {
      const mark = {
        position: i,
        label: `${op.tipo}: $${op.pnl.toFixed(2)}`,
        color: op.pnl > 0 ? '#10B981' : '#EF4444',
      };
      chart.addMark(mark);
    });

    setTimeout(() => {
      chartRef.current?.fitContent();
    }, 100);

    setTimeout(() => {
      chartRef.current?.remove();
    }, 300000);
  }, [datosOperaciones, theme]);

  return (
    <div className="rounded-md p-4 border border-gray-200">
      <h3 className="text-sm font-medium text-gray-700 mb-2">Curva de Equity</h3>
      <div
        ref={chartRef}
        style={{ height: height }}
        className="h-[400px] w-full"
      />
      <p className="text-xs text-gray-500 mt-2">
        Drawdown Máx: {((calcularDrawdownSubmarino(
        datosOperaciones.map((op) => Math.abs(op.pnl) + 1000)
      ).maxDrawdown).toFixed(2))}%
    </div>
  );
};

export default CurvaEquity;