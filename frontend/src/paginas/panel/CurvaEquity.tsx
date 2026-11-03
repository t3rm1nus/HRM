import React, { useEffect, useRef } from 'react';
import {
  createChart,
  LineSeries,
  createSeriesMarkers,
  type IChartApi,
  type ISeriesApi,
  type Time,
} from 'lightweight-charts';
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
  const contenedorRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const contenedor = contenedorRef.current;
    if (!contenedor) return;

    const grafica: IChartApi = createChart(contenedor, {
      width: width || contenedor.clientWidth,
      height,
      layout: {
        background: { color: theme === 'dark' ? '#111827' : '#FFFFFF' },
        textColor: theme === 'dark' ? '#F9FAFB' : '#111827',
      },
      grid: {
        vertLines: { color: theme === 'dark' ? '#374151' : '#E5E7EB' },
        horzLines: { color: theme === 'dark' ? '#374151' : '#E5E7EB' },
      },
    });

    const serie: ISeriesApi<'Line'> = grafica.addSeries(LineSeries, {
      color: '#10B981',
      lineWidth: 2,
      title: 'Curva Equity',
    });

    const datos = datosOperaciones
      .map((operacion) => ({
        time: Math.floor(new Date(operacion.timestamp).getTime() / 1000) as Time,
        value: operacion.pnl,
      }))
      .filter((punto) => Number.isFinite(punto.time) && Number.isFinite(punto.value))
      .sort((a, b) => Number(a.time) - Number(b.time));

    serie.setData(datos);
    createSeriesMarkers(
      serie,
      datosOperaciones
        .map((operacion) => ({
          time: Math.floor(new Date(operacion.timestamp).getTime() / 1000) as Time,
          position: operacion.pnl >= 0 ? 'belowBar' as const : 'aboveBar' as const,
          color: operacion.pnl >= 0 ? '#10B981' : '#EF4444',
          shape: operacion.pnl >= 0 ? 'arrowUp' as const : 'arrowDown' as const,
          text: `${operacion.tipo}: ${operacion.pnl.toFixed(2)}`,
        }))
        .sort((a, b) => Number(a.time) - Number(b.time))
    );

    grafica.timeScale().fitContent();

    const observador = new ResizeObserver(([entrada]) => {
      if (entrada) grafica.applyOptions({ width: Math.max(entrada.contentRect.width, 1) });
    });
    observador.observe(contenedor);

    return () => {
      observador.disconnect();
      grafica.remove();
    };
  }, [datosOperaciones, height, theme, width]);

  const preciosEquity = datosOperaciones.reduce<number[]>((acumulados, operacion) => {
    const anterior = acumulados[acumulados.length - 1] ?? 1000;
    acumulados.push(anterior + operacion.pnl);
    return acumulados;
  }, []);
  const drawdown = calcularDrawdownSubmarino(preciosEquity);

  return (
    <div className="rounded-md p-4 border border-gray-200">
      <h3 className="text-sm font-medium text-gray-700 mb-2">Curva de Equity</h3>
      <div ref={contenedorRef} style={{ height }} className="w-full" />
      <p className="text-xs text-gray-500 mt-2">
        Drawdown Máx: {drawdown.maxDrawdown.toFixed(2)}%
      </p>
    </div>
  );
};

export default CurvaEquity;
