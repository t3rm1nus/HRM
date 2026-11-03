import React, { useRef, useEffect, useState } from 'react';
import { LightweightCharts } from 'lightweight-charts';

export interface HistogramaProps {
  titulo: string;
  rangos: { rango: string; count: number; monto?: number }[];
  width?: number;
  height?: number;
  theme?: 'light' | 'dark';
}

export const Histograma = ({
  titulo,
  rangos,
  width = 800,
  height = 400,
  theme = 'dark',
}: HistogramaProps) => {
  const chartRef = useRef<LightweightCharts.Widget | null>(null);
  const [isMounted, setIsMounted] = useState(false);

  useEffect(() => {
    if (!chartRef.current) return;

    const chart = chartRef.current.renderer();
    const themeConfig = theme === 'dark' ? {
      background: { type: 'solid', color: '#111827' },
      grid: { vertLine: { color: '#374151' }, horzLine: { color: '#374151' } },
      timeScale: { borderVisible: false, timeVisible: true },
      crosshair: { mode: 'hide' },
      layout: { background: { type: 'solid', color: '#111827' }, textColor: '#F9FAFB' },
    } : {
      background: { type: 'solid', color: '#FFFFFF' },
      grid: { vertLine: { color: '#E5E7EB' }, horzLine: { color: '#E5E7EB' } },
      timeScale: { borderVisible: true, timeVisible: true },
      crosshair: { mode: 'hide' },
      layout: { background: { type: 'solid', color: '#FFFFFF' }, textColor: '#111827' },
    };

    chart.applyOptions(themeConfig);

    // Histograma usando barra series con agrupación
    const series = chart.addHistogramSeries({
      color: '#F59E0B',
      lineWidth: 0,
    });

    const data = rangos.map(r => ({
      time: r.rango,
      value: r.count,
    }));

    series.setData(data);

    setIsMounted(true);

    return () => {
      chartRef.current?.remove();
    };
  }, [rangos, theme]);

  if (!isMounted) {
    return <div className="h-[400px] w-full" />;
  }

  return (
    <div className="rounded-md p-4 border border-gray-200">
      <h3 className="text-sm font-medium text-gray-700 mb-2">{titulo}</h3>
      <div
        ref={chartRef}
        style={{ height: height }}
        className="h-[400px] w-full"
      />
    </div>
  );
};

export default Histograma;