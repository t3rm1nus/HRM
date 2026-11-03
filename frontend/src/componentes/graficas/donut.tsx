import React, { useRef, useEffect, useState } from 'react';
import { LightweightCharts } from 'lightweight-charts';

export interface DonutProps {
  titulo: string;
  datos: { nombre: string; valor: number; color?: string }[];
  width?: number;
  height?: number;
  theme?: 'light' | 'dark';
}

export const Donut = ({
  titulo,
  datos,
  width = 400,
  height = 400,
  theme = 'dark',
}: DonutProps) => {
  const chartRef = useRef<LightweightCharts.Widget | null>(null);
  const [isMounted, setIsMounted] = useState(false);

  useEffect(() => {
    if (!chartRef.current) return;

    const chart = chartRef.current.renderer();
    const themeConfig = theme === 'dark' ? {
      background: { type: 'solid', color: '#111827' },
      grid: { vertLine: { color: '#374151' }, horzLine: { color: '#374151' } },
      timeScale: { borderVisible: false },
      crosshair: { mode: 'hide' },
      layout: { background: { type: 'solid', color: '#111827' }, textColor: '#F9FAFB' },
    } : {
      background: { type: 'solid', color: '#FFFFFF' },
      grid: { vertLine: { color: '#E5E7EB' }, horzLine: { color: '#E5E7EB' } },
      timeScale: { borderVisible: true },
      crosshair: { mode: 'hide' },
      layout: { background: { type: 'solid', color: '#FFFFFF' }, textColor: '#111827' },
    };

    chart.applyOptions(themeConfig);

    // Donut chart usando pie series
    const series = chart.addPieSeries({
      data: datos.map(d => ({ price: d.valor, label: d.nombre })),
      stroke: '#FFFFFF',
      strokeWidth: 2,
    });

    setIsMounted(true);

    return () => {
      chartRef.current?.remove();
    };
  }, [datos, theme]);

  if (!isMounted) {
    return <div className="h-[300px] w-full" />;
  }

  return (
    <div className="rounded-md p-4 border border-gray-200">
      <h3 className="text-sm font-medium text-gray-700 mb-2">{titulo}</h3>
      <div
        ref={chartRef}
        style={{ width: width, height: height }}
        className="h-[300px] w-full"
      />
    </div>
  );
};

export default Donut;