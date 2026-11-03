import React, { useRef, useEffect, useState } from 'react';
import { LightweightCharts } from 'lightweight-charts';

export interface GraficaLineaProps {
  titulo: string;
  datos: { timestamp: string; valor: number }[];
  width?: number;
  height?: number;
  theme?: 'light' | 'dark';
  showGrid?: boolean;
  crosshairSource?: string;
  onPointClick?: (valor: number, timestamp: string) => void;
}

export const GraficaLinea = ({
  titulo,
  datos,
  width = 800,
  height = 400,
  theme = 'dark',
  showGrid = true,
  crosshairSource,
  onPointClick,
}: GraficaLineaProps) => {
  const chartRef = useRef<LightweightCharts.Widget | null>(null);
  const [isMounted, setIsMounted] = useState(false);

  useEffect(() => {
    if (!chartRef.current) return;

    const chart = chartRef.current.renderer();
    const themeConfig = theme === 'dark' ? {
      background: { type: 'solid', color: '#111827' },
      grid: { vertLine: { color: '#374151' }, horzLine: { color: '#374151' } },
      rightPriceScale: { borderVisible: false, backgroundColor: '#111827', textColor: '#F9FAFB' },
      timeScale: { borderVisible: false, timeVisible: true, secondsVisible: false },
      crosshair: {
        mode: 'visible',
        color: '#FFFFFF',
        vertLine: { color: '#FFFFFF' },
        horzLine: { color: '#FFFFFF' },
        horzLineVisibleInRange: false,
      },
      layout: { background: { type: 'solid', color: '#111827' }, textColor: '#F9FAFB' },
    } : {
      background: { type: 'solid', color: '#FFFFFF' },
      grid: { vertLine: { color: '#E5E7EB' }, horzLine: { color: '#E5E7EB' } },
      rightPriceScale: { borderVisible: true, backgroundColor: '#FFFFFF', textColor: '#111827' },
      timeScale: { borderVisible: true, timeVisible: true, secondsVisible: false },
      crosshair: {
        mode: 'visible',
        color: '#000000',
        vertLine: { color: '#000000' },
        horzLine: { color: '#000000' },
        horzLineVisibleInRange: false,
      },
      layout: { background: { type: 'solid', color: '#FFFFFF' }, textColor: '#111827' },
    });

    chart.applyOptions(themeConfig);

    const series = chart.addLineSeries({
      color: '#2563EB',
      lineWidth: 2,
      title: titulo,
    });

    series.setData(datos.map(d => ({ time: d.timestamp, value: d.valor })));

    if (crosshairSource) {
      // Sincronizar crosshair con otra gráfica
      // Esto se manejaría mediante un estado global o props
    }

    setIsMounted(true);

    return () => {
      chartRef.current?.remove();
    };
  }, [datos, theme, width, height, crosshairSource, onPointClick]);

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

export default GraficaLinea;