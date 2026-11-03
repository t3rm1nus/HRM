import React, { useRef, useEffect, useState } from 'react';
import { LightweightCharts } from 'lightweight-charts';

export interface LineaTemporalRegimenesProps {
  titulo: string;
  datos: { timestamp: string; regimen: 'PAPEL' | 'TESTNET' | 'LIVE'; activo: boolean }[];
  width?: number;
  height?: number;
  theme?: 'light' | 'dark';
  mostrarBandas?: boolean;
}

export const LineaTemporalRegimenes = ({
  titulo,
  datos,
  width = 800,
  height = 400,
  theme = 'dark',
  mostrarBandas = true,
}: LineaTemporalRegimenesProps) => {
  const chartRef = useRef<LightweightCharts.Widget | null>(null);
  const [isMounted, setIsMounted] = useState(false);

  useEffect(() => {
    if (!chartRef.current) return;

    const chart = chartRef.current.renderer();
    const themeConfig = theme === 'dark' ? {
      background: { type: 'solid', color: '#111827' },
      grid: { vertLine: { color: '#374151' }, horzLine: { color: '#374151' } },
      timeScale: { borderVisible: false, timeVisible: true, secondsVisible: false },
      crosshair: { mode: 'hide' },
      layout: { background: { type: 'solid', color: '#111827' }, textColor: '#F9FAFB' },
    } : {
      background: { type: 'solid', color: '#FFFFFF' },
      grid: { vertLine: { color: '#E5E7EB' }, horzLine: { color: '#E5E7EB' } },
      timeScale: { borderVisible: true, timeVisible: true, secondsVisible: false },
      crosshair: { mode: 'hide' },
      layout: { background: { type: 'solid', color: '#FFFFFF' }, textColor: '#111827' },
    };

    chart.applyOptions(themeConfig);

    const series = chart.addLineSeries({
      color: '#6B7280',
      lineWidth: 1,
      title: titulo,
    });

    // Añadir bandas de color para regímenes
    if (mostrarBandas) {
      const regimeBands = [
        { from: 0, to: datos.length / 3, color: '#10B981', label: 'PAPEL' },
        { from: datos.length / 3, to: (datos.length * 2) / 3, color: '#F59E0B', label: 'TESTNET' },
        { from: (datos.length * 2) / 3, to: datos.length, color: '#EF4444', label: 'LIVE' },
      ];

      regimeBands.forEach((band, i) => {
        const mark = chart.createPriceBand(
          i,
          band.from,
          band.to,
          { color: band.color, text: band.label }
        );
        chart.addMark(mark);
      });
    }

    series.setData(datos.map(d => ({ time: d.timestamp, value: d.activo ? 1 : 0 })));

    setIsMounted(true);

    return () => {
      chartRef.current?.remove();
    };
  }, [datos, theme, mostrarBandas]);

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

export default LineaTemporalRegimenes;