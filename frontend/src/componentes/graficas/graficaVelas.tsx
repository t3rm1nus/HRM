import React, { useRef, useEffect, useState } from 'react';
import { LightweightCharts } from 'lightweight-charts';

export interface GraficaVelasProps {
  simbolo: string;
  candles: { timestamp: string; open: number; high: number; low: number; close: number; volume?: number }[];
  width?: number;
  height?: number;
  theme?: 'light' | 'dark';
  showVolume?: boolean;
  /** Crosshair sincronizado entre gráficas */
  crosshairSource?: string;
  /** Callback cuando cambia el precio seleccionado */
  onPriceChange?: (price: number, timestamp: string) => void;
}

export const GraficaVelas = ({
  simbolo,
  candles,
  width = 800,
  height = 400,
  theme = 'dark',
  showVolume = true,
  crosshairSource,
  onPriceChange,
}: GraficaVelasProps) => {
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
    };

    chart.applyOptions(themeConfig);

    const plotData = candles.map((c) => ({
      time: c.timestamp,
      open: c.open,
      high: c.high,
      low: c.low,
      close: c.close,
      volume: c.volume,
    }));

    chart.setMarkup({
      // Add volume bars if showVolume
    });

    chart.set(plotData);

    setIsMounted(true);

    return () => {
      chartRef.current?.remove();
    };
  }, [candles, theme, width, height, crosshairSource, onPriceChange]);

  if (!isMounted) {
    return <div className="h-[400px] w-full" />;
  }

  return (
    <div
      ref={chartRef}
      style={{ width: width, height: height }}
      className="border rounded-md"
    />
  );
};

export default GraficaVelas;