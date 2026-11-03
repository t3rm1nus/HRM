import React, { useRef, useEffect, useState } from 'react';
import { LightweightCharts } from 'lightweight-charts';

export interface GraficaMercadoProps {
  simbolo: string;
  velas: { timestamp: string; open: number; high: number; low: number; close: number; volume: number }[];
  timeframe: string;
  capas?: {
    ma50?: boolean;
    ma200?: boolean;
    bollinger?: boolean;
    soportes?: boolean;
    resistencias?: boolean;
  };
  tema?: 'light' | 'dark';
}

export const GraficaMercado: React.FC<GraficaMercadoProps> = ({
  simbolo,
  velas,
  timeframe,
  capas = {},
  tema = 'dark',
}) => {
  const chartRef = useRef<LightweightCharts.Widget | null>(null);
  const [isMounted, setIsMounted] = useState(false);

  useEffect(() => {
    if (!chartRef.current) return;

    const chart = chartRef.current.renderer();
    const theme = tema === 'dark' ? {
      background: { type: 'solid', color: '#111827' },
      grid: { vertLine: { color: '#374151' }, horzLine: { color: '#374151' } },
      rightPriceScale: { borderVisible: false, backgroundColor: '#111827', textColor: '#F9FAFB' },
      timeScale: { borderVisible: false, timeVisible: true, secondsVisible: false },
      crosshair: { mode: 'visible', color: '#FFFFFF', vertLine: { color: '#FFFFFF' }, horzLine: { color: '#FFFFFF' } },
      layout: { background: { type: 'solid', color: '#111827' }, textColor: '#F9FAFB' },
    } : {
      background: { type: 'solid', color: '#FFFFFF' },
      grid: { vertLine: { color: '#E5E7EB' }, horzLine: { color: '#E5E7EB' } },
      rightPriceScale: { borderVisible: true, backgroundColor: '#FFFFFF', textColor: '#111827' },
      timeScale: { borderVisible: true, timeVisible: true, secondsVisible: false },
      crosshair: { mode: 'visible', color: '#000000', vertLine: { color: '#000000' }, horzLine: { color: '#000000' } },
      layout: { background: { type: 'solid', color: '#FFFFFF' }, textColor: '#111827' },
    };

    chart.applyOptions(theme);

    // Series de velas
    const series = chart.addCandlestickSeries({
      upColor: '#10B981',
      downColor: '#EF4444',
      borderUpColor: '#10B981',
      borderDownColor: '#EF4444',
      wickUpColor: '#10B981',
      wickDownColor: '#EF4444',
    });

    const formattedData = velas.map((v) => ({
      time: v.timestamp,
      open: v.open,
      high: v.high,
      low: v.low,
      close: v.close,
      volume: v.volume,
    }));

    series.setData(formattedData);

    // Capas MA50
    if (capas.ma50) {
      const ma50 = chart.addMovingAverage({ period: 50, color: '#F59E0B' });
      // Data would need to be calculated - simplified here
    }

    // Capas MA200
    if (capas.ma200) {
      const ma200 = chart.addMovingAverage({ period: 200, color: '#3B82F6' });
    }

    // Capas Bollinger
    if (capas.bollinger) {
      // Bollinger Bands simplificado
    }

    // Soportes y resistencias
    if (capas.soportes || capas.resistencias) {
      // Líneas de soporte/resistencia
    }

    setIsMounted(true);

    return () => {
      chartRef.current?.remove();
    };
  }, [velas, timeframe, capas, tema]);

  if (!isMounted) {
    return <div className="h-[400px] w-full" />;
  }

  return (
    <div
      ref={chartRef}
      style={{ width: '100%', height: '400px' }}
      className="border rounded-md"
    />
  );
};

export default GraficaMercado;