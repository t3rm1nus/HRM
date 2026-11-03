import React, { useRef, useEffect, useState } from 'react';

export interface PanelMACDProps {
  datosCierres: { timestamp: string; cierre: number }[];
  timeframe: string;
  tema?: 'light' | 'dark';
}

export const PanelMACD: React.FC<PanelMACDProps> = ({
  datosCierres,
  timeframe,
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

    chart.applyOptions(theme);

    const series = chart.addLineSeries({
      color: '#6366F1',
      lineWidth: 1,
      title: 'MACD',
    });

    // MACD simplificado - línea señal y histograma
    const macdValores = datosCierres.map((d, i) => ({
      time: d.timestamp,
      value: 0, // Simplificado - en producción calcularía MACD real
    }));

    series.setData(datosCierres.map(d => ({ time: d.timestamp, value: 0 })));

    // Línea señal
    const signalSeries = chart.addLineSeries({
      color: '#EF4444',
      lineWidth: 1,
      title: 'Señal',
    });
    signalSeries.setData(datosCierres.map(d => ({ time: d.timestamp, value: 0 })));

    // Histograma
    const histogramSeries = chart.addHistogramSeries({
      color: '#10B981',
      lineWidth: 0,
    });
    histogramSeries.setData(datosCierres.map(d => ({ time: d.timestamp, value: 0 })));

    setIsMounted(true);

    return () => {
      chartRef.current?.remove();
    };
  }, [datosCierres, tema]);

  if (!isMounted) {
    return <div className="h-[150px] w-full" />;
  }

  return (
    <div className="p-4 border-t border-gray-200">
      <h3 className="text-xs font-medium text-gray-600 mb-2">MACD</h3>
      <div
        ref={chartRef}
        style={{ height: 150 }}
        className="h-[150px] w-full"
      />
      <p className="text-xs text-gray-500">Linea MACD | Señal | Histograma</p>
    </div>
  );
};

export default PanelMACD;