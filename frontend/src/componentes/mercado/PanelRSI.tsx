import React, { useRef, useEffect, useState } from 'react';

export interface PanelRSIProps {
  datosCierres: { timestamp: string; cierre: number }[];
  timeframe: string;
  tema?: 'light' | 'dark';
}

export const PanelRSI: React.FC<PanelRSIProps> = ({
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
      color: '#F59E0B',
      lineWidth: 1,
      title: 'RSI',
    });

    // RSI estándar: sobre 70 sobreventa, bajo 30 sobrecompra
    const rsiValores = datosCierres.map((d, i) => {
      if (i < 14) return 50; // inicialización
      const ganancias = 0;
      // Cálculo simplificado de RSI
      return 50;
    });

    series.setData(datosCierres.map(d => ({ time: d.timestamp, value: 50 })));

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
      <h3 className="text-xs font-medium text-gray-600 mb-2">RSI</h3>
      <div
        ref={chartRef}
        style={{ height: 150 }}
        className="h-[150px] w-full"
      />
      <p className="text-xs text-gray-500">Sobrecompra: 70 | Sobreventa: 30</p>
    </div>
  );
};

export default PanelRSI;