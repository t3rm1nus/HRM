import React from 'react';

export interface ComparadorNormalizadoProps {
  btcDatos: { timestamp: string; precio: number }[];
  ethDatos: { timestamp: string; precio: number }[];
  width?: number;
  height?: number;
  tema?: 'light' | 'dark';
}

export const ComparadorNormalizado: React.FC<ComparadorNormalizadoProps> = ({
  btcDatos,
  ethDatos,
  width = 400,
  height = 200,
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

    // Series BTC
    const btcSeries = chart.addLineSeries({
      color: '#F59E0B',
      lineWidth: 2,
      title: 'BTC',
    });
    btcSeries.setData(btcDatos.map(d => ({ time: d.timestamp, value: d.precio })));

    // Series ETH
    const ethSeries = chart.addLineSeries({
      color: '#3B82F6',
      lineWidth: 2,
      title: 'ETH',
    });
    ethSeries.setData(ethDatos.map(d => ({ time: d.timestamp, value: d.precio })));

    setIsMounted(true);

    return () => {
      chartRef.current?.remove();
    };
  }, [btcDatos, ethDatos, tema]);

  if (!isMounted) {
    return <div className="h-[200px] w-full" />;
  }

  return (
    <div className="rounded-md p-4 border border-gray-200">
      <h3 className="text-sm font-medium text-gray-700 mb-2">Comparación BTC/ETH</h3>
      <div
        ref={chartRef}
        style={{ width: width, height: height }}
        className="h-[200px] w-full"
      />
    </div>
  );
};

export default ComparadorNormalizado;