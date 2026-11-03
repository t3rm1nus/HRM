import { useEffect, useRef } from 'react';
import { CandlestickSeries, ColorType, createChart, HistogramSeries, type UTCTimestamp } from 'lightweight-charts';

export interface GraficaVelasProps {
  simbolo: string;
  candles: { timestamp: string; open: number; high: number; low: number; close: number; volume?: number }[];
  width?: number;
  height?: number;
  theme?: 'light' | 'dark';
  showVolume?: boolean;
  crosshairSource?: string;
  onPriceChange?: (price: number, timestamp: string) => void;
}

function convertirTiempo(timestamp: string): UTCTimestamp {
  const milisegundos = new Date(timestamp).getTime();
  if (!Number.isFinite(milisegundos)) throw new Error(`Timestamp de vela no válido: ${timestamp}`);
  return Math.floor(milisegundos / 1000) as UTCTimestamp;
}

export const GraficaVelas = ({ simbolo, candles, width, height = 400, theme = 'dark', showVolume = true, onPriceChange }: GraficaVelasProps) => {
  const contenedorRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const contenedor = contenedorRef.current;
    if (!contenedor) return undefined;

    const oscuro = theme === 'dark';
    const grafica = createChart(contenedor, {
      width: contenedor.clientWidth,
      height,
      layout: {
        background: { type: ColorType.Solid, color: oscuro ? '#111827' : '#FFFFFF' },
        textColor: oscuro ? '#F9FAFB' : '#111827',
      },
      grid: {
        vertLines: { color: oscuro ? '#374151' : '#E5E7EB' },
        horzLines: { color: oscuro ? '#374151' : '#E5E7EB' },
      },
      rightPriceScale: { borderVisible: false },
      timeScale: { borderVisible: false, timeVisible: true, secondsVisible: false },
    });

    const velas = grafica.addSeries(CandlestickSeries, {
      upColor: '#10B981', downColor: '#EF4444', borderUpColor: '#10B981',
      borderDownColor: '#EF4444', wickUpColor: '#10B981', wickDownColor: '#EF4444',
    });
    velas.setData(candles.map((vela) => ({
      time: convertirTiempo(vela.timestamp), open: vela.open, high: vela.high,
      low: vela.low, close: vela.close,
    })));

    if (showVolume) {
      const volumen = grafica.addSeries(HistogramSeries, {
        priceFormat: { type: 'volume' }, priceScaleId: '',
      });
      volumen.priceScale().applyOptions({ scaleMargins: { top: 0.82, bottom: 0 } });
      volumen.setData(candles.filter((vela) => vela.volume !== undefined).map((vela) => ({
        time: convertirTiempo(vela.timestamp), value: vela.volume ?? 0,
        color: vela.close >= vela.open ? '#10B98166' : '#EF444466',
      })));
    }

    grafica.timeScale().fitContent();
    if (onPriceChange) {
      grafica.subscribeCrosshairMove((evento) => {
        const punto = evento.seriesData.get(velas);
        if (punto && 'close' in punto && typeof evento.time === 'number') {
          onPriceChange(punto.close, new Date(evento.time * 1000).toISOString());
        }
      });
    }

    const observador = new ResizeObserver((entradas) => {
      const ancho = entradas[0]?.contentRect.width;
      if (ancho && ancho > 0) grafica.applyOptions({ width: ancho });
    });
    observador.observe(contenedor);

    return () => {
      observador.disconnect();
      grafica.remove();
    };
  }, [candles, height, theme, showVolume, onPriceChange]);

  return (
    <section className="w-full rounded-md border border-gray-200 p-4">
      <h3 className="mb-2 text-sm font-medium">{simbolo}</h3>
      <div ref={contenedorRef} className="max-w-full" style={{ width: width ? `${width}px` : '100%', height }} />
    </section>
  );
};

export default GraficaVelas;
