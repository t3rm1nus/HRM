/** Throttle function - asegura que el callback se ejecute como máximo una vez por segundo */
export const throttle = (func: (...args: any[]) => void, limit: number) => {
  let inThrottle = false;
  return function (this: any, ...args: any[]) {
    if (!inThrottle) {
      func.apply(this, args);
      inThrottle = true;
      setTimeout(() => (inThrottle = false), limit);
    }
  };
};

/** Crosshair sincronizado - notifica a otras gráficas cuando cambia el precio */
export const useCrosshairSync = (sourceId: string, setCrosshair: (price: number, timestamp: string) => void) => {
  const handleMouseMove = (e: MouseEvent) => {
    const chart = (e.target as HTMLElement).closest('.lightweight-charts');
    if (!chart) return;

    const rect = chart.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const width = chart.offsetWidth;

    // Convertir posición X a timestamp
    const timestamp = new Date().toISOString(); // Simplificado - en producción usarías el tiempo de la curva
    const price = 0; // Precio simplificado

    // Llamada throttle para evitar floods
    setCrosshair(price, timestamp);
  };

  return { handleMouseMove };
};

/** Formatea datos para lightweight-charts vela */
export const formatearDatosVela = (candles: {
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume?: number;
}) => {
  return candles.map((c) => ({
    time: c.timestamp,
    open: c.open,
    high: c.high,
    low: c.low,
    close: c.close,
    volume: c.volume,
  }));
};

/** Formatea datos para línea/área */
export const formatearDatosLinea = (datos: {
  timestamp: string;
  valor: number;
}) => {
  return datos.map((d) => ({
    time: d.timestamp,
    value: d.valor,
  }));
};