export type FuncionConArgumentos<Argumentos extends unknown[]> = (...argumentos: Argumentos) => void;

export function throttle<Argumentos extends unknown[]>(funcion: FuncionConArgumentos<Argumentos>, limiteMs: number): FuncionConArgumentos<Argumentos> {
  let bloqueado = false;
  return (...argumentos: Argumentos) => {
    if (bloqueado) return;
    funcion(...argumentos);
    bloqueado = true;
    setTimeout(() => { bloqueado = false; }, limiteMs);
  };
}

export function useCrosshairSync(_sourceId: string, setCrosshair: (precio: number, timestamp: string) => void) {
  const handleMouseMove = (evento: MouseEvent): void => {
    const elemento = evento.target instanceof HTMLElement ? evento.target.closest<HTMLElement>('[data-chart-point]') : null;
    const precio = Number(elemento?.dataset.price);
    const timestamp = elemento?.dataset.timestamp;
    if (Number.isFinite(precio) && timestamp) setCrosshair(precio, timestamp);
  };
  return { handleMouseMove };
}

export interface DatosVela {
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume?: number;
}

export function formatearDatosVela(candles: readonly DatosVela[]) {
  return candles.map((vela) => ({ ...vela }));
}

export function formatearDatosLinea(datos: readonly { timestamp: string; valor: number }[]) {
  return datos.map((dato) => ({ time: dato.timestamp, value: dato.valor }));
}
