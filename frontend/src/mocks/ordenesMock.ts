import type { LadoFiltrable, LoteOrdenes, Orden } from '../dominio';
import { TRADES_FIQUETAS, type TradeFiqueta } from './fixtures';

export interface ParametrosOrdenesMock {
  pagina: number;
  tamano: number;
  simbolo: string | null;
  lado: LadoFiltrable | null;
}

const convertirTradeEnOrden = (trade: TradeFiqueta): Orden => ({
  order_id: trade.operation_id,
  simbolo: trade.symbol,
  lado: trade.side,
  cantidad: trade.quantity,
  precio_llenado: trade.price,
  comision: trade.commission,
  status: 'filled',
  timestamp: trade.timestamp,
  estrategia: 'mock',
  metadata: { ciclo: null, motivo: null, condiciones_mercado: {} },
});

export const ORDENES_FIQUETAS: Orden[] = TRADES_FIQUETAS.map(convertirTradeEnOrden);

const ordenarMasRecientesPrimero = (ordenes: Orden[]): Orden[] =>
  [...ordenes].sort((a, b) => b.timestamp.localeCompare(a.timestamp));

const filtrarPorSimbolo = (ordenes: Orden[], simbolo: string | null): Orden[] =>
  simbolo ? ordenes.filter((orden) => orden.simbolo.toLowerCase() === simbolo.toLowerCase()) : ordenes;

const filtrarPorLado = (ordenes: Orden[], lado: LadoFiltrable | null): Orden[] =>
  lado ? ordenes.filter((orden) => orden.lado === lado) : ordenes;

export const crearLoteOrdenesMock = (parametros: ParametrosOrdenesMock): LoteOrdenes => {
  const filtradas = filtrarPorLado(
    filtrarPorSimbolo(ordenarMasRecientesPrimero(ORDENES_FIQUETAS), parametros.simbolo),
    parametros.lado,
  );
  const paginaTotal = Math.max(1, Math.ceil(filtradas.length / parametros.tamano));
  const inicio = (parametros.pagina - 1) * parametros.tamano;
  return {
    total: filtradas.length,
    ordenes: filtradas.slice(inicio, inicio + parametros.tamano),
    has_more: parametros.pagina < paginaTotal,
    pagina: parametros.pagina,
    pagina_total: paginaTotal,
  };
};

const leerNumero = (valor: string | null, porDefecto: number): number => {
  const numero = Number.parseInt(valor ?? '', 10);
  return Number.isFinite(numero) && numero > 0 ? numero : porDefecto;
};

const leerLado = (valor: string | null): LadoFiltrable | null =>
  valor === 'buy' || valor === 'sell' ? valor : null;

export const leerParametrosOrdenes = (url: URL): ParametrosOrdenesMock => ({
  pagina: leerNumero(url.searchParams.get('pagina'), 1),
  tamano: leerNumero(url.searchParams.get('tamano'), 50),
  simbolo: url.searchParams.get('simbolo'),
  lado: leerLado(url.searchParams.get('lado')),
});
