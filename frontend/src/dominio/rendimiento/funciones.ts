import { Operacion, ParametrosRendimiento, ResultadoBins, ResultadosRendimiento } from './schema';

/** Calcula el equity a partir de operaciones y comisiones */
export function calcularEquity(
  operaciones: Operacion[],
  comisionPorOperacion: number = 0
): { equityFinal: number; equityInicial: number; comisionTotal: number } {
  let equity = 100;
  let comisionTotal = 0;

  for (const op of operaciones) {
    const montoNeto = op.monto - op.comision;
    equity += montoNeto;
    comisionTotal += op.comision;
  }

  return { equityFinal: equity, equityInicial: 100, comisionTotal };
}

/** Calcula drawdown máximo (pico a valle) */
export function calcularDrawdownSubmarino(
  valoresEquity: number[]
): { drawdownMaximo: number; drawdownActual: number; peakIndex: number } {
  if (valoresEquity.length < 2) {
    return { drawdownMaximo: 0, drawdownActual: 0, peakIndex: 0 };
  }

  let drawdownMaximo = 0;
  let peak = valoresEquity[0];
  let peakIndex = 0;

  for (let i = 1; i < valoresEquity.length; i++) {
    if (valoresEquity[i] > peak) {
      peak = valoresEquity[i];
      peakIndex = i;
    }
  }

  const peakValue = valoresEquity[peakIndex];
/** Calcula PnL por ciclo (agrupar cada 10 operaciones) */
export function calcularPnLCiclo(
  operaciones: Operacion[]
): number[] {
  const pnls: number[] = [];
  let pnlActual = 0;

  for (let i = 0; i < operaciones.length; i++) {
    const op = operaciones[i];
    pnlActual += (op.pnl ?? 0) - op.comision;

    if ((i + 1) % 10 === 0) {
      pnls.push(pnlActual);
      pnlActual = 0;
    }
  }

  if (pnlActual !== 0 || pnls.length === 0) {
    pnls.push(pnlActual);
  }
/** Calcula histograma de retornos */
export function calcularHistogramaRetornos(
  retornos: number[],
  numBins: number = 20
): Map<number, number> {
  const mapa = new Map<number, number>();

  if (retornos.length === 0) return mapa;

  const min = Math.min(...retornos);
  const max = Math.max(...retornos);
  const binSize = (max - min) / numBins;

  for (const r of retornos) {
    const binIndex = Math.floor((r - min) / binSize);
    const clave = min + binIndex * binSize;
    mapa.set(clave, (mapa.get(clave) || 0) + 1);
  }

  return mapa;
}
/** Calcula Sortino ratio móvil */
/** Calcula rentabilidad por símbolo */
export function calcularRentabilidadSimbolo(
  operaciones: Operacion[],
  simbolo: string
): number {
  const simOps = operaciones.filter(op => op.simbolo === simbolo);
  if (simOps.length === 0) return 0;

  const { equityFinal } = calcularEquity(simOps);
  return ((equityFinal - 100) / 100) * 100;
}

/** Calcula rentabilidad por régimen */
export function calcularRentabilidadRegimen(
  operaciones: Operacion[],
  regimen: 'PAPEL' | 'TESTNET' | 'LIVE'
): number {
  const simOps = operaciones.filter(op => op.regimen === regimen);
  if (simOps.length === 0) return 0;
  const { equityFinal } = calcularEquity(simOps);
  return ((equityFinal - 100) / 100) * 100;
}

/** Calcula rentabilidad por modelo (agrupando por clave) */
export function calcularRentabilidadModelo(
  operaciones: Operacion[],
  modeloKey: (op: Operacion) => string
): Map<string, number> {
  const mapa = new Map<string, number>();

  for (const op of operaciones) {
    const clave = modeloKey(op);
    const existente = mapa.get(clave) ?? 0;
    const { equityFinal } = calcularEquity([
      { ...op, monto: op.monto }
    ]);
    mapa.set(clave, existente + ((equityFinal - 100) / 100) * 100);
  }

  return mapa;
}

/** Calcula mapa de calor día-hora */
export function calcularMapaCalorDiaHora(
  operaciones: Operacion[],
  diasSemana: string[] = ['Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb', 'Dom']
): number[][] {
  const mapa = Array.from({ length: 7 }, () => Array(24).fill(0));

  for (const op of operaciones) {
    const fecha = new Date(op.fecha);
    const diaIdx = fecha.getDay();
    const hora = fecha.getHours();
    mapa[diaIdx][hora] += 1;
  }

  return mapa;
}

/** Calcula comisiones acumuladas */
export function calcularComisionesAcumuladas(
  operaciones: Operacion[]
): number {
  return operaciones.reduce((sum, op) => sum + op.comision, 0);
}

/** Calcula comparación con buy and hold */
export function calcularComparacionBuyAndHold(
  operaciones: Operacion[],
  precioInicial: number,
  precioFinal: number
): {
  retornoEstrategia: number;
  retornoBuyAndHold: number;
  outperformance: number;
} {
  const { equityFinal } = calcularEquity(operaciones);
  const retornoEstrategia = ((equityFinal - 100) / 100) * 100;
  const retornoBuyAndHold = ((precioFinal / precioInicial) - 1) * 100;
  const outperformance = retornoEstrategia - retornoBuyAndHold;

  return { retornoEstrategia, retornoBuyAndHold, outperformance };
}

/** Funciones puras exportadas */
export { calcularEquity, calcularDrawdownSubmarino, calcularPnLCiclo, calcularHistogramaRetornos, calcularSharpeMovil, calcularSortinoMovil, calcularRentabilidadSimbolo, calcularRentabilidadRegimen, calcularRentabilidadModelo, calcularMapaCalorDiaHora, calcularComisionesAcumuladas, calcularComparacionBuyAndHold };
export function calcularSortinoMovil(
  retornos: number[],
  ventana: number = 30,
  riskFreeRate: number = 0
): number[] {
  const sortino: number[] = [];

  for (let i = ventana - 1; i < retornos.length; i++) {
    const slice = retornos.slice(i - ventana + 1, i + 1);
    const media = slice.reduce((a, b) => a + b, 0) / ventana;

    // Desviación a la baja (solo retornos negativos)
    const negativos = slice.filter((r) => r < media);
    const downsideDeviation =
      negativos.length > 0
        ? Math.sqrt(
            negativos.reduce(
              (a, b) => a + Math.pow(b - media, 2),
              0
            ) / negativos.length
          )
        : 0;

    if (downsideDeviation === 0 || isNaN(downsideDeviation)) {
      sortino.push(0);
    } else {
      sortino.push((media - riskFreeRate) / downsideDeviation * Math.sqrt(252));
    }
  }

  while (sortino.length < retornos.length) {
    sortino.unshift(0);
  }

  return sortino;
}

  return pnls;
}
  const ultimoValor = valoresEquity[valoresEquity.length - 1];
  const drawdownMaximo = peakValue > 0 ? ((peakValue - Math.min(...valoresEquity)) / peakValue) * 100 : 0;
  const drawdownActual = peakValue > 0 ? ((peakValue - ultimoValor) / peakValue) * 100 : 0;

  return { drawdownMaximo, drawdownActual, peakIndex };
}