import type { Operacion } from './schema';

const CAPITAL_INICIAL_POR_DEFECTO = 100;
const FACTOR_ANUALIZACION = Math.sqrt(252);

export function calcularEquity(
  operaciones: Operacion[],
  comisionPorOperacion = 0,
): { equityFinal: number; equityInicial: number; comisionTotal: number } {
  let equityFinal = CAPITAL_INICIAL_POR_DEFECTO;
  let comisionTotal = 0;

  for (const operacion of operaciones) {
    const comision = operacion.comision + comisionPorOperacion;
    const resultado = operacion.pnl ?? (
      operacion.tipo === 'egreso' ? -operacion.monto : operacion.monto
    );
    equityFinal += resultado - comision;
    comisionTotal += comision;
  }

  return { equityFinal, equityInicial: CAPITAL_INICIAL_POR_DEFECTO, comisionTotal };
}

export function calcularDrawdownSubmarino(
  valoresEquity: number[],
): { drawdownMaximo: number; drawdownActual: number; peakIndex: number } {
  if (valoresEquity.length === 0) {
    return { drawdownMaximo: 0, drawdownActual: 0, peakIndex: 0 };
  }

  let pico = valoresEquity[0] ?? 0;
  let indicePico = 0;
  let peakIndex = 0;
  let drawdownMaximo = 0;

  valoresEquity.forEach((valor, indice) => {
    if (valor > pico) {
      pico = valor;
      indicePico = indice;
    }
    const drawdown = pico > 0 ? ((pico - valor) / pico) * 100 : 0;
    if (drawdown > drawdownMaximo) {
      drawdownMaximo = drawdown;
      peakIndex = indicePico;
    }
  });

  const ultimoValor = valoresEquity[valoresEquity.length - 1] ?? 0;
  const drawdownActual = pico > 0 ? ((pico - ultimoValor) / pico) * 100 : 0;
  return { drawdownMaximo, drawdownActual, peakIndex };
}

export function calcularPnLCiclo(operaciones: Operacion[], tamanoCiclo = 10): number[] {
  if (tamanoCiclo < 1) throw new Error('El tamaño del ciclo debe ser mayor que cero');

  const resultados: number[] = [];
  for (let inicio = 0; inicio < operaciones.length; inicio += tamanoCiclo) {
    const grupo = operaciones.slice(inicio, inicio + tamanoCiclo);
    resultados.push(
      grupo.reduce((total, operacion) => total + (operacion.pnl ?? 0) - operacion.comision, 0),
    );
  }
  return resultados;
}

export function calcularHistogramaRetornos(
  retornos: number[],
  numeroIntervalos = 20,
): Map<number, number> {
  const histograma = new Map<number, number>();
  if (retornos.length === 0) return histograma;
  if (!Number.isInteger(numeroIntervalos) || numeroIntervalos < 1) {
    throw new Error('El número de intervalos debe ser un entero positivo');
  }

  const minimo = Math.min(...retornos);
  const maximo = Math.max(...retornos);
  const amplitud = (maximo - minimo) / numeroIntervalos;

  for (const retorno of retornos) {
    const indice = amplitud === 0
      ? 0
      : Math.min(numeroIntervalos - 1, Math.floor((retorno - minimo) / amplitud));
    const inicioIntervalo = amplitud === 0 ? minimo : minimo + indice * amplitud;
    histograma.set(inicioIntervalo, (histograma.get(inicioIntervalo) ?? 0) + 1);
  }
  return histograma;
}

function calcularDesviacion(valores: number[]): number {
  if (valores.length < 2) return 0;
  const media = valores.reduce((total, valor) => total + valor, 0) / valores.length;
  const varianza = valores.reduce((total, valor) => total + (valor - media) ** 2, 0) / valores.length;
  return Math.sqrt(varianza);
}

export function calcularSharpeMovil(
  retornos: number[],
  ventana = 30,
  tasaLibreRiesgo = 0,
): number[] {
  if (ventana < 1) throw new Error('La ventana debe ser mayor que cero');
  return retornos.map((_, indice) => {
    if (indice + 1 < ventana) return 0;
    const muestra = retornos.slice(indice - ventana + 1, indice + 1);
    const media = muestra.reduce((total, valor) => total + valor, 0) / muestra.length;
    const desviacion = calcularDesviacion(muestra);
    return desviacion === 0 ? 0 : ((media - tasaLibreRiesgo) / desviacion) * FACTOR_ANUALIZACION;
  });
}

export function calcularSortinoMovil(
  retornos: number[],
  ventana = 30,
  tasaLibreRiesgo = 0,
): number[] {
  if (ventana < 1) throw new Error('La ventana debe ser mayor que cero');
  return retornos.map((_, indice) => {
    if (indice + 1 < ventana) return 0;
    const muestra = retornos.slice(indice - ventana + 1, indice + 1);
    const media = muestra.reduce((total, valor) => total + valor, 0) / muestra.length;
    const perdidas = muestra.filter((valor) => valor < tasaLibreRiesgo);
    if (perdidas.length === 0) return 0;
    const desviacionBajista = Math.sqrt(
      perdidas.reduce((total, valor) => total + (valor - tasaLibreRiesgo) ** 2, 0) / muestra.length,
    );
    return desviacionBajista === 0
      ? 0
      : ((media - tasaLibreRiesgo) / desviacionBajista) * FACTOR_ANUALIZACION;
  });
}

function calcularRentabilidad(operaciones: Operacion[]): number {
  if (operaciones.length === 0) return 0;
  const { equityFinal, equityInicial } = calcularEquity(operaciones);
  return ((equityFinal / equityInicial) - 1) * 100;
}

export function calcularRentabilidadSimbolo(operaciones: Operacion[], simbolo: string): number {
  return calcularRentabilidad(operaciones.filter((operacion) => operacion.simbolo === simbolo));
}

export function calcularRentabilidadRegimen(
  operaciones: Operacion[],
  regimen: Operacion['regimen'],
): number {
  return calcularRentabilidad(operaciones.filter((operacion) => operacion.regimen === regimen));
}

export function calcularRentabilidadModelo(
  operaciones: Operacion[],
  obtenerModelo: (operacion: Operacion) => string,
): Map<string, number> {
  const grupos = new Map<string, Operacion[]>();
  for (const operacion of operaciones) {
    const modelo = obtenerModelo(operacion);
    const grupo = grupos.get(modelo) ?? [];
    grupo.push(operacion);
    grupos.set(modelo, grupo);
  }
  return new Map([...grupos.entries()].map(([modelo, grupo]) => [modelo, calcularRentabilidad(grupo)]));
}

export function calcularMapaCalorDiaHora(operaciones: Operacion[]): number[][] {
  const mapa = Array.from({ length: 7 }, () => Array<number>(24).fill(0));
  for (const operacion of operaciones) {
    const fecha = new Date(operacion.fecha);
    if (Number.isNaN(fecha.getTime())) continue;
    const diaSemana = (fecha.getDay() + 6) % 7;
    const hora = fecha.getHours();
    mapa[diaSemana]![hora] = (mapa[diaSemana]![hora] ?? 0) + 1;
  }
  return mapa;
}

export function calcularComisionesAcumuladas(operaciones: Operacion[]): number {
  return operaciones.reduce((total, operacion) => total + operacion.comision, 0);
}

export function calcularComparacionBuyAndHold(
  operaciones: Operacion[],
  precioInicial: number,
  precioFinal: number,
): { retornoEstrategia: number; retornoBuyAndHold: number; outperformance: number } {
  if (precioInicial <= 0) throw new Error('El precio inicial debe ser mayor que cero');
  const retornoEstrategia = calcularRentabilidad(operaciones);
  const retornoBuyAndHold = ((precioFinal / precioInicial) - 1) * 100;
  return {
    retornoEstrategia,
    retornoBuyAndHold,
    outperformance: retornoEstrategia - retornoBuyAndHold,
  };
}
