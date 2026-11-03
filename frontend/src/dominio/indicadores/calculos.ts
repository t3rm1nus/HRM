export interface PuntoIndicador {
  timestamp: string;
  valor: number;
}

export function calcularMediaMovil(valores: number[], periodo: number): (number | null)[] {
  if (!Number.isInteger(periodo) || periodo <= 0) return valores.map(() => null);
  return valores.map((_, indice) => {
    if (indice + 1 < periodo) return null;
    const ventana = valores.slice(indice + 1 - periodo, indice + 1);
    return ventana.reduce((suma, valor) => suma + valor, 0) / periodo;
  });
}

export function calcularRsi(valores: number[], periodo = 14): (number | null)[] {
  const resultado: (number | null)[] = valores.map(() => null);
  if (!Number.isInteger(periodo) || periodo <= 0 || valores.length <= periodo) return resultado;
  let ganancias = 0;
  let perdidas = 0;
  for (let indice = 1; indice <= periodo; indice += 1) {
    const cambio = (valores[indice] ?? 0) - (valores[indice - 1] ?? 0);
    ganancias += Math.max(cambio, 0);
    perdidas += Math.max(-cambio, 0);
  }
  let gananciaMedia = ganancias / periodo;
  let perdidaMedia = perdidas / periodo;
  resultado[periodo] = perdidaMedia === 0 ? (gananciaMedia === 0 ? 50 : 100) : gananciaMedia === 0 ? 0 : 100 - 100 / (1 + gananciaMedia / perdidaMedia);
  for (let indice = periodo + 1; indice < valores.length; indice += 1) {
    const cambio = (valores[indice] ?? 0) - (valores[indice - 1] ?? 0);
    gananciaMedia = (gananciaMedia * (periodo - 1) + Math.max(cambio, 0)) / periodo;
    perdidaMedia = (perdidaMedia * (periodo - 1) + Math.max(-cambio, 0)) / periodo;
    resultado[indice] = perdidaMedia === 0 ? (gananciaMedia === 0 ? 50 : 100) : gananciaMedia === 0 ? 0 : 100 - 100 / (1 + gananciaMedia / perdidaMedia);
  }
  return resultado;
}

export function calcularMacd(valores: number[], periodoRapido = 12, periodoLento = 26, periodoSenal = 9) {
  const periodosValidos = [periodoRapido, periodoLento, periodoSenal].every((periodo) => Number.isInteger(periodo) && periodo > 0) && periodoRapido < periodoLento;
  if (!periodosValidos) {
    const vacio = valores.map(() => null);
    return { macd: vacio, senal: [...vacio], histograma: [...vacio] };
  }
  const calcularEma = (datos: number[], periodo: number): (number | null)[] => {
    const salida: (number | null)[] = datos.map(() => null);
    if (datos.length < periodo || periodo <= 0) return salida;
    const factor = 2 / (periodo + 1);
    let ema = datos.slice(0, periodo).reduce((suma, valor) => suma + valor, 0) / periodo;
    salida[periodo - 1] = ema;
    for (let indice = periodo; indice < datos.length; indice += 1) {
      ema = (datos[indice] ?? 0) * factor + ema * (1 - factor);
      salida[indice] = ema;
    }
    return salida;
  };
  const rapido = calcularEma(valores, periodoRapido);
  const lento = calcularEma(valores, periodoLento);
  const macd = valores.map((_, indice) => {
    const valorRapido = rapido[indice] ?? null;
    const valorLento = lento[indice] ?? null;
    return valorRapido === null || valorLento === null ? null : valorRapido - valorLento;
  });
  const puntosValidos = macd.map((valor, indice) => ({ valor, indice })).filter((punto): punto is { valor: number; indice: number } => punto.valor !== null);
  const senalCompacta = calcularEma(puntosValidos.map((punto) => punto.valor), periodoSenal);
  const senal: (number | null)[] = valores.map(() => null);
  puntosValidos.forEach((punto, indice) => { senal[punto.indice] = senalCompacta[indice] ?? null; });
  const histograma = macd.map((valor, indice) => {
    const valorSenal = senal[indice] ?? null;
    return valor === null || valorSenal === null ? null : valor - valorSenal;
  });
  return { macd, senal, histograma };
}
