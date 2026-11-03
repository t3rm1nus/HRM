import {
  TipoOperacion,
  EstadoOperacion,
  TipoFecha,
} from './tipos';

export interface OperacionInput {
  usuarioId: string;
  tipo: TipoOperacion;
  monto: number;
  fecha: string;
  descripcion: string;
  estado?: EstadoOperacion;
}

export interface OperacionResultado {
  id: string;
  estado: EstadoOperacion;
  mensaje: string;
  montoConvertido: number;
  fechaFormateada: string;
}

export const ORIGEN_DEL_FECHA = {
  HOY: TipoFecha.HOY,
  RELATIVO: TipoFecha.RELATIVO,
} as const;

export const interpolarFecha = (tipo: TipoFecha, valor: string): string => {
  const fecha = new Date();
  if (tipo === TipoFecha.RELATIVO) {
    const dias = Number.parseInt(valor, 10);
    if (!Number.isFinite(dias)) throw new Error('El desplazamiento debe ser un número de días');
    fecha.setDate(fecha.getDate() + dias);
  } else if (valor.trim()) {
    const fechaIndicada = new Date(valor);
    if (Number.isNaN(fechaIndicada.getTime())) throw new Error('La fecha indicada no es válida');
    return fechaIndicada.toISOString().slice(0, 10);
  }
  return fecha.toISOString().slice(0, 10);
};

export const proximaFecha = (tipo: TipoFecha, valor: string): string => {
  const fecha = new Date(interpolarFecha(tipo, valor));
  fecha.setDate(fecha.getDate() + 1);
  return fecha.toISOString().slice(0, 10);
};

export const calcularIVA = (monto: number, tasa = 0.19): number => monto * tasa;
export const calcularTotalConIVA = (monto: number, tasa = 0.19): number => monto + calcularIVA(monto, tasa);

export const formatearMonto = (monto: number, simbolo = '€'): string =>
  `${simbolo}${new Intl.NumberFormat('es-ES', {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  }).format(monto)}`;

export const obtenerLetraDelDia = (fecha: Date): string =>
  ['Domingo', 'Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado'][fecha.getDay()] ?? '';

export const obtenerLetraDelMes = (fecha: Date): string =>
  [
    'Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio',
    'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre',
  ][fecha.getMonth()] ?? '';

export const esEstadoValido = (estado: string): boolean =>
  Object.values(EstadoOperacion).includes(estado as EstadoOperacion);

export const esTipoOperacionValido = (tipo: string): boolean =>
  Object.values(TipoOperacion).includes(tipo as TipoOperacion);

export const obtenerColorEstado = (estado: EstadoOperacion): string => {
  switch (estado) {
    case EstadoOperacion.COMPLETADO: return 'bg-green-100 text-green-800';
    case EstadoOperacion.PROCESANDO: return 'bg-yellow-100 text-yellow-800';
    case EstadoOperacion.PENDIENTE: return 'bg-gray-100 text-gray-800';
    case EstadoOperacion.FALLIDO: return 'bg-red-100 text-red-800';
    default: return 'bg-gray-100 text-gray-800';
  }
};

export const obtenerColorTipo = (tipo: TipoOperacion): string => {
  switch (tipo) {
    case TipoOperacion.INGRESO: return 'bg-green-100 text-green-800';
    case TipoOperacion.EGRESO: return 'bg-red-100 text-red-800';
    case TipoOperacion.TRANSFERENCIA: return 'bg-blue-100 text-blue-800';
    default: return 'bg-gray-100 text-gray-800';
  }
};

export const ordenarOperaciones = (a: OperacionInput, b: OperacionInput): number =>
  new Date(b.fecha).getTime() - new Date(a.fecha).getTime();

export const filtrarPorRangoFecha = (
  operaciones: OperacionInput[],
  inicio: string,
  fin: string,
): OperacionInput[] => {
  const fechaInicio = new Date(inicio).getTime();
  const fechaFin = new Date(fin).getTime();
  if (!Number.isFinite(fechaInicio) || !Number.isFinite(fechaFin) || fechaInicio > fechaFin) return [];
  return operaciones.filter((operacion) => {
    const fechaOperacion = new Date(operacion.fecha).getTime();
    return fechaOperacion >= fechaInicio && fechaOperacion <= fechaFin;
  });
};

export const calcularSumaMonto = (operaciones: OperacionInput[]): number =>
  operaciones.reduce((suma, operacion) => suma + operacion.monto, 0);

export interface Estadisticas {
  totalOperaciones: number;
  operacionesExitosas: number;
  montoTotal: number;
  tasaExitos: number;
}

export const obtenerEstadisticas = (operaciones: OperacionInput[]): Estadisticas => {
  const totalOperaciones = operaciones.length;
  const operacionesExitosas = operaciones.filter(
    (operacion) => operacion.estado === EstadoOperacion.COMPLETADO,
  ).length;
  return {
    totalOperaciones,
    operacionesExitosas,
    montoTotal: calcularSumaMonto(operaciones),
    tasaExitos: totalOperaciones > 0 ? (operacionesExitosas / totalOperaciones) * 100 : 0,
  };
};
