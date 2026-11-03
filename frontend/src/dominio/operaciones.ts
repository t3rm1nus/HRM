import {
  TipoOperacion,
  EstadoOperacion,
  TipoMoneda,
  TipoFecha,
} from '../dominio/tipos';

export interface OperacionInput {
  usuarioId: string;
  tipo: TipoOperacion;
  monto: number;
  fecha: string;
  descripcion: string;
}

export interface OperacionResultado {
  id: string;
  estado: EstadoOperacion;
  mensaje: string;
  montoConvertido: number;
  fechaFormateada: string;
}

export const ORIGEN_DEL_FECHA = {
  HOY: 'HOY',
  RELATIVO: 'RELATIVO',
} as const;

export const interpolarFecha = (tipo: TipoFecha, valor: string): string => {
  const fechaActual = new Date();
  const año = fechaActual.getFullYear();
  const mes = fechaActual.getMonth();
  const dia = fechaActual.getDate();

  const partes = valor.split('-');
  let anio: number;
  let mes: number;
  let dia: number;

  switch (tipo) {
    case 'Anio':
      anio = parseInt(valor, 10);
      mes = mes;
      dia = dia;
      break;
    case 'Mes':
      anio = año;
      mes = parseInt(valor, 10) - 1;
      dia = dia;
      break;
    case 'Dia':
      anio = año;
      mes = mes;
      dia = parseInt(valor, 10);
      break;
    default:
      anio = año;
      mes = mes;
      dia = dia;
  }

  return `${anio}-${String(mes + 1).padStart(2, '0')}-${String(dia).padStart(2, '0')}`;
};

export const proximaFecha = (tipo: TipoFecha, valor: string): string => {
  const fecha = new Date(interpolarFecha(tipo, valor));
  fecha.setDate(fecha.getDate() + 1);
  return fecha.toISOString().split('T')[0];
};

export const calcularIVA = (monto: number, tasa: number = 0.19): number => {
  return monto * tasa;
};

export const calcularTotalConIVA = (monto: number, tasa: number = 0.19): number => {
  return monto + calcularIVA(monto, tasa);
};

export const formatearMonto = (monto: number, simbolo: string = '$'): string => {
  return new Intl.NumberFormat('es-ES', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  }).format(monto);
};

export const obtenerLetraDelDia = (fecha: Date): string => {
  const dias = ['Domingo', 'Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado'];
  return dias[fecha.getDay()];
};

export const obtenerLetraDelMes = (fecha: Date): string => {
  const meses = [
    'Enero',
    'Febrero',
    'Marzo',
    'Abril',
    'Mayo',
    'Junio',
    'Julio',
    'Agosto',
    'Septiembre',
    'Octubre',
    'Noviembre',
    'Diciembre',
  ];
  return meses[fecha.getMonth()];
};

export const esEstadoValido = (estado: string): boolean => {
  const estadosValidos = Object.values(EstadoOperacion);
  return estadosValidos.includes(estado as EstadoOperacion);
};

export const esTipoOperacionValido = (tipo: string): boolean => {
  const tiposValidos = Object.values(TipoOperacion);
  return tiposValidos.includes(tipo as TipoOperacion);
};

export const obtenerColorEstado = (estado: EstadoOperacion): string => {
  switch (estado) {
    case EstadoOperacion.COMPLETADO:
      return 'bg-green-100 text-green-800';
    case EstadoOperacion.PROCESANDO:
      return 'bg-yellow-100 text-yellow-800';
    case EstadoOperacion.PENDIENTE:
      return 'bg-gray-100 text-gray-800';
    case EstadoOperacion.FALLIDO:
      return 'bg-red-100 text-red-800';
    default:
      return 'bg-gray-100 text-gray-800';
  }
};

export const obtenerColorTipo = (tipo: TipoOperacion): string => {
  switch (tipo) {
    case TipoOperacion.INGRESO:
      return 'bg-green-100 text-green-800';
    case TipoOperacion.EGRESO:
      return 'bg-red-100 text-red-800';
    case TipoOperacion.TRANSFERENCIA:
      return 'bg-blue-100 text-blue-800';
    default:
      return 'bg-gray-100 text-gray-800';
  }
};

export const ordenarOperaciones = (a: OperacionInput, b: OperacionInput): number => {
  const fechaA = new Date(a.fecha).getTime();
  const fechaB = new Date(b.fecha).getTime();
  return fechaB - fechaA;
};

export const filtrarPorRangoFecha = (
  operaciones: OperacionInput[],
  inicio: string,
  fin: string
): OperacionInput[] => {
  const fechaInicio = new Date(inicio).getTime();
  const fechaFin = new Date(fin).getTime();

  return operaciones.filter((op) => {
    const fechaOp = new Date(op.fecha).getTime();
    return fechaOp >= fechaInicio && fechaOp <= fechaFin;
  });
};

export const calcularSumaMonto = (operaciones: OperacionInput[]): number => {
  return operaciones.reduce((suma, op) => suma + op.monto, 0);
};

export const obtenerEstadisticas = (operaciones: OperacionInput[]): Estadisticas => {
  const total = calcularSumaMonto(operaciones);
  const count = operaciones.length;
  const exitosas = operaciones.filter((op) => op.estado === EstadoOperacion.COMPLETADO).length;
  return {
    totalOperaciones: count,
    operacionesExitosas: exitosas,
    montoTotal: total,
    tasaExitos: count > 0 ? (exitosas / count) * 100 : 0,
  };
};

export interface Estadisticas {
  totalOperaciones: number;
  operacionesExitosas: number;
  montoTotal: number;
  tasaExitos: number;
};