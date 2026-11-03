export interface Usuario {
  id: string;
  nombre: string;
  email: string;
  rol: Rol;
  activo: boolean;
}

export interface Operacion {
  id: string;
  usuario: Usuario;
  tipo: TipoOperacion;
  monto: number;
  fecha: string;
  estado: EstadoOperacion;
  descripcion: string;
}

export interface CrearOperacionDto {
  usuarioId: string;
  tipo: TipoOperacion;
  monto: number;
  descripcion: string;
}

export interface Reportes {
  totalOperaciones: number;
  operacionesExitosas: number;
  montoTotal: number;
  operacionesPorDia: OperacionDia[];
}

export interface OperacionDia {
  fecha: string;
  monto: number;
  count: number;
}

export enum Rol {
  ADMIN = 'admin',
  OPERADOR = 'operador',
  ANALISTA = 'analista',
}

export enum TipoOperacion {
  INGRESO = 'ingreso',
  EGRESO = 'egreso',
  TRANSFERENCIA = 'transferencia',
}

export enum EstadoOperacion {
  PENDIENTE = 'pendiente',
  PROCESANDO = 'procesando',
  COMPLETADO = 'completado',
  FALLIDO = 'fallido',
}

export enum TipoMoneda {
  USD = 'usd',
  EUR = 'eur',
  PEN = 'PEN',
}

export enum TipoFecha {
  HOY = 'HOY',
  RELATIVO = 'RELATIVO',
}