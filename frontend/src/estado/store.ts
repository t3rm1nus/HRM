import { create } from 'zustand';
import { TipoOperacion, EstadoOperacion } from '../dominio/tipos';

export interface Usuario {
  id: string;
  nombre: string;
  email: string;
  rol: Rol;
  activo: boolean;
}

export enum Rol {
  ADMIN = 'admin',
  OPERADOR = 'operador',
  ANALISTA = 'analista',
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

export interface OperacionInput {
  usuarioId: string;
  tipo: TipoOperacion;
  monto: number;
  fecha: string;
  descripcion: string;
}

export interface OperacionActiva {
  id: string;
  estado: EstadoOperacion;
  mensaje: string;
  montoConvertido: number;
  fechaFormateada: string;
}

export interface EstadoAplicacion {
  usuario: Usuario | null;
  token: string | null;
  operaciones: Operacion[];
  operacionActiva: OperacionActiva | null;
  isLoading: boolean;
  error: string | null;
  filtro: {
    fechaInicio: string;
    fechaFin: string;
    tipoOperacion: TipoOperacion | 'todos';
    estado: EstadoOperacion | 'todos';
  };
}

export interface Acciones {
  iniciarSesion: (email: string, password: string) => Promise<void>;
  cerrarSesion: () => void;
  obtenerOperaciones: (page: number) => Promise<void>;
  crearOperacion: (dados: OperacionInput) => Promise<void>;
  eliminarOperacion: (id: string) => Promise<void>;
  activarOperacion: (id: string) => void;
  completarOperacion: (id: string, resultado: string) => void;
  filtrarPorRangoFecha: (inicio: string, fin: string) => void;
  filtrarPorTipo: (tipo: TipoOperacion | 'todos') => void;
  filtrarPorEstado: (estado: EstadoOperacion | 'todos') => void;
}

export type StoreType = EstadoAplicacion & Acciones;