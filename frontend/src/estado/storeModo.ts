import { create } from 'zustand';

export enum ModoAplicacion {
  PAPERA = 'papel',
  TESTNET = 'testnet',
  LIVE = 'live',
}

export type ModoValor = `${ModoAplicacion}`;

export interface IStoreModo {
  modo: ModoAplicacion;
  modoConfirmado: boolean;
  umbralLatencia: number;
  latenciaMs: number;
  estadoConexion: 'conectado' | 'desconectado' | 'pendiente';
  enPausa: boolean;
  deteniendo: boolean;
  cicloCount: number;
  alarmeActivado: boolean;
  operador: string;

  establecerModo: (modo: ModoAplicacion) => void;
  confirmarModoDelBackend: (modo: ModoAplicacion) => void;
  invalidarModoConfirmado: () => void;
  rotarModo: () => void;
  actualizarEstadoConexion: (estado: string, latencia?: number) => void;
  pausar: () => void;
  reanudar: () => void;
  iniciarDetencionEmergencia: () => void;
  cancelarDetencionEmergencia: () => void;
  confirmarDetencionEmergencia: () => void;
  incrementarCiclo: () => void;
  establecerCiclo: (ciclo: number) => void;
  activarAlarme: () => void;
  desactivarAlarme: () => void;
  setOperador: (nombre: string) => void;
}

export const useModoStore = create<IStoreModo>((set, get) => ({
  modo: ModoAplicacion.PAPERA,
  modoConfirmado: false,
  umbralLatencia: 500,
  latenciaMs: -1,
  estadoConexion: 'pendiente',
  enPausa: false,
  deteniendo: false,
  cicloCount: 0,
  alarmeActivado: false,
  operador: '',

  establecerModo: (modo) => set({ modo, latenciaMs: -1, estadoConexion: 'pendiente' }),

  confirmarModoDelBackend: (modo) => set({ modo, modoConfirmado: true }),

  invalidarModoConfirmado: () => set({ modoConfirmado: false }),

  rotarModo: () => {
    const actual = get().modo;
    let siguiente: ModoAplicacion;
    switch (actual) {
      case ModoAplicacion.PAPERA:
        siguiente = ModoAplicacion.TESTNET;
        break;
      case ModoAplicacion.TESTNET:
        siguiente = ModoAplicacion.LIVE;
        break;
      default:
        siguiente = ModoAplicacion.PAPERA;
        break;
    }
    get().establecerModo(siguiente);
  },

  actualizarEstadoConexion: (estado, latencia = -1) => {
    set({
      estadoConexion: estado as IStoreModo['estadoConexion'],
      latenciaMs: latencia,
    });
  },

  pausar: () => set({ enPausa: true, deteniendo: true, alarmeActivado: true }),

  reanudar: () => set({ enPausa: false, deteniendo: false, alarmeActivado: false }),

  iniciarDetencionEmergencia: () => set({ deteniendo: true, enPausa: true }),

  cancelarDetencionEmergencia: () => set({ deteniendo: false }),

  confirmarDetencionEmergencia: () => {
    set({ deteniendo: true, enPausa: true, alarmeActivado: true });
  },

  incrementarCiclo: () => set((state) => ({ cicloCount: state.cicloCount + 1 })),

  establecerCiclo: (ciclo) => set({ cicloCount: ciclo }),

  activarAlarme: () => set({ alarmeActivado: true }),

  desactivarAlarme: () => set({ alarmeActivado: false }),

  setOperador: (nombre) => set({ operador: nombre }),
}));
