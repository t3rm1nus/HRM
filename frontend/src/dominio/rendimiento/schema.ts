import { z } from 'zod';

export interface Operacion {
  id: string;
  simbolo: string;
  regimen: 'PAPEL' | 'TESTNET' | 'LIVE';
  tipo: 'ingreso' | 'egreso' | 'transferencia';
  monto: number;
  fecha: string;
  comision: number;
  pnl?: number;
}

export interface ParametrosRendimiento {
  inicio: string;
  fin: string;
  simularRetornos?: boolean;
}

export interface ResultadoBins {
  rango: string;
  count: number;
}

export interface ResultadosRendimiento {
  equityFinal: number;
  equityInicial: number;
  drawdownMaximo: number;
  drawdownActual: number;
  pnLCiclo: number[];
  sharpeMovil: number[];
  sortinoMovil: number[];
  rentabilidadSimbolo: Map<string, number>;
  rentabilidadRegimen: Map<string, number>;
  rentabilidadModelo: Map<string, number>;
  mapaCalorDiaHora: number[][];
  comisionesAcumuladas: number;
  comparacionBuyAndHold: {
    retornoEstrategia: number;
    retornoBuyAndHold: number;
    outperformance: number;
  };
  totalOperaciones: number;
  comisionTotal: number;
}