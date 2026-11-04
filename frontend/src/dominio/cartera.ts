export interface Asignacion {
  activo: string;
  objetivo: number;
  real: number;
}

export const calcularDesviacion = (asignaciones: Asignacion[]): number => {
  if (asignaciones.length === 0) return 0;
  const suma = asignaciones.reduce((acc, a) => acc + Math.abs(a.objetivo - a.real), 0);
  return suma / asignaciones.length;
};

export interface Rentabilidad {
  pnl: number;
  porcentaje: number | null;
}

export const calcularRentabilidad = (valorActual: number, capitalInicial: number): Rentabilidad => {
  const pnl = valorActual - capitalInicial;
  const porcentaje = capitalInicial > 0 ? (pnl / capitalInicial) * 100 : null;
  return { pnl, porcentaje };
};
