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
