export type EstadoConexionBackend = 'pendiente' | 'conectado' | 'desconectado';

export interface ResultadoConsulta {
  hayError: boolean;
  hayDatos: boolean;
}

export const derivarEstadoConexion = ({ hayError, hayDatos }: ResultadoConsulta): EstadoConexionBackend => {
  if (hayError) return 'desconectado';
  if (hayDatos) return 'conectado';
  return 'pendiente';
};
