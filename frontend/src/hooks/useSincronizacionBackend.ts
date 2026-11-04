import { useEffect } from 'react';
import { derivarEstadoConexion } from '../dominio/conexion';
import { ModoAplicacion, useModoStore } from '../estado/storeModo';
import { usarEstado } from './useEstadoQuery';

const MODOS_CONFIRMABLES: Record<string, ModoAplicacion> = {
  paper: ModoAplicacion.PAPERA,
  live: ModoAplicacion.LIVE,
};

export const traducirModoDelBackend = (modoBackend: string | undefined): ModoAplicacion | null =>
  (modoBackend !== undefined ? MODOS_CONFIRMABLES[modoBackend] : undefined) ?? null;

export const usarSincronizacionBackend = (): void => {
  const { datos, error } = usarEstado();
  const actualizarEstadoConexion = useModoStore((estado) => estado.actualizarEstadoConexion);
  const establecerCiclo = useModoStore((estado) => estado.establecerCiclo);
  const confirmarModoDelBackend = useModoStore((estado) => estado.confirmarModoDelBackend);
  const invalidarModoConfirmado = useModoStore((estado) => estado.invalidarModoConfirmado);

  const hayError = error !== null;
  const hayDatos = datos !== undefined;
  const cicloActual = datos?.cycle_count;
  const modoBackend = hayError ? undefined : datos?.mode;

  useEffect(() => {
    actualizarEstadoConexion(derivarEstadoConexion({ hayError, hayDatos }));
  }, [hayError, hayDatos, actualizarEstadoConexion]);

  useEffect(() => {
    if (cicloActual !== undefined) establecerCiclo(cicloActual);
  }, [cicloActual, establecerCiclo]);

  useEffect(() => {
    const modo = traducirModoDelBackend(modoBackend);
    if (modo === null) invalidarModoConfirmado();
    else confirmarModoDelBackend(modo);
  }, [modoBackend, confirmarModoDelBackend, invalidarModoConfirmado]);
};
