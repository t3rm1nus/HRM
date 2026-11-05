import { useState } from 'react';
import type { ModoSesion } from '../dominio';
import { useModoSesion } from './useModoSesion';

export const useCambioModoSesion = () => {
  const { modoActual, cargando, cambiarModo, refetch } = useModoSesion();
  const [confirmacionPendiente, setConfirmacionPendiente] = useState(false);

  const aplicarModo = async (modo: ModoSesion): Promise<void> => {
    try {
      await cambiarModo(modo);
      await refetch();
    } catch (error) {
      console.error(`Error al cambiar al modo de sesión "${modo}":`, error);
    }
  };

  const solicitarCambio = (modo: ModoSesion): void => {
    if (modo === 'limpia') {
      setConfirmacionPendiente(true);
      return;
    }
    void aplicarModo(modo);
  };

  const confirmarCambio = async (): Promise<void> => {
    setConfirmacionPendiente(false);
    await aplicarModo('limpia');
  };

  const cancelarCambio = (): void => setConfirmacionPendiente(false);

  return {
    modoConfigurado: modoActual?.modo_configurado ?? null,
    reinicioPendiente: modoActual?.reinicio_pendiente ?? false,
    cargando,
    confirmacionPendiente,
    solicitarCambio,
    confirmarCambio,
    cancelarCambio,
  };
};
