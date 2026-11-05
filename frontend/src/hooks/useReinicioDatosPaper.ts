import { useState } from 'react';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { reiniciarDatosPaper } from '../api/modoSesion';

export const useReinicioDatosPaper = () => {
  const queryClient = useQueryClient();
  const [confirmacionPendiente, setConfirmacionPendiente] = useState(false);
  const [mensajeError, setMensajeError] = useState<string | null>(null);

  const mutacion = useMutation({
    mutationFn: (confirmacion: string) => reiniciarDatosPaper(confirmacion),
    onSuccess: async () => {
      setMensajeError(null);
      await queryClient.invalidateQueries({ queryKey: ['modo-sesion'] });
    },
    onError: (error: Error) => {
      setMensajeError(error.message);
    },
  });

  const solicitarReinicio = (): void => {
    setMensajeError(null);
    setConfirmacionPendiente(true);
  };

  const confirmarReinicio = async (): Promise<void> => {
    try {
      await mutacion.mutateAsync('CONFIRMAR');
      setConfirmacionPendiente(false);
    } catch {
      // El detalle del error ya se expone mediante onError.
    }
  };

  const cancelarReinicio = (): void => {
    setConfirmacionPendiente(false);
  };

  return {
    confirmacionPendiente,
    reinicioEnCurso: mutacion.isPending,
    mensajeError,
    solicitarReinicio,
    confirmarReinicio,
    cancelarReinicio,
  };
};
