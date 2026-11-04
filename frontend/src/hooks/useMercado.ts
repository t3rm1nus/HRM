import { useCallback } from 'react';
import { useQuery, useQueryClient, type QueryClient } from '@tanstack/react-query';
import { fetchEstadoMercado, fetchVelas } from '../api/mercado';
import {
  fusionarEstadoConActualizacion,
  fusionarUltimaVela,
  interpretarActualizacion,
  type ActualizacionMercado,
  type EstadoMercado,
  type LoteVelas,
} from '../dominio/mercado';
import { obtenerURLWS, usarEventosEnVivo, type EventoEnVivo } from './useEventosEnVivo';

export const TIEMPO_REFRESCO_ESTADO_MERCADO = 15_000;
export const TIEMPO_REFRESCO_VELAS = 60_000;
const EVENTO_MERCADO = 'mercado';

const aplicarActualizacion = (cliente: QueryClient, actualizacion: ActualizacionMercado): void => {
  Object.entries(actualizacion.ultimas_velas).forEach(([simbolo, vela]) => {
    cliente.setQueryData<LoteVelas>(['mercado', 'velas', simbolo], (lote) =>
      lote === undefined ? lote : fusionarUltimaVela(lote, vela, actualizacion.frescura),
    );
  });
  cliente.setQueryData<EstadoMercado>(['mercado', 'estado'], (estado) =>
    estado === undefined ? estado : fusionarEstadoConActualizacion(estado, actualizacion),
  );
};

export const usarMercado = (simbolo: string) => {
  const cliente = useQueryClient();

  const estado = useQuery({
    queryKey: ['mercado', 'estado'],
    queryFn: fetchEstadoMercado,
    refetchInterval: TIEMPO_REFRESCO_ESTADO_MERCADO,
  });

  const velas = useQuery({
    queryKey: ['mercado', 'velas', simbolo],
    queryFn: () => fetchVelas(simbolo),
    enabled: simbolo !== '',
    refetchInterval: TIEMPO_REFRESCO_VELAS,
  });

  const alRecibirEvento = useCallback((evento: EventoEnVivo): void => {
    if (evento.tipo !== EVENTO_MERCADO) return;
    const actualizacion = interpretarActualizacion(evento.datos);
    if (actualizacion !== null) aplicarActualizacion(cliente, actualizacion);
  }, [cliente]);

  // Al (re)conectar se recarga todo: cubre las velas perdidas durante la desconexión.
  const alConectar = useCallback((): void => {
    void cliente.invalidateQueries({ queryKey: ['mercado'] });
  }, [cliente]);

  const conexion = usarEventosEnVivo({
    url: obtenerURLWS(),
    onEvent: alRecibirEvento,
    onConnect: alConectar,
  });

  return { estado, velas, conexion: conexion.estado };
};
