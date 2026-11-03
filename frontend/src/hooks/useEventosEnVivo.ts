import { useCallback, useEffect, useRef, useState } from 'react';

export interface EventoEnVivo {
  tipo: string;
  datos: unknown;
  timestamp: string;
}

export interface WebSocketConfig {
  url: string;
  heartbeatInterval?: number;
  maxReconnects?: number;
  onConnect?: () => void;
  onDisconnect?: (motivo: string) => void;
  onEvent?: (evento: EventoEnVivo) => void;
  onReconnect?: (intentos: number) => void;
  validateEvent?: (datos: unknown) => boolean;
}

type EstadoConexion = 'conectando' | 'conectado' | 'desconectado';

export const obtenerURLWS = (): string => {
  if (import.meta.env.VITE_USAR_MOCKS === 'true') return 'ws://localhost:3001';
  const protocolo = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${protocolo}//${window.location.hostname}:8000/ws`;
};

const calcularEsperaReconexion = (intento: number, base: number, maximo: number): number =>
  Math.min(base * 2 ** Math.max(0, intento - 1), maximo);

export const usarEventosEnVivo = (configuracion: WebSocketConfig) => {
  const [estado, setEstado] = useState<EstadoConexion>('desconectado');
  const [intentosReconexion, setIntentosReconexion] = useState(0);
  const [versionConexion, setVersionConexion] = useState(0);
  const socketRef = useRef<WebSocket | null>(null);
  const configuracionRef = useRef(configuracion);
  configuracionRef.current = configuracion;

  useEffect(() => {
    let desmontado = false;
    let temporizador: ReturnType<typeof setTimeout> | null = null;
    let socket: WebSocket | null = null;
    let intentos = 0;

    const conectar = (): void => {
      if (desmontado) return;
      setEstado('conectando');

      try {
        socket = new WebSocket(configuracion.url);
        socketRef.current = socket;

        socket.onopen = () => {
          if (desmontado) return socket?.close();
          intentos = 0;
          setIntentosReconexion(0);
          setEstado('conectado');
          configuracionRef.current.onConnect?.();
        };

        socket.onmessage = (mensaje) => {
          try {
            const datos: unknown = JSON.parse(String(mensaje.data));
            if (configuracionRef.current.validateEvent &&
                !configuracionRef.current.validateEvent(datos)) return;

            const evento = typeof datos === 'object' && datos !== null
              ? datos as Record<string, unknown>
              : {};
            configuracionRef.current.onEvent?.({
              tipo: typeof evento.tipo === 'string' ? evento.tipo : 'generico',
              datos: evento.datos ?? datos,
              timestamp: new Date().toISOString(),
            });
          } catch {
            configuracionRef.current.onDisconnect?.('Se recibió un mensaje WebSocket no válido');
          }
        };

        socket.onerror = () => {
          if (socket?.readyState === WebSocket.CONNECTING) socket.close();
        };

        socket.onclose = () => {
          if (socketRef.current === socket) socketRef.current = null;
          if (desmontado) return;

          const maximo = configuracionRef.current.maxReconnects ?? 5;
          if (intentos >= maximo) {
            setEstado('desconectado');
            configuracionRef.current.onDisconnect?.('Máximo de reconexiones alcanzado');
            return;
          }

          intentos += 1;
          setIntentosReconexion(intentos);
          setEstado('conectando');
          configuracionRef.current.onReconnect?.(intentos);

          const esperaBase = configuracionRef.current.heartbeatInterval ?? 1000;
          temporizador = setTimeout(
            conectar,
            calcularEsperaReconexion(intentos, esperaBase, 30_000),
          );
        };
      } catch (error) {
        setEstado('desconectado');
        configuracionRef.current.onDisconnect?.(
          error instanceof Error ? error.message : 'No se pudo abrir WebSocket',
        );
      }
    };

    conectar();
    return () => {
      desmontado = true;
      if (temporizador) clearTimeout(temporizador);
      socket?.close(1000, 'Componente desmontado');
      socketRef.current = null;
    };
  }, [configuracion.url, configuracion.maxReconnects, configuracion.heartbeatInterval, versionConexion]);

  const enviar = useCallback((datos: unknown): boolean => {
    const socket = socketRef.current;
    if (!socket || socket.readyState !== WebSocket.OPEN) return false;
    socket.send(JSON.stringify(datos));
    return true;
  }, []);

  const forzarReconexion = useCallback(() => {
    socketRef.current?.close(4000, 'Reconexión solicitada');
    setVersionConexion((version) => version + 1);
  }, []);

  const cerrarConexion = useCallback(() => {
    socketRef.current?.close(1000, 'Cierre solicitado');
    socketRef.current = null;
    setEstado('desconectado');
  }, []);

  return {
    estado,
    enviar,
    forzarReconexion,
    intentosReconexion,
    onDisconnect: cerrarConexion,
  };
};
