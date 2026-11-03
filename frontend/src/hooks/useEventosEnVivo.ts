import { useState, useCallback, useEffect, useRef } from 'react';
import { TIEMPO_REFRESCO_POR_DEFECTO } from './tiempos';

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
  onDisconnect?: (reason: string) => void;
  onEvent?: (evento: EventoEnVivo) => void;
  onReconnect?: (intentos: number) => void;
  validateEvent?: (data: unknown) => boolean;
}

const obtenerURLWS = (): string => {
  if (import.meta.env.VITE_USAR_MOCKS === 'true') {
    return 'ws://localhost:3001';
  }
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${protocol}//${window.location.hostname}:8000/ws`;
};

const calcularDelayReconexion = (intentos: number, base: number, maximo: number): number => {
  const delay = base * 2 ** intentos;
  return Math.min(delay, maximo);
};

export const usarEventosEnVivo = (config: WebSocketConfig) => {
  const [estado, setEstado] = useState<'conectando' | 'conectado' | 'desconectado'>('desconectado');
  const [intentosReconexion, setIntentosReconexion] = useState(0);
  const socketRef = useRef<WebSocket | null>(null);
  const intentosRef = useRef(0);
  const reconnectTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const cerrarConexion = useCallback(() => {
    if (socketRef.current) {
      socketRef.current.close();
      socketRef.current = null;
    }
    setEstado('desconectado');
    config.onDisconnect?.(intentosRef.current > 0 ? 'Reconexión' : 'Desconexión');
  }, [config]);

  const conectar = useCallback(() => {
    const url = config.url;
    try {
      const socket = new WebSocket(url);
      socketRef.current = socket;

      socket.onopen = () => {
        intentosRef.current = 0;
        setIntentosReconexion(0);
        setEstado('conectado');
        config.onConnect?.();
      };

      socket.onmessage = (evento) => {
        try {
          const datos = JSON.parse(evento.data);
          if (config.validateEvent?.(datos)) {
            config.onEvent?.({
              tipo: datos.tipo ?? 'generico',
              datos: datos.datos ?? datos,
              timestamp: new Date().toISOString(),
            });
          }
        } catch (error) {
          console.error('Error al parsear evento:', error);
        }
      };

      socket.onclose = () => {
        socketRef.current = null;
        if (estadoRef.current === 'conectado') {
          setEstado('conectando');
          intentosRef.current += 1;
          config.onReconnect?.(intentosRef.current);
          if (intentosRef.current <= config.maxReconnects ?? 5) {
            const delay = calcularDelayReconexion(
              intentosRef.current,
              config.heartbeatInterval ?? 1000,
              config.heartbeatInterval ?? 30000
            );
            reconnectTimeoutRef.current = setTimeout(conectar, delay);
          } else {
            setEstado('desconectado');
            config.onDisconnect?.('Máximo de reintentos alcanzado');
          }
        }
      };

      socket.onerror = () => {
        if (socket.readyState === WebSocket.CONNECTING) {
          socket.close();
        }
      };
    } catch (error) {
      console.error('Error al conectar WebSocket:', error);
      setEstado('desconectado');
      config.onDisconnect?.(error instanceof Error ? error.message : 'Error de conexión');
    }
  }, [config]);

  const estadoRef = useRef(estado);
  useEffect(() => {
    estadoRef.current = estado;
  });

  useEffect(() => {
    conectar();
    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      cerrarConexion();
    };
  }, [conectar, cerrarConexion, estado]);

  const enviar = useCallback(
    (datos: unknown) => {
      if (socketRef.current?.readyState === WebSocket.OPEN) {
        socketRef.current.send(JSON.stringify(datos));
      } else {
        const mensaje = { tipo: 'modo_sincrono', datos };
        config.onEvent?.({
          tipo: 'modo_sincrono',
          datos: mensaje,
          timestamp: new Date().toISOString(),
        });
      }
    },
    [config]
  );

  const forzarReconexion = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
    }
    cerrarConexion();
    conectar();
  }, [conectar, cerrarConexion]);

  return {
    estado,
    enviar,
    forzarReconexion,
    intentosReconexion: intentosRef.current,
    onDisconnect: cerrarConexion,
  };
};

export const obtenerURLWS = (): string => {
  const storage = localStorage;
  const usarMocks = storage.getItem('VITE_USAR_MOCKS') === 'true';
  if (usarMocks) {
    return 'ws://localhost:3001';
  }
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${protocol}//${window.location.hostname}:8000/ws`;
};