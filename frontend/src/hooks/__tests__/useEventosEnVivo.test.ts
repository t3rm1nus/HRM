import { renderHook, act, waitFor } from '@testing-library/react';
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { usarEventosEnVivo, obtenerURLWS } from '../useEventosEnVivo';

const URL_PRUEBA = 'ws://localhost:3001';

class WebSocketSimulado {
  static CONNECTING = 0;
  static OPEN = 1;
  static CLOSING = 2;
  static CLOSED = 3;
  static instancias: WebSocketSimulado[] = [];

  readyState = WebSocketSimulado.CONNECTING;
  onopen: (() => void) | null = null;
  onmessage: ((evento: { data: string }) => void) | null = null;
  onclose: (() => void) | null = null;
  onerror: (() => void) | null = null;
  close = vi.fn((): void => this.cerrarSiEstaAbierto());

  constructor(public url: string) {
    WebSocketSimulado.instancias.push(this);
    queueMicrotask(() => this.abrir());
  }

  send(datos: string): void {
    this.onmessage?.({ data: datos });
  }

  private abrir(): void {
    this.readyState = WebSocketSimulado.OPEN;
    this.onopen?.();
  }

  private cerrarSiEstaAbierto(): void {
    if (this.readyState === WebSocketSimulado.CLOSED) return;
    this.readyState = WebSocketSimulado.CLOSED;
    this.onclose?.();
  }
}

const renderizarHookConectado = async (configuracion: Parameters<typeof usarEventosEnVivo>[0]) => {
  const resultado = renderHook(() => usarEventosEnVivo(configuracion));
  await waitFor(() => expect(resultado.result.current.estado).toBe('conectado'));
  return resultado;
};

describe('obtenerURLWS', () => {
  afterEach(() => {
    vi.unstubAllEnvs();
  });

  it('debe devolver ws://localhost:3001 cuando VITE_USAR_MOCKS=true', () => {
    vi.stubEnv('VITE_USAR_MOCKS', 'true');
    expect(obtenerURLWS()).toBe(URL_PRUEBA);
  });

  it('debe devolver ws://localhost:8000/ws cuando VITE_USAR_MOCKS no es true', () => {
    vi.stubEnv('VITE_USAR_MOCKS', 'false');
    expect(obtenerURLWS()).toBe('ws://localhost:8000/ws');
  });
});

describe('usarEventosEnVivo', () => {
  beforeEach(() => {
    WebSocketSimulado.instancias = [];
    vi.stubGlobal('WebSocket', WebSocketSimulado);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('debe conectar al WebSocket y establecer estado conectado', async () => {
    const alConectar = vi.fn();

    await renderizarHookConectado({ url: URL_PRUEBA, onConnect: alConectar });

    expect(alConectar).toHaveBeenCalledTimes(1);
  });

  it('debe enviar datos mediante el método enviar', async () => {
    const alRecibirEvento = vi.fn();
    const { result } = await renderizarHookConectado({ url: URL_PRUEBA, onEvent: alRecibirEvento });

    act(() => {
      result.current.enviar({ tipo: 'test', datos: { value: 1 } });
    });

    expect(alRecibirEvento).toHaveBeenCalledWith(
      expect.objectContaining({ tipo: 'test', timestamp: expect.any(String) }),
    );
  });

  it('debe cerrar la conexión al desmontar', async () => {
    const { unmount } = await renderizarHookConectado({ url: URL_PRUEBA });
    const [socket] = WebSocketSimulado.instancias;

    unmount();

    expect(socket.close).toHaveBeenCalledWith(1000, 'Componente desmontado');
  });

  it('debe manejar reintentos de reconexión', async () => {
    const alReconectar = vi.fn();
    const { result } = await renderizarHookConectado({
      url: URL_PRUEBA,
      maxReconnects: 1,
      onReconnect: alReconectar,
    });

    await act(async () => {
      result.current.forzarReconexion();
    });

    expect(alReconectar).toHaveBeenCalledWith(1);
  });

  it('debe descartar eventos que no superan la función de validación', async () => {
    const alRecibirEvento = vi.fn();
    const validarEvento = vi.fn(() => false);
    const { result } = await renderizarHookConectado({
      url: URL_PRUEBA,
      validateEvent: validarEvento,
      onEvent: alRecibirEvento,
    });

    act(() => {
      result.current.enviar({ tipo: 'test', datos: { value: 1 } });
    });

    expect(validarEvento).toHaveBeenCalled();
    expect(alRecibirEvento).not.toHaveBeenCalled();
  });
});
