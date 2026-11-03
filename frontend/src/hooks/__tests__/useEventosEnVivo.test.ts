import { renderHook, act, waitFor } from '@testing-library/react';
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { usarEventosEnVivo, obtenerURLWS } from '../useEventosEnVivo';

describe('obtenerURLWS', () => {
  it('debe devolver ws://localhost:3001 cuando VITE_USAR_MOCKS=true', () => {
    vi.stubGlobal('import.meta', {
      env: { VITE_USAR_MOCKS: 'true' },
    });

    const url = obtenerURLWS();
    expect(url).toBe('ws://localhost:3001');
  });

  it('debe devolver ws://localhost:8000/ws cuando VITE_USAR_MOCKS no está definido', () => {
    vi.stubGlobal('import.meta', {
      env: { VITE_USAR_MOCKS: 'false' },
    });

    const url = obtenerURLWS();
    expect(url).toBe('ws://localhost:8000/ws');
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });
});

describe('usarEventosEnVivo', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    localStorage.clear();
    Object.defineProperty(window, 'WebSocket', {
      value: class MockWebSocket {
        static CONNECTING = 0;
        static OPEN = 1;
        static CLOSING = 2;
        static CLOSED = 3;

        readyState = MockWebSocket.OPEN;
        onopen: ((event: Event) => void) | null = null;
        onmessage: ((event: MessageEvent) => void) | null = null;
        onclose: ((event: CloseEvent) => void) | null = null;
        onerror: ((event: Event) => void) | null = null;

        constructor(url: string) {
          this.url = url;
        }

        send(data: string): void {
          if (this.readyState === MockWebSocket.OPEN) {
            this.onmessage?.({ data } as MessageEvent);
          }
        }

        close(): void {
          this.readyState = MockWebSocket.CLOSED;
          this.onclose?.({ code: 1000, reason: '', wasClean: true } as CloseEvent);
        }
      },
    });
  });

  afterEach(() => {
    vi.restoreAllMocks();
    delete (window as unknown as { WebSocket: unknown }).WebSocket;
  });

  it('debe conectar al WebSocket y establecer estado conectado', () => {
    const mockOnConnect = vi.fn();
    const mockOnEvent = vi.fn();

    const { result } = renderHook(() =>
      usarEventosEnVivo({
        url: 'ws://localhost:3001',
        onConnect: mockOnConnect,
        onEvent: mockOnEvent,
      })
    );

    expect(result.current.estado).toBe('conectado');
    expect(mockOnConnect).toHaveBeenCalled();
  });

  it('debe enviar datos mediante el método enviar', () => {
    const mockOnEvent = vi.fn();

    const { result } = renderHook(() =>
      usarEventosEnVivo({
        url: 'ws://localhost:3001',
        onEvent: mockOnEvent,
      })
    );

    act(() => {
      result.current.enviar({ tipo: 'test', datos: { value: 1 } });
    });

    expect(mockOnEvent).toHaveBeenCalledWith(
      expect.objectContaining({
        tipo: 'test',
        timestamp: expect.any(String),
      })
    );
  });

  it('debe cerrar la conexión al desmontar', () => {
    const { unmount } = renderHook(() =>
      usarEventosEnVivo({
        url: 'ws://localhost:3001',
      })
    );

    expect(result.current.estado).toBe('conectado');

    unmount();

    expect(result.current.estado).toBe('desconectado');
  });

  it('debe manejar reintentos de reconexión', () => {
    const mockOnReconnect = vi.fn();
    const { result, rerender } = renderHook((props) =>
      usarEventosEnVivo({ ...props, maxReconnects: 1 }),
      { initialProps: { url: 'ws://localhost:3001' } }
    );

    expect(result.current.estado).toBe('conectado');

    act(() => {
      result.current.forzarReconexion();
    });

    expect(mockOnReconnect).not.toHaveBeenCalled();

    rerender({
      url: 'ws://localhost:3001',
      maxReconnects: 1,
      onReconnect: mockOnReconnect,
    });

    expect(mockOnReconnect).toHaveBeenCalled();
  });

  it('debe validar eventos con la función de validación', () => {
    const mockOnEvent = vi.fn();
    let receivedEvent: { tipo: string; datos: unknown; timestamp: string } | null = null;

    const validateEvent = vi.fn((data: unknown) => {
      return typeof (data as { tipo?: string }).tipo === 'string';
    });

    const { result } = renderHook(() =>
      usarEventosEnVivo({
        url: 'ws://localhost:3001',
        validateEvent,
        onEvent: mockOnEvent,
      })
    );

    act(() => {
      result.current.enviar({ tipo: 'test', datos: { value: 1 } });
    });

    expect(validateEvent).toHaveBeenCalled();
    expect(receivedEvent).toBeNull();
  });
});