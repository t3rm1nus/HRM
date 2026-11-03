import { http, HttpResponse } from 'msw';
import {
  VERSION_RESPUESTA,
  HEALTH_RESPUESTA,
  PORTFOLIO_RESPUESTA,
  USUARIOS_FIQUETAS,
  getTradesSinteticas,
  getTradesMock,
} from './fixtures';

// Estado mutable para simular cambios de modo
let estadoModoActual: 'papel' | 'testnet' | 'live' = 'papel';

export const setModoMock = (modo: 'papel' | 'testnet' | 'live'): void => {
  estadoModoActual = modo;
};

export const getModoMock = (): 'papel' | 'testnet' | 'live' => {
  return estadoModoActual;
};

// Respuestas dinámicas según el modo
const getStatusRespuesta = (): any => {
  const esLive = estadoModoActual === 'live';
  const esTestnet = estadoModoActual === 'testnet';
  const esPapel = estadoModoActual === 'papel';

  return {
    status: esPapel ? 'paused' : 'running',
    trading_active: !esPapel,
    cycle_count: esLive ? 1000000 : esTestnet ? 42 : 0,
    uptime: esLive ? 'sistema en producción' : esTestnet ? 'entorno de pruebas' : 'modo simulación',
  };
};

export const handlers = [
  http.get('http://localhost:8000/health', () => {
    return HttpResponse.json(HEALTH_RESPUESTA, { status: 200 });
  }),

  http.get('http://localhost:8000/status', () => {
    return HttpResponse.json(getStatusRespuesta(), { status: 200 });
  }),

  http.get('http://localhost:8000/portfolio', () => {
    return HttpResponse.json(PORTFOLIO_RESPUESTA, { status: 200 });
  }),

  http.get('http://localhost:8000/version', () => {
    return HttpResponse.json(VERSION_RESPUESTA, { status: 200 });
  }),

  http.get('http://localhost:8000/api/trades', () => {
    const trades = getTradesMock(20);
    return HttpResponse.json(trades, { status: 200 });
  }),

  http.get('http://localhost:8000/api/usuarios', () => {
    return HttpResponse.json(USUARIOS_FIQUETAS, { status: 200 });
  }),

  http.post('http://localhost:8000/api/auth/login', async ({ request }) => {
    const body = await request.json();
    if (body.email === 'admin@hrm.local' && body.password === 'admin123') {
      const token = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.mock-token';
      return HttpResponse.json(
        {
          token,
          usuario: USUARIOS_FIQUETAS[0],
        },
        { status: 200 }
      );
    }
    return HttpResponse.json(
      {
        detail: 'Credenciales inválidas',
        code: 'INVALID_CREDENTIALS',
      },
      { status: 401 }
    );
  }),

  http.get('http://localhost:8000/ws', ({ request }) => {
    const url = new URL(request.url);
    const events = url.searchParams.get('events');
    const simulaEventos = events === 'true';
    if (!simulaEventos) {
      return new HttpResponse('Streaming WebSocket events', { status: 200, headers: { 'Content-Type': 'text/plain' } });
    }
    return new Response(
      new ReadableStream({
        start(controller) {
          const interval = setInterval(() => {
            const evento = {
              tipo: 'trades_actualizadas',
              datos: getTradesSinteticas(),
              timestamp: new Date().toISOString(),
            };
            try {
              controller.enqueue(JSON.stringify(evento));
            } catch {
              clearInterval(interval);
              controller.close();
            }
          }, 5000);
        },
      }),
      {
        status: 101,
        headers: {
          'Content-Type': 'text/event-stream',
          Connection: 'keep-alive',
        },
      }
    );
  }),
];
