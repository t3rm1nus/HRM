import { setupServer } from 'msw/node';
import { handlers, setModoMock, getModoMock } from './handlers';

export const mswServer = setupServer(...handlers);

export const mswSetup = (): void => {
  if (import.meta.env.VITE_USAR_MOCKS === 'true') {
    mswServer.listen();
  }
};

export const mswTeardown = (): void => {
  if (import.meta.env.VITE_USAR_MOCKS === 'true') {
    mswServer.close();
  }
};

// API para cambiar el modo de simulación
export { setModoMock, getModoMock };
