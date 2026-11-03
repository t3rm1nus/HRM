import type { Usuario } from '../dominio';

export interface HealthCheckRespuesta {
  status: 'ok' | 'error';
  timestamp: string;
  version: string;
}

export interface StatusRespuesta {
  status: 'running' | 'paused';
  trading_active: boolean;
  cycle_count: number;
  uptime: string;
}

export interface PortfolioRespuesta {
  btc_balance: number;
  eth_balance: number;
  usdt_balance: number;
  total_nav: number;
  initial_capital: number;
}

export interface VersionRespuesta {
  app_version: string;
  framework: string;
  python_version: string;
}

export interface TradeFiqueta {
  operation_id: string;
  timestamp: string;
  symbol: string;
  side: 'buy' | 'sell';
  quantity: number;
  price: number;
  total_value: number;
  commission: number;
  net_value: number;
  exchange: string;
  tax_year: number;
}

export const VERSION_RESPUESTA: VersionRespuesta = {
  app_version: '2.0.0',
  framework: 'FastAPI',
  python_version: '3.10+',
};

export const HEALTH_RESPUESTA: HealthCheckRespuesta = {
  status: 'ok',
  timestamp: new Date().toISOString(),
  version: '2.0.0',
};

export const STATUS_RESPUESTA: StatusRespuesta = {
  status: 'running',
  trading_active: true,
  cycle_count: 42,
  uptime: 'sistema activo',
};

export const PORTFOLIO_RESPUESTA: PortfolioRespuesta = {
  btc_balance: 0.002345,
  eth_balance: 0.004567,
  usdt_balance: 3000.0,
  total_nav: 3000.0,
  initial_capital: 3000.0,
};

export const USUARIOS_FIQUETAS: Usuario[] = [
  {
    id: 'd3b07384-d477-448d-882b-8c7f2b6d4c9a',
    nombre: 'Ana Martinez',
    email: 'admin@hrm.local',
    rol: 'admin',
    activo: true,
  },
  {
    id: 'e4c18a9e-6f12-42a3-9b78-3d8f1a2e4d5b',
    nombre: 'Carlos Ruiz',
    email: 'operador@hrm.local',
    rol: 'operador',
    activo: true,
  },
  {
    id: 'f5d29ba4-7a23-53b4-8c99-4e9a2b3f5e6c',
    nombre: 'Laura Gomez',
    email: 'analista@hrm.local',
    rol: 'analista',
    activo: false,
  },
];

export const TRADES_FIQUETAS: TradeFiqueta[] = [
  {
    operation_id: 'BTCUSDT_2026-02-12T16:29:21.561813_2837773664064',
    timestamp: '2026-02-12T16:29:21.561813',
    symbol: 'BTCUSDT',
    side: 'buy',
    quantity: 0.001,
    price: 50000.0,
    total_value: 50.0,
    commission: 0.25,
    net_value: 49.75,
    exchange: 'Binance',
    tax_year: 2026,
  },
  {
    operation_id: 'BTCUSDT_2026-02-12T20:18:48.090567_2949490830144',
    timestamp: '2026-02-12T20:18:48.090567',
    symbol: 'BTCUSDT',
    side: 'buy',
    quantity: 0.0015201964945180954,
    price: 65813.860485,
    total_value: 100.05,
    commission: 0.1,
    net_value: 99.95,
    exchange: 'Binance',
    tax_year: 2026,
  },
  {
    operation_id: 'BTCUSDT_2026-02-12T20:45:17.057581_1509573291328',
    timestamp: '2026-02-12T20:45:17.057581',
    symbol: 'BTCUSDT',
    side: 'buy',
    quantity: 0.0015253044965366437,
    price: 65593.46033999999,
    total_value: 100.05,
    commission: 0.1,
    net_value: 99.95,
    exchange: 'Binance',
    tax_year: 2026,
  },
  {
    operation_id: 'BTCUSDT_2026-02-12T23:06:28.016876_1509678766464',
    timestamp: '2026-02-12T23:06:28.016876',
    symbol: 'BTCUSDT',
    side: 'buy',
    quantity: 0.0030089674540828577,
    price: 66476.271525,
    total_value: 200.02,
    commission: 0.2,
    net_value: 199.82,
    exchange: 'Binance',
    tax_year: 2026,
  },
  {
    operation_id: 'BTCUSDT_2026-02-12T23:06:28.097314_1507573389120',
    timestamp: '2026-02-12T23:06:28.097314',
    symbol: 'BTCUSDT',
    side: 'sell',
    quantity: 0.0015187821710044979,
    price: 66409.828475,
    total_value: 100.86,
    commission: 0.1,
    net_value: 100.76,
    exchange: 'Binance',
    tax_year: 2026,
  },
  {
    operation_id: 'ETHUSDT_2026-02-12T23:06:28.100163_1507573399232',
    timestamp: '2026-02-12T23:06:28.100163',
    symbol: 'ETHUSDT',
    side: 'buy',
    quantity: 0.07715269122806953,
    price: 1948.65384,
    total_value: 150.34,
    commission: 0.15,
    net_value: 150.19,
    exchange: 'Binance',
    tax_year: 2026,
  },
  {
    operation_id: 'BTCUSDT_2026-02-12T23:07:06.844993_1507573393856',
    timestamp: '2026-02-12T23:07:06.844993',
    symbol: 'BTCUSDT',
    side: 'buy',
    quantity: 0.0011277277546427942,
    price: 66490.73875499998,
    total_value: 74.98,
    commission: 0.07,
    net_value: 74.91,
    exchange: 'Binance',
    tax_year: 2026,
  },
  {
    operation_id: 'BTCUSDT_2026-02-12T23:07:42.979117_1507573387264',
    timestamp: '2026-02-12T23:07:42.979117',
    symbol: 'BTCUSDT',
    side: 'buy',
    quantity: 0.0005631381254987494,
    price: 66476.531655,
    total_value: 37.44,
    commission: 0.04,
    net_value: 37.4,
    exchange: 'Binance',
    tax_year: 2026,
  },
  {
    operation_id: 'BTCUSDT_2026-02-12T23:08:19.098892_1507573535488',
    timestamp: '2026-02-12T23:08:19.098892',
    symbol: 'BTCUSDT',
    side: 'buy',
    quantity: 0.0002813977554484799,
    price: 66417.192,
    total_value: 18.69,
    commission: 0.02,
    net_value: 18.67,
    exchange: 'Binance',
    tax_year: 2026,
  },
  {
    operation_id: 'BTCUSDT_2026-02-12T23:08:55.225635_1507573535168',
    timestamp: '2026-02-12T23:08:55.225635',
    symbol: 'BTCUSDT',
    side: 'buy',
    quantity: 0.00014043079049368116,
    price: 66444.13546499998,
    total_value: 9.33,
    commission: 0.01,
    net_value: 9.32,
    exchange: 'Binance',
    tax_year: 2026,
  },
];