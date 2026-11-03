export const ICONO_PAGINAS = {
  HOME: '🏠',
  PROFILE: '👤',
  CONFIG: '⚙️',
  REPORTES: '📊',
  OPERACIONES: '📋',
  PORTAFOLIO: '💼',
  FISCAL: '💵',
  SALUD: '🏥',
  SISTEMA: '🖥️',
  ERROR_404: '❌',
  NOT_FOUND: '❌',
  EMERGENCIA: '🛑',
  REFRESCAR: '🔄',
  TEMA: '☀️',
  NOTIFICACIONES: '📶',
  ERROR: '⚠️',
} as const;

export interface PaginaConfig {
  ruta: string;
  nombre: string;
  icono: string;
  seccion: 'principal' | 'configuracion' | 'operaciones' | 'informes' | 'comunidad';
}

export const PAGINAS: PaginaConfig[] = [
  { ruta: '/', nombre: 'Panel', icono: ICONO_PAGINAS.HOME, seccion: 'principal' },
  { ruta: '/mercado', nombre: 'Mercado', icono: '📈', seccion: 'principal' },
  { ruta: '/operaciones', nombre: 'Operaciones', icono: ICONO_PAGINAS.OPERACIONES, seccion: 'operaciones' },
  { ruta: '/cartera', nombre: 'Cartera', icono: ICONO_PAGINAS.PORTAFOLIO, seccion: 'principal' },
  { ruta: '/rendimiento', nombre: 'Rendimiento', icono: ICONO_PAGINAS.REPORTES, seccion: 'informes' },
  { ruta: '/configuracion', nombre: 'Configuración', icono: ICONO_PAGINAS.CONFIG, seccion: 'configuracion' },
  { ruta: '/senales', nombre: 'Senales', icono: ICONO_PAGINAS.HOME, seccion: 'principal' },
  { ruta: '/estrategia', nombre: 'Estrategia', icono: ICONO_PAGINAS.HOME, seccion: 'principal' },
  { ruta: '/riesgo', nombre: 'Riesgo', icono: ICONO_PAGINAS.HOME, seccion: 'principal' },
  { ruta: '/autoaprendizaje', nombre: 'Autoaprendizaje', icono: ICONO_PAGINAS.HOME, seccion: 'principal' },
  { ruta: '/modelos', nombre: 'Modelos', icono: ICONO_PAGINAS.HOME, seccion: 'principal' },
  { ruta: '/backtesting', nombre: 'Backtesting', icono: ICONO_PAGINAS.HOME, seccion: 'principal' },
  { ruta: '/fiscal', nombre: 'Fiscal', icono: ICONO_PAGINAS.FISCAL, seccion: 'principal' },
  { ruta: '/salud', nombre: 'Salud', icono: ICONO_PAGINAS.SALUD, seccion: 'principal' },
  { ruta: '/sistema', nombre: 'Sistema', icono: ICONO_PAGINAS.SISTEMA, seccion: 'principal' },
];
