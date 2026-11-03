// Constantes comunes del frontend

export const RUTAS = {
  INICIO: '/',
  MERCADO: '/mercado',
  RENDIMIENTO: '/rendimiento',
  OPERACIONES: '/operaciones',
  CARTERA: '/cartera',
  SEÑALES: '/senales',
  ESTRATEGIA: '/estrategia',
  RIESGO: '/riesgo',
  AUTOAPRENDIZAJE: '/autoaprendizaje',
  MODELOS: '/modelos',
  BACKTESTING: '/backtesting',
  CONFIGURACION: '/configuracion',
  FISCAL: '/fiscal',
  SALUD: '/salud',
  SISTEMA: '/sistema',
  ERROR_404: '/404',
} as const;

export const COLORES = {
  PRIMARIO: '#2563EB',
  SECUNDARIO: '#1D4ED8',
  EXITO: '#16A34A',
  ADVERTENCIA: '#F59E0B',
  ERROR: '#DC2626',
  ASCENSO: '#22C55E',
  DESCENSO: '#EF4444',
  FRAME: '#E2E8F0',
  FONDO: '#F8FAFC',
  SOFISTICADO: '#F1F5F9',
  TEXTO_PRIMARIO: '#1E293B',
  TEXTO_SECUNDARIO: '#64748B',
} as const;

export const NUMEROS = {
  TIEMPO_VIDA_RANGO: 30,
} as const;

export const ESTADOS = {
  PENDIENTE: 'pendiente',
  PROCESANDO: 'procesando',
  COMPLETO: 'completo',
  ERROR: 'error',
} as const;

export const IDENTIFICADOR_POR_DEFECTO = '00000000-0000-0000-0000-000000000000';

export const TITULOS = {
  PAGINA_INICIO: 'Panel de Control',
  LOGIN: 'Acceso de Usuario',
  BIENVENIDA: 'Bienvenido al Sistema HRM',
  LOGOUT: 'Cerrar Sesión',
  CONSULTA: 'Búsqueda',
  ACCION: 'Acción',
  DETALLE: 'Detalles',
} as const;

export const APIS = {
  BASE_URL: 'http://127.0.0.1:8000/api',
  USUARIOS: '/usuarios',
  OPERACIONES: '/operaciones',
  REPORTES: '/reportes',
} as const;
