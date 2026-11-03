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
} as const

export type RutaKey = keyof typeof RUTAS

export const rutasDelPlan: RutaKey[] = [
  'INICIO',
  'MERCADO',
  'RENDIMIENTO',
  'OPERACIONES',
  'CARTERA',
  'SEÑALES',
  'ESTRATEGIA',
  'RIESGO',
  'AUTOAPRENDIZAJE',
  'MODELOS',
  'BACKTESTING',
  'CONFIGURACION',
  'FISCAL',
  'SALUD',
  'SISTEMA',
]

export const rutasDevOnly: RutaKey[] = ['CONFIGURACION'] // Could add more dev routes

// Mapeo de rutas a nombres de componentes
export const mapeoComponentes: Record<RutaKey, string> = {
  INICIO: 'PaginaPrincipal',
  MERCADO: 'PaginaMercado',
  RENDIMIENTO: 'PaginaRendimiento',
  OPERACIONES: 'PaginaOperaciones',
  CARTERA: 'PaginaCartera',
  SEÑALES: 'PaginaSenales',
  ESTRATEGIA: 'PaginaEstrategia',
  RIESGO: 'PaginaRiesgo',
  AUTOAPRENDIZAJE: 'PaginaAutoaprendizaje',
  MODELOS: 'PaginaModelos',
  BACKTESTING: 'PaginaBacktesting',
  CONFIGURACION: 'PaginaConfiguracion',
  FISCAL: 'PaginaFiscal',
  SALUD: 'PaginaSalud',
  SISTEMA: 'PaginaSistema',
}