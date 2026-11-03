export const ICONO_PAGINAS = {
  HOME: '🏠',
  PROFILE: '👤',
  CONFIG: '⚙️',
  REPORTES: '📊',
  OPERACIONES: '📋',
  PORTAFOLIO: '💼',
  INGRESOS: '💰',
  EGRESOS: '⬅️',
  VENTAS: '🤝',
  MARKETING: '📣',
  USUARIOS: '👥',
  SOPORTE: '🆘',
  COMUNICACIONES: '✉️',
  CATALOGO: '🏪',
  PERSONAL: '👔',
  REGISTRO: '📜',
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
  { ruta: '/operaciones', nombre: 'Operaciones', icono: ICONO_PAGINAS.OPERACIONES, seccion: 'operaciones' },
  { ruta: '/portafolio', nombre: 'Portafolio', icono: ICONO_PAGINAS.PORTAFOLIO, seccion: 'principal' },
  { ruta: '/reportes', nombre: 'Reportes', icono: ICONO_PAGINAS.REPORTES, seccion: 'informes' },
  { ruta: '/usuarios', nombre: 'Usuarios', icono: ICONO_PAGINAS.USUARIOS, seccion: 'configuracion' },
  { ruta: '/configuracion', nombre: 'Configuración', icono: ICONO_PAGINAS.CONFIG, seccion: 'configuracion' },
  { ruta: '/ventas', nombre: 'Ventas', icono: ICONO_PAGINAS.VENTAS, seccion: 'principal' },
  { ruta: '/marketing', nombre: 'Marketing', icono: ICONO_PAGINAS.MARKETING, seccion: 'principal' },
  { ruta: '/soporte', nombre: 'Soporte', icono: ICONO_PAGINAS.SOPORTE, seccion: 'principal' },
  { ruta: '/comunicacion', nombre: 'Comunicación', icono: ICONO_PAGINAS.COMUNICACIONES, seccion: 'principal' },
  { ruta: '/catalogo', nombre: 'Catálogo', icono: ICONO_PAGINAS.CATALOGO, seccion: 'principal' },
  { ruta: '/personal', nombre: 'Personal', icono: ICONO_PAGINAS.PERSONAL, seccion: 'principal' },
  { ruta: '/registro', nombre: 'Registro', icono: ICONO_PAGINAS.REGISTRO, seccion: 'principal' },
  { ruta: '/monedas', nombre: 'Monedas', icono: ICONO_PAGINAS.HOME, seccion: 'principal' },
  { ruta: '/version', nombre: 'Version', icono: ICONO_PAGINAS.HOME, seccion: 'principal' },
];
