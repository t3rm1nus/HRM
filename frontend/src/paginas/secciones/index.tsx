import React from 'react';

const Pages: Record<string, React.LazyExoticComponent<() => JSX.Element>> = {
  inicio: React.lazy(async () => import('./secciones/paginaInicio')),
  operaciones: React.lazy(async () => import('./secciones/paginaOperaciones')),
  portafolio: React.lazy(async () => import('./secciones/paginaPortafolio')),
  reportes: React.lazy(async () => import('./secciones/paginaReportes')),
  usuarios: React.lazy(async () => import('./secciones/paginaUsuarios')),
  configuracion: React.lazy(async () => import('./secciones/paginaConfiguracion')),
  ventas: React.lazy(async () => import('./secciones/paginaVentas')),
  marketing: React.lazy(async () => import('./secciones/paginaMarketing')),
  soporte: React.lazy(async () => import('./secciones/paginaSoporte')),
  comunicacion: React.lazy(async () => import('./secciones/paginaComunicacion')),
  catalago: React.lazy(async () => import('./secciones/paginaCatalogo')),
  personal: React.lazy(async () => import('./secciones/paginaPersonal')),
  registro: React.lazy(async () => import('./secciones/paginaRegistro')),
  monedas: React.lazy(async () => import('./secciones/paginaMonedas')),
  version: React.lazy(async () => import('./secciones/paginaVersion')),
};

export const lazyPaginas = Pages;
