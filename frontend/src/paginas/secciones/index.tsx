import React from 'react';

const Pages: Record<string, React.LazyExoticComponent<() => JSX.Element>> = {
  inicio: React.lazy(async () => import('./paginaInicio')),
  operaciones: React.lazy(async () => import('./paginaOperaciones')),
  portafolio: React.lazy(async () => import('./paginaPortafolio')),
  reportes: React.lazy(async () => import('./paginaReportes')),
  usuarios: React.lazy(async () => import('./paginaUsuarios')),
  configuracion: React.lazy(async () => import('./paginaConfiguracion')),
  ventas: React.lazy(async () => import('./paginaVentas')),
  marketing: React.lazy(async () => import('./paginaMarketing')),
  soporte: React.lazy(async () => import('./paginaSoporte')),
  comunicacion: React.lazy(async () => import('./paginaComunicacion')),
  catalago: React.lazy(async () => import('./paginaCatalogo')),
  personal: React.lazy(async () => import('./paginaPersonal')),
  registro: React.lazy(async () => import('./paginaRegistro')),
  monedas: React.lazy(async () => import('./paginaMonedas')),
  version: React.lazy(async () => import('./paginaVersion')),
};

export const lazyPaginas = Pages;
