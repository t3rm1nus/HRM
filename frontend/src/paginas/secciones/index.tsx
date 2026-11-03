import React from 'react';

const Pages: Record<string, React.LazyExoticComponent<React.ComponentType>> = {
  inicio: React.lazy(async () => { const modulo = await import('./paginaInicio'); return { default: modulo.PaginaInicial }; }),
  operaciones: React.lazy(async () => { const modulo = await import('./paginaOperaciones'); return { default: modulo.PaginaOperaciones }; }),
  portafolio: React.lazy(async () => { const modulo = await import('./paginaPortafolio'); return { default: modulo.PaginaPortafolio }; }),
  reportes: React.lazy(async () => { const modulo = await import('./paginaReportes'); return { default: modulo.PaginaReportes }; }),
  usuarios: React.lazy(async () => { const modulo = await import('./paginaUsuarios'); return { default: modulo.PaginaUsuarios }; }),
  configuracion: React.lazy(async () => { const modulo = await import('./paginaConfiguracion'); return { default: modulo.PaginaConfiguracion }; }),
  ventas: React.lazy(async () => { const modulo = await import('./paginaVentas'); return { default: modulo.PaginaVentas }; }),
  marketing: React.lazy(async () => { const modulo = await import('./paginaMarketing'); return { default: modulo.PaginaMarketing }; }),
  soporte: React.lazy(async () => { const modulo = await import('./paginaSoporte'); return { default: modulo.PaginaSoporte }; }),
  comunicacion: React.lazy(async () => { const modulo = await import('./paginaComunicacion'); return { default: modulo.PaginaComunicacion }; }),
  catalago: React.lazy(async () => { const modulo = await import('./paginaCatalogo'); return { default: modulo.PaginaCatalogo }; }),
  personal: React.lazy(async () => { const modulo = await import('./paginaPersonal'); return { default: modulo.PaginaPersonal }; }),
  registro: React.lazy(async () => { const modulo = await import('./paginaRegistro'); return { default: modulo.PaginaRegistro }; }),
  monedas: React.lazy(async () => { const modulo = await import('./paginaMonedas'); return { default: modulo.PaginaMonedas }; }),
  version: React.lazy(async () => { const modulo = await import('./paginaVersion'); return { default: modulo.PaginaVersion }; }),
};

export const lazyPaginas = Pages;
