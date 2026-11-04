import React, { Suspense, lazy } from 'react';
import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom';
import { Sidebar } from '../componentes/sidebar';
import { Navbar } from '../componentes/navbar';
import { RUTAS } from '../constantes/constantes';
import { usarSincronizacionBackend } from '../hooks/useSincronizacionBackend';

const Inicio = lazy(() =>
  import('../paginas/secciones/paginaInicio').then((modulo) => ({ default: modulo.PaginaInicial })),
);
const Mercado = lazy(() =>
  import('../paginas/mercado/Mercado').then((modulo) => ({ default: modulo.Mercado })),
);
const Operaciones = lazy(() =>
  import('../paginas/secciones/paginaOperaciones').then((modulo) => ({ default: modulo.PaginaOperaciones })),
);
const Cartera = lazy(() =>
  import('../paginas/secciones/paginaCartera').then((modulo) => ({ default: modulo.PaginaCartera })),
);
const Rendimiento = lazy(() =>
  import('../paginas/secciones/paginaReportes').then((modulo) => ({ default: modulo.PaginaReportes })),
);
const Senales = lazy(() =>
  import('../paginas/secciones/paginaSenales').then((modulo) => ({ default: modulo.PaginaSenales })),
);
const Estrategia = lazy(() =>
  import('../paginas/secciones/paginaEstrategia').then((modulo) => ({ default: modulo.PaginaEstrategia })),
);
const Riesgo = lazy(() =>
  import('../paginas/secciones/paginaRiesgo').then((modulo) => ({ default: modulo.PaginaRiesgo })),
);
const Autoaprendizaje = lazy(() =>
  import('../paginas/secciones/paginaAutoaprendizaje').then((modulo) => ({ default: modulo.PaginaAutoaprendizaje })),
);
const Modelos = lazy(() =>
  import('../paginas/secciones/paginaModelos').then((modulo) => ({ default: modulo.PaginaModelos })),
);
const Backtesting = lazy(() =>
  import('../paginas/secciones/paginaBacktesting').then((modulo) => ({ default: modulo.PaginaBacktesting })),
);
const Configuracion = lazy(() =>
  import('../paginas/secciones/paginaConfiguracion').then((modulo) => ({ default: modulo.PaginaConfiguracion })),
);
const Fiscal = lazy(() =>
  import('../paginas/secciones/paginaFiscal').then((modulo) => ({ default: modulo.PaginaFiscal })),
);
const Salud = lazy(() =>
  import('../paginas/secciones/paginaSalud').then((modulo) => ({ default: modulo.PaginaSalud })),
);
const Sistema = lazy(() =>
  import('../paginas/secciones/paginaSistema').then((modulo) => ({ default: modulo.PaginaSistema })),
);
const NotFound = lazy(() =>
  import('../paginas/notFound').then((modulo) => ({ default: modulo.NotFoundPage })),
);

export const AppRouter: React.FC = () => {
  usarSincronizacionBackend();
  return (
  <BrowserRouter>
    <div className="flex min-h-screen bg-gray-100">
      <Sidebar />
      <div className="flex min-w-0 flex-1 flex-col">
        <Navbar />
        <main className="flex-1 overflow-auto p-4">
          <Suspense fallback={<div className="p-6" role="status">Cargando módulo…</div>}>
            <Routes>
              <Route path={RUTAS.INICIO} element={<Inicio />} />
              <Route path={RUTAS.MERCADO} element={<Mercado />} />
              <Route path={RUTAS.OPERACIONES} element={<Operaciones />} />
              <Route path={RUTAS.CARTERA} element={<Cartera />} />
              <Route path={RUTAS.RENDIMIENTO} element={<Rendimiento />} />
              <Route path={RUTAS.SEÑALES} element={<Senales />} />
              <Route path={RUTAS.ESTRATEGIA} element={<Estrategia />} />
              <Route path={RUTAS.RIESGO} element={<Riesgo />} />
              <Route path={RUTAS.AUTOAPRENDIZAJE} element={<Autoaprendizaje />} />
              <Route path={RUTAS.MODELOS} element={<Modelos />} />
              <Route path={RUTAS.BACKTESTING} element={<Backtesting />} />
              <Route path={RUTAS.CONFIGURACION} element={<Configuracion />} />
              <Route path={RUTAS.FISCAL} element={<Fiscal />} />
              <Route path={RUTAS.SALUD} element={<Salud />} />
              <Route path={RUTAS.SISTEMA} element={<Sistema />} />
              <Route path={RUTAS.ERROR_404} element={<NotFound />} />
              <Route path="*" element={<Navigate to={RUTAS.ERROR_404} replace />} />
            </Routes>
          </Suspense>
        </main>
      </div>
    </div>
  </BrowserRouter>
  );
};
