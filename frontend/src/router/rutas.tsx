import React, { lazy } from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { Sidebar } from '../componentes/sidebar';
import { Navbar } from '../componentes/navbar';

// Importar omitiendo dependencias circulares
import { RUTAS } from '../constantes/constantes';

// Páginas lazy-loaded (wrapper para manejar exportaciones con nombre)
const lazyImport = (ruta: string, nombreExport: string) =>
  lazy(() => import(ruta).then((mod) => ({ default: mod[nombreExport] })));

const Inicio = lazyImport('../paginas/secciones/paginaInicio', 'PaginaInicial');
const Operaciones = lazyImport('../paginas/secciones/paginaOperaciones', 'PaginaOperaciones');
const Portafolio = lazyImport('../paginas/secciones/paginaPortafolio', 'PaginaPortafolio');
const Reportes = lazyImport('../paginas/secciones/paginaReportes', 'PaginaReportes');
const Usuarios = lazyImport('../paginas/secciones/paginaUsuarios', 'PaginaUsuarios');
const Configuracion = lazyImport('../paginas/secciones/paginaConfiguracion', 'PaginaConfiguracion');
const Ventas = lazyImport('../paginas/secciones/paginaVentas', 'PaginaVentas');
const Marketing = lazyImport('../paginas/secciones/paginaMarketing', 'PaginaMarketing');
const Soporte = lazyImport('../paginas/secciones/paginaSoporte', 'PaginaSoporte');
const Comunicacion = lazyImport('../paginas/secciones/paginaComunicacion', 'PaginaComunicacion');
const Catalogo = lazyImport('../paginas/secciones/paginaCatalogo', 'PaginaCatalogo');
const Personal = lazyImport('../paginas/secciones/paginaPersonal', 'PaginaPersonal');
const Registro = lazyImport('../paginas/secciones/paginaRegistro', 'PaginaRegistro');
const Monedas = lazyImport('../paginas/secciones/paginaMonedas', 'PaginaMonedas');
const Version = lazyImport('../paginas/secciones/paginaVersion', 'PaginaVersion');
const NotFound = lazyImport('../paginas/notFound', 'NotFoundPage');
const DevComponentes = lazyImport('../paginas/devComponentes', 'PaginaDevComponentes');

export const AppRouter: React.FC = () => {
  return (
    <BrowserRouter>
      <div className="flex min-h-screen bg-gray-100">
        <Sidebar />
        <div className="flex-1 flex flex-col">
          <Navbar />
          <main className="flex-1 p-4 overflow-auto">
            <Routes>
              <Route path={RUTAS.INICIO} element={<Inicio />} />
              <Route path={RUTAS.OPERACIONES} element={<Operaciones />} />
              <Route path={RUTAS.PORTAFOLIO} element={<Portafolio />} />
              <Route path={RUTAS.REPORTES} element={<Reportes />} />
              <Route path={RUTAS.USUARIOS} element={<Usuarios />} />
              <Route path={RUTAS.CONFIGURACION} element={<Configuracion />} />
              <Route path={RUTAS.VENTAS} element={<Ventas />} />
              <Route path={RUTAS.MARKETING} element={<Marketing />} />
              <Route path={RUTAS.SOPORTE} element={<Soporte />} />
              <Route path={RUTAS.COMUNICACION} element={<Comunicacion />} />
              <Route path={RUTAS.CATALOGO} element={<Catalogo />} />
              <Route path={RUTAS.PERSONAL} element={<Personal />} />
              <Route path={RUTAS.REGISTRO} element={<Registro />} />
              <Route path={RUTAS.MONEDAS} element={<Monedas />} />
              <Route path={RUTAS.VERSION} element={<Version />} />

              {/* Dev */}
              <Route path="/dev/componentes" element={<DevComponentes />} />
              <Route path="/panel" element={<Panel />} />
      <Route path="/mercado" element={<Mercado />} />

              {/* 404 */}
              <Route path={RUTAS.ERROR_404} element={<NotFound />} />

              {/* Redirección a 404 para rutas desconocidas */}
              <Route path="*" element={<Navigate to={RUTAS.ERROR_404} replace />} />
            </Routes>
          </main>
        </div>
      </div>
    </BrowserRouter>
  );
};

