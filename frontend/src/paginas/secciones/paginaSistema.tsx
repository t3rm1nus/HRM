import React from 'react';

export const PaginaSistema: React.FC = () => {
  const [soloLectura] = React.useState(true);
  const accion = (nombre: string) => {
    if (soloLectura) return;
    if (window.confirm(`¿Confirmar ${nombre}?`)) {
      // noop
    }
  };
  return (
    <div className="max-w-7xl mx-auto px-4 py-8">
      <h1 className="text-2xl font-bold text-gray-900">Sistema</h1>
      <div className="mt-4 bg-red-50 border border-red-200 rounded-xl p-3 text-sm text-red-800">
        Aviso permanente: arrancar el bot borra el estado y los trades paper.
      </div>
      {soloLectura && (
        <div className="mt-4 bg-gray-100 border rounded-xl p-3 text-sm text-gray-700">
          Modo solo lectura: controles desactivados.
        </div>
      )}
      <div className="mt-6 flex gap-2">
        <button type="button" disabled={soloLectura} onClick={() => accion('pausar')} className="px-4 py-2 bg-yellow-500 text-white rounded-lg disabled:opacity-50">Pausar</button>
        <button type="button" disabled={soloLectura} onClick={() => accion('reanudar')} className="px-4 py-2 bg-green-600 text-white rounded-lg disabled:opacity-50">Reanudar</button>
        <button type="button" disabled={soloLectura} onClick={() => accion('detener')} className="px-4 py-2 bg-red-600 text-white rounded-lg disabled:opacity-50">Detener</button>
      </div>
      <div className="mt-6 bg-white border rounded-xl p-4">
        <h2 className="font-semibold mb-2">Comprobaciones del entorno</h2>
        <ul className="text-sm">
          <li>Dependencias: OK</li>
          <li>Modelos: ausentes</li>
          <li>Claves configuradas: sí</li>
          <li>Python: 3.10</li>
        </ul>
      </div>
      <div className="mt-6 bg-white border rounded-xl p-4">
        <h2 className="font-semibold mb-2">Versión del código</h2>
        <p className="text-sm">v2.0.0</p>
      </div>
      <div className="mt-6 bg-white border rounded-xl p-4">
        <h2 className="font-semibold mb-2">Registro de auditoría</h2>
        <ul className="text-sm">
          <li>2026-02-12 - pausa solicitada</li>
        </ul>
      </div>
    </div>
  );
};
