import React from 'react';

const PESTANAS = ['General', 'Trading', 'Riesgo', 'Modelos', 'Autoaprendizaje', 'Cartera', 'Señales', 'Notificaciones', 'Seguridad', 'Avanzado'];

export const PaginaConfiguracion: React.FC = () => {
  const [tab, setTab] = React.useState(PESTANAS[0]);
  const [cambios, setCambios] = React.useState(0);
  return (
    <div className="max-w-7xl mx-auto px-4 py-8">
      <h1 className="text-2xl font-bold text-gray-900">Configuración</h1>
      <div className="mt-4 flex flex-wrap gap-2">
        {PESTANAS.map((p) => (
          <button key={p} type="button" onClick={() => setTab(p)} className={`px-3 py-1.5 text-sm rounded-lg ${tab === p ? 'bg-blue-600 text-white' : 'bg-gray-100 text-gray-700'}`}>
            {p}
          </button>
        ))}
      </div>
      <div className="mt-6 bg-white border rounded-xl p-4">
        <p>Pestaña activa: {tab}</p>
        <p className="text-sm text-gray-600">Formularios generados desde esquema (mock). Credenciales: solo escritura.</p>
      </div>
      <div className="fixed bottom-4 right-4 bg-white border shadow rounded-lg px-4 py-2 text-sm">
        {cambios} cambios sin guardar
      </div>
    </div>
  );
};
