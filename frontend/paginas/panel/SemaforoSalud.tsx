import React from 'react';

export const SemaforoSalud: React.FC = () => {
  return (
    <div className="bg-green-100 border-green-400 rounded-lg p-4">
      <h3 className="text-sm font-medium text-green-800">Semáforo de Salud</h3>
      <p className="text-xs text-green-600">Estado del semáforo: <span className="font-semibold text-green-600">Normal</span></p>
    </div>
  );
};

export default SemaforoSalud;