import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { ZONA_LOCAL, type PreferenciaZona } from '../dominio/zonaHoraria';

interface EstadoZonaHoraria {
  zona: PreferenciaZona;
  establecerZona: (zona: PreferenciaZona) => void;
}

export const usarZonaHoraria = create<EstadoZonaHoraria>()(
  persist(
    (set) => ({
      zona: ZONA_LOCAL,
      establecerZona: (zona) => set({ zona }),
    }),
    { name: 'hrm-zona-horaria' },
  ),
);
