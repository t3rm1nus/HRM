import type { ReactNode, Ref } from 'react';

export const BOTON_PRIMARIO = 'Primary';
export const BOTON_SECUNDARIO = 'Secondary';
export const BOTON_ACEPTAR = 'Aceptar';
export const BOTON_CANCELAR = 'Cancelar';
export const BOTON_CONFIRMAR = 'Confirmar';
export const BOTON_EMERGENCIA = 'Emergencia';
export const BOTON_CERRAR = 'Cerrar';
export const BOTON_SALUDAR = 'Salir';
export const BOTON_ESTADOS = 'Estados';
export const BOTON_CAMBIAR = 'Cambiar';
export const BOTON_VER = 'Ver';
export const BOTON_EDITAR = 'Editar';
export const BOTON_ELIMINAR = 'Eliminar';
export const BOTON_AGREGAR = 'Agregar';
export const BOTON_BUSCAR = 'Buscar';
export const BOTON_FILTRAR = 'Filtrar';
export const BOTON_REFRESCAR = 'Refrescar';
export const BOTON_DESTROZO = 'Destrozo';
export const BOTON_FIRE = 'Retirar';
export const BOTON_STOP = 'Detener';
export const BOTON_Pausa = 'Pausa';
export const BOTON_REANUDAR = 'Reanudar';
export const BOTON_DETENER = 'Detener';
export const BOTON_CONFIRMAR_EDAD = 'Confirmar edad';

export interface BotonProps {
  children: ReactNode;
  onClick?: () => void;
  tipo?: 'primary' | 'secundario' | 'success' | 'warning' | 'error' | 'ghost';
  cargando?: boolean;
  disabled?: boolean;
  className?: string;
  tipoRef?: Ref<HTMLButtonElement>;
  icono?: ReactNode;
}
