import { TipoMoneda } from '../dominio';

export const SímboloMoneda = ({ símbolo }: { símbolo: TipoMoneda }) => {
  const símbolos: Record<TipoMoneda, string> = {
    [TipoMoneda.USD]: '$',
    [TipoMoneda.EUR]: '€',
    [TipoMoneda.PEN]: 'S/',
  };
  return <>{símbolos[símbolo]}</>;
};

export const DescripciónMoneda = ({ símbolo }: { símbolo: TipoMoneda }) => {
  const descripciones: Record<TipoMoneda, string> = {
    [TipoMoneda.USD]: 'Dólar Estadounidense',
    [TipoMoneda.EUR]: 'Euro',
    [TipoMoneda.PEN]: 'Sol',
  };
  return <>{descripciones[símbolo]}</>;
};

export const NombreMoneda = ({ símbolo }: { símbolo: TipoMoneda }) => {
  const nombres: Record<TipoMoneda, string> = {
    [TipoMoneda.USD]: 'Dólar',
    [TipoMoneda.EUR]: 'Euro',
    [TipoMoneda.PEN]: 'Sol',
  };
  return <>{nombres[símbolo]}</>;
};