export const Tema = {
  claro: {
    FONDO: '#f8f9fa',
    SUPERFONDO: '#ffffff',
    TEXTO: '#212529',
    TEXTO_SECUNDARIO: '#6c757d',
    BORDE: '#dee2e6',
    EXITO: '#28a745',
    PELIGRO: '#dc3545',
    ADVERTENCIA: '#ffc107',
    NEUTRO: '#6c757d',
  },
  oscuro: {
    FONDO: '#212529',
    SUPERFONDO: '#343a40',
    TEXTO: '#f8f9fa',
    TEXTO_SECUNDARIO: '#adb5bd',
    BORDE: '#495057',
    EXITO: '#28a745',
    PELIGRO: '#dc3545',
    ADVERTENCIA: '#ffc107',
    NEUTRO: '#adb5bd',
  },
};

export type TemaClaroOscuro = 'claro' | 'oscuro';

export const useTema = (): { tema: keyof typeof Tema; toggle: () => void } => {
  const [tema, setTema] = React.useState<'claro' | 'oscuro'>('claro');
  
  const toggle = () => {
    setTema(tema === 'claro' ? 'oscuro' : 'claro');
  };
  
  return { tema, toggle };
};