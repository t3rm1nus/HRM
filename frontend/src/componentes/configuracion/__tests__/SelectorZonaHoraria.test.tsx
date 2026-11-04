import { fireEvent, render, screen } from '@testing-library/react';
import { beforeEach, describe, expect, it } from 'vitest';
import { SelectorZonaHoraria } from '../SelectorZonaHoraria';
import { usarZonaHoraria } from '../../../estado/storeZonaHoraria';
import { ZONA_LOCAL } from '../../../dominio/zonaHoraria';

describe('SelectorZonaHoraria', () => {
  beforeEach(() => usarZonaHoraria.setState({ zona: ZONA_LOCAL }));

  it('arranca en automática', () => {
    render(<SelectorZonaHoraria />);
    expect(screen.getByRole('combobox')).toHaveValue(ZONA_LOCAL);
  });

  it('guarda la zona elegida', () => {
    render(<SelectorZonaHoraria />);
    fireEvent.change(screen.getByRole('combobox'), { target: { value: 'UTC' } });
    expect(usarZonaHoraria.getState().zona).toBe('UTC');
  });
});
