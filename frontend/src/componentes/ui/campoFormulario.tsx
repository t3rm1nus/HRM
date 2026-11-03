import React from 'react';
import { z } from 'zod';

export interface CampoFormularioProps {
  label: string;
  name: string;
  value: string;
  onChange: (value: string) => void;
  onBlur?: () => void;
  error?: string;
  schema?: z.ZodType<string>;
  type?: 'text' | 'email' | 'password' | 'number' | 'textarea';
  placeholder?: string;
  required?: boolean;
  disabled?: boolean;
  ayuda?: string;
  className?: string;
}

export const CampoFormulario: React.FC<CampoFormularioProps> = ({
  label,
  name,
  value,
  onChange,
  onBlur,
  error,
  schema,
  type = 'text',
  placeholder,
  required = false,
  disabled = false,
  ayuda,
  className = '',
}) => {
  const [touched, setTouched] = React.useState(false);

  const validar = React.useCallback(() => {
    if (schema) {
      const resultado = schema.safeParse(value);
      if (!resultado.success) {
        return resultado.error.issues[0]?.message || 'Valor inválido';
      }
    }
    return '';
  }, [schema, value]);

  const handleBlur = () => {
    setTouched(true);
    if (onBlur) {
      onBlur();
    }
  };

  const errorActual = touched ? (error || validar()) : error;

  const inputProps = {
    id: name,
    name,
    value,
    onChange: (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => onChange(e.target.value),
    onBlur: handleBlur,
    placeholder,
    required,
    disabled,
    'aria-invalid': !!errorActual,
    'aria-describedby': ayuda ? `${name}-ayuda` : undefined,
    className: `w-full px-3 py-2 border rounded-lg text-sm transition focus:outline-none focus:ring-2 ${
      errorActual
        ? 'border-red-300 focus:ring-red-200 focus:border-red-500'
        : 'border-gray-300 focus:ring-blue-200 focus:border-blue-500'
    } ${disabled ? 'bg-gray-100 cursor-not-allowed' : ''}`,
  };

  return (
    <div className={`space-y-1 ${className}`}>
      <label
        htmlFor={name}
        className="block text-sm font-medium text-gray-700"
      >
        {label}
        {required && <span className="text-red-500 ml-1">*</span>}
      </label>
      {type === 'textarea' ? (
        <textarea {...inputProps} rows={4} />
      ) : (
        <input {...inputProps} type={type} />
      )}
      {errorActual && (
        <p className="text-xs text-red-600" role="alert">
          {errorActual}
        </p>
      )}
      {ayuda && !errorActual && (
        <p id={`${name}-ayuda`} className="text-xs text-gray-500">
          {ayuda}
        </p>
      )}
    </div>
  );
};

export default CampoFormulario;
