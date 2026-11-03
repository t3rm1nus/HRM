import { spawn } from 'node:child_process';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const directorioScripts = dirname(fileURLToPath(import.meta.url));
const ejecutableVitest = resolve(directorioScripts, '../node_modules/vitest/vitest.mjs');
const argumentos = process.argv.slice(2);

const proceso = spawn(process.execPath, [ejecutableVitest, ...argumentos], {
  stdio: 'inherit',
  env: { ...process.env, VITE_USAR_MOCKS: 'true' },
});

proceso.on('error', (error) => {
  console.error('No se pudo iniciar Vitest:', error.message);
  process.exitCode = 1;
});

proceso.on('exit', (codigo, senal) => {
  process.exitCode = codigo ?? (senal ? 1 : 0);
});
