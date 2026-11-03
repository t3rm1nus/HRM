/// <reference types="vite/client" />

declare module '*.vue' {
  const component: import('vue').DefineComponent;
  export default component;
}

declare module 'lightweight-charts';
declare module 'msw';