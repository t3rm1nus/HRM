import pathlib
base=pathlib.Path('c:/proyectos/HRM/frontend/src')
f=list((base/'componentes/mercado').glob('SelectorSim*.tsx'))[0]
s=f.read_text(encoding='utf-8')
s=s.replace('        </option>\n','')
f.write_text(s,encoding='utf-8')
f=base/'paginas/panel/CurvaEquity.tsx'
s=f.read_text(encoding='utf-8')
s=s.replace('       ).maxDrawdown).toFixed(2))}%\n','       ).maxDrawdown).toFixed(2))}%\n      </p>\n')
f.write_text(s,encoding='utf-8')
f=base/'paginas/panel/Panel.tsx'
s=f.read_text(encoding='utf-8')
s=s.replace("import React, { useEffect, useRef, useState } from 'react';","import React, { useEffect, useRef, useState } from 'react';")
f.write_text(s,encoding='utf-8')
print('done')
