import pathlib
base=pathlib.Path('c:/proyectos/HRM/frontend/src')
for p in ['componentes/graficas/graficaLinea.tsx','componentes/graficas/graficaBarras.tsx','componentes/graficas/graficaArea.tsx']:
    f=base/p
    s=f.read_text(encoding='utf-8')
    s=s.replace("    });\n\n    chart.applyOptions","    };\n\n    chart.applyOptions")
    f.write_text(s,encoding='utf-8')
# SelectorSimbodo
f=base/'componentes/mercado/SelectorSimbodo.tsx'
s=f.read_text(encoding='utf-8')
s=s.replace("          ))}\n        </option>\n","          ))}\n")
f.write_text(s,encoding='utf-8')
# CurvaEquity
f=base/'paginas/panel/CurvaEquity.tsx'
s=f.read_text(encoding='utf-8')
s=s.replace("       ).maxDrawdown).toFixed(2))}%\n","       ).maxDrawdown).toFixed(2))}%\n      </p>\n")
f.write_text(s,encoding='utf-8')
# KPIs
f=base/'paginas/panel/KPIsPrincipales.tsx'
s=f.read_text(encoding='utf-8')
s=s.replace("  pnlTotalDiario = 285.50;","  const pnlTotalDiario = 285.50;")
s=s.replace("  drawdownActual = -12.50;","  const drawdownActual = -12.50;")
s=s.replace("  drawdownMaximo = -45.80;","  const drawdownMaximo = -45.80;")
s=s.replace("  winRate = 67.5;","  const winRate = 67.5;")
s=s.replace("  cicloActual = 12;","  const cicloActual = 12;")
s=s.replace("valor={drawdownActual.toFixed(2)}%","valor={`${drawdownActual.toFixed(2)}%`}")
s=s.replace("valor={drawdownMaximo.toFixed(2)}%","valor={`${drawdownMaximo.toFixed(2)}%`}")
s=s.replace("valor={winRate.toFixed(1)}%","valor={`${winRate.toFixed(1)}%`}")
f.write_text(s,encoding='utf-8')
# Panel
f=base/'paginas/panel/Panel.tsx'
s=f.read_text(encoding='utf-8')
s=s.replace("import React, { useEffect, useRef } from 'react';","import React, { useEffect, useRef, useState } from 'react';")
s=s.replace("        </p>\n      )\n\n       {/* KPIs Principales */}","        </p>\n      )\n\n       {/* KPIs Principales */}")
f.write_text(s,encoding='utf-8')
print('done')
PY