# Primera reparación del programa

## Cambios

1. Clasificadores Q=0 y Q=alpha*rho: eliminada la asignación futura incorrecta
   de Tipo II para nu<0 y Tipo IV para 0<nu<1/2. Con A>0 y H>0 la densidad
   aumenta; nu<=1/2 corresponde a divergencia sólo en tiempo infinito.
2. Cálculos exactos compartidos en `src/_analytic.py`. `log1p` y `expm1` evitan
   cancelación cerca de nu=1/2 y nu=1, sin convertir valores próximos en la
   frontera exacta. Se añadió `singular_time()` a Q=0; los datasets pueden
   construir su grilla antes del Big Rip/Tipo III.
3. `log_a_of_rho` en ambos modelos exactos. Para el modelo barotrópico omega=-1
   se rechaza a(rho): la expansión de de Sitter debe calcularse en tiempo.
4. Controles analíticos explícitos para beta=1 phantom y la rama decreciente
   dominada por difusión. Se preservan los límites observados numéricamente.
   No se cambia la procedencia de un resultado analítico a resultado numérico.
5. Eliminada la definición duplicada e idéntica de `BarotropicCaseAnalysis`.
6. Reproducido el TypeError de etiquetas mediante una recarga parcial de
   `src.singularities`. Se aceptan versiones anteriores de la misma enum por
   módulo, nombre y valor; no enums ajenas. La recarga es una causa reproducida,
   pero no prueba retrospectiva del estado exacto de cada kernel histórico.
7. Normalización consistente de enums y etiquetas leídas de CSV, preservando
   valores públicos como `little rip` por compatibilidad.
8. Checkpoints con identidad de grilla/opciones/código/entorno, escritura atómica,
   validación inicial y reintento de errores. Los archivos históricos sin firma
   se rechazan para reanudación, sin modificarlos.
9. README, dependencias numéricas fijadas, pruebas y script de validación.

## Evidencia y reproducción

Ejecutar desde la raíz `cosmo_singularities/`:

```bash
python -m unittest discover -s tests -v
python -m scripts.validate_core
```

Las pruebas incluyen orientación futura, ambos lados inmediatos de nu=1/2 y
nu=1, equivalencia de Q=0 y difusión lineal con alpha=0, continuidad y
Raychaudhuri mediante diferencias finitas, tiempo exacto al Big Rip,
crecimiento de de Sitter, frontera beta=2, recarga de enums, persistencia y
rechazo de checkpoints incompatibles sin sobrescritura.

El script conserva clasificación numérica/mixta y referencia analítica en
columnas distintas; genera también residuos de Friedmann y el signo de
Sigma=T*dot(S)/(3VH). No supone una ley de temperatura ni calcula S absoluto.
La primera corrida intermedia mostró `regular_expansion` en el caso difusivo;
esto motivó el control analítico explícito. Se conserva esa corrida con sus
hashes como registro anterior a la corrección, no como resultado vigente.

Resultado final de esta reparación: 20 pruebas aprobadas. Validación vigente:
`data/validation/20260922T003821590182Z/`, seis controles sin excepciones y
residuo relativo máximo de Friedmann de 3.55e-16. Las seis etiquetas concuerdan
con las referencias analíticas; dos usan explícitamente evidencia mixta, por
lo que ese acuerdo no constituye seis confirmaciones numéricas independientes.
La marca UTC corresponde a la noche del 21 de septiembre en Santiago.

Una copia de los siete archivos Python originales está en
`maintenance/baseline_20260921.zip`. No se movieron ni reescribieron datasets,
figuras, manuscritos ni notebooks históricos. La auditoría inicial de la carpeta
superior es un registro anterior a estas reparaciones.

## Pendientes delimitados

- Reejecutar los pilotos termodinámicos históricos en rutas nuevas y comparar
  resultados fila por fila. Todavía no se han regenerado sus 453/51 casos.
- Revisar las heurísticas de límites y confianza del clasificador general,
  incluyendo etiquetas de otros regímenes no phantom. Los controles exactos
  incorporados no constituyen una revisión completa de esas heurísticas.
- Centralizar las funciones termodinámicas aún definidas en notebooks, con
  hipótesis explícitas de Gibbs, temperatura, partículas y volumen.
- Preparar una interfaz compartida para el modelo no lineal general, reutilizando
  Q,Q', presión y diagnósticos; distinguir geometría y materia, especialmente
  la excepción nu=0 en D=0.
- El caso inicial exacto D0=0 exige análisis separado; los barridos regulares
  lo excluyen. No se ha construido una continuación por esa frontera.
- Evaluar migración del paquete `src` a un nombre instalable sin romper imports;
  por ahora se conserva deliberadamente la interfaz histórica.
- Revisar notebooks de gráficos y ejecución completa en kernel limpio. Ninguna
  salida antigua se considera actualizada sólo por haberse corregido el core.

La revisión bibliográfica, las condiciones S>0 y ddot(S)<0 y la clasificación
de posibles comportamientos nuevos siguen requiriendo trabajo científico.
