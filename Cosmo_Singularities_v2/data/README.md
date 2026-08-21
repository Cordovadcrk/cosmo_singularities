# Datos procesados

Esta carpeta contiene datos numéricos derivados de los modelos cosmológicos del
proyecto. No son mediciones observacionales: se obtienen mediante barridos de
parámetros, integración de ecuaciones y clasificación analítica o numérica.

Todos los archivos están en formato CSV, con encabezado y una fila por punto de
la grilla o caso estudiado. Los valores ausentes se conservan como campos vacíos
cuando una magnitud no aplica o no pudo estimarse.

## Inventario

| Archivo | Dimensiones | Contenido | Origen y reproducibilidad |
| --- | ---: | --- | --- |
| [`processed/barotropic/parameter_scan_full.csv`](processed/barotropic/parameter_scan_full.csv) | 5.425 filas × 37 columnas | Barrido completo del modelo barotrópico para siete valores de `omega` y una grilla en `alpha` y `beta`. Conserva parámetros, dominio inicial, ruta de análisis, evento terminal, clasificación y límites estimados. | Es el artefacto completo versionado que carga por defecto [`notebooks/03_espacio_parametros_barotropico.ipynb`](../notebooks/03_espacio_parametros_barotropico.ipynb). Para recalcularlo desde cero hay que cambiar `RUN_PARAMETER_SCAN = True`; es una operación considerablemente más costosa que ejecutar el resto del notebook. |
| [`processed/barotropic/scientific_table_full.csv`](processed/barotropic/scientific_table_full.csv) | 5.425 filas × 20 columnas | Vista científica compacta del barrido completo: parámetros, escalas críticas, clasificación, límites, turnaround y confianza. | Lo exporta el notebook 03 a partir del barrido cargado o recién calculado. |
| [`processed/barotropic/representative_cases.csv`](processed/barotropic/representative_cases.csv) | 16 filas × 31 columnas | Catálogo B1–F4 de casos representativos, con clasificación, límites y estado del solver. | Lo regenera el notebook 03 al analizar nuevamente el catálogo de casos. |
| [`processed/phantom_results_revised.csv`](processed/phantom_results_revised.csv) | 2.325 filas × 50 columnas | Resultado consolidado de las tres secciones phantom del barrido (`omega = -1.4, -1.2, -1.05`). A las 37 columnas del esquema base se añaden 13 campos de revisión analítica y diagnóstico futuro, entre ellos `analytic_behavior`, `analytic_correction` y `final_behavior`. | [`notebooks/04_resultados_phantom_barotropico.ipynb`](../notebooks/04_resultados_phantom_barotropico.ipynb) **consume** este CSV, pero no reconstruye la etapa completa de revisión que lo produjo. Por ello se conserva como artefacto procesado versionado y no debe reemplazarse sin registrar el procedimiento de revisión. |

## Familias de columnas

Los CSV barotrópicos separan conceptualmente cuatro grupos de información:

- parámetros y condiciones iniciales: `omega`, `alpha`, `beta`, `rho0`, `a0`
  y `t0`;
- dominio y escalas críticas: `F0`, `D0`, `rho_F`, `rho_D` y
  `valid_initial_domain`;
- diagnóstico dinámico: `analysis_route`, `termination_event`,
  `standard_type`, `behavior_label` y `status`;
- límites y control numérico: `time_limit`, `rho_limit`, `a_limit`, `H_limit`,
  derivadas, confianza y mensajes de error.

`phantom_results_revised.csv` agrega una capa explícita de revisión: conserva el
diagnóstico numérico original, registra si hubo corrección analítica y presenta
la clasificación final utilizada en las tablas y figuras.

El barrido completo registra 3.570 filas `robust`, 1.355 `provisional`, 418
`new_behavior_candidate` y 82 `numerically_inconclusive`. Estos estados son
parte del resultado: las filas no robustas quedan visibles para análisis y no
deben interpretarse como conclusiones cerradas.

## Cómo regenerar los datos

Ejecuta los notebooks desde la raíz del repositorio y en este orden:

1. `03_espacio_parametros_barotropico.ipynb` para volver a exportar la tabla
   científica y el catálogo representativo. Con `RUN_PARAMETER_SCAN = False`
   usa el barrido completo incluido; con `True` recalcula las 5.425 filas y
   guarda checkpoints por cada valor de `omega`.
2. `04_resultados_phantom_barotropico.ipynb` para producir las tablas y figuras
   derivadas del CSV phantom revisado. Este paso no regenera dicho CSV.

Los últimos dígitos de cantidades de punto flotante pueden variar levemente
entre versiones de Python, NumPy y SciPy o por cambios en las tolerancias del
solver. Las clasificaciones marcadas como `robust` deberían permanecer estables;
los estados provisionales, candidatos o inconclusos son deliberadamente
revisables. Cualquier cambio de clasificación requiere una revisión científica,
no solo una comparación textual de los CSV.
