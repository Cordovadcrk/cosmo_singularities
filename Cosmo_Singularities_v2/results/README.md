# Resultados

Esta carpeta reúne las salidas seleccionadas para consulta, tesis y presentación:
38 archivos de figura (29 PDF y 9 PNG) y tres tablas. Los PDF conservan el formato vectorial
y son la opción preferida para documentos; los PNG acompañan las figuras que se
benefician de una vista previa directa en GitHub.

## Figuras

| Carpeta o archivo | Artefactos | Contenido | Notebook generador |
| --- | ---: | --- | --- |
| [`figures/cos_cosh/`](figures/cos_cosh/) | 1 figura en PDF y PNG | Evolución y diagnóstico de convergencia del modelo Cos–Cosh. | [`01_modelo_cos_cosh.ipynb`](../notebooks/01_modelo_cos_cosh.ipynb) |
| [`figures/barotropic/representative_cases/`](figures/barotropic/representative_cases/) | 6 PDF | Evolución de los casos B2, C1, C2, C3, D3 y F2. | [`03_espacio_parametros_barotropico.ipynb`](../notebooks/03_espacio_parametros_barotropico.ipynb) |
| [`figures/barotropic/behavior_maps/`](figures/barotropic/behavior_maps/) | 7 figuras en PDF y PNG | Diagnósticos de comportamiento para `omega = -1.4, -1.2, -1.05, -1.0, -0.95, -0.8, 0.0`; los símbolos distinguen estados provisionales, candidatos e inconclusos. | `03_espacio_parametros_barotropico.ipynb` |
| [`figures/barotropic/finite_infinite_beta_transition.pdf`](figures/barotropic/finite_infinite_beta_transition.pdf) | 1 PDF | Transición entre regímenes de tiempo finito e infinito en función de `beta`. | `03_espacio_parametros_barotropico.ipynb` |
| [`figures/barotropic/phantom_scan/`](figures/barotropic/phantom_scan/) | 1 figura en PDF y PNG | Mapa phantom revisado para `omega = -1.2`. | [`04_resultados_phantom_barotropico.ipynb`](../notebooks/04_resultados_phantom_barotropico.ipynb) |
| [`figures/phantom_q_alpha_rho/`](figures/phantom_q_alpha_rho/) | 8 PDF | Cinco figuras multipanel de evolución por régimen de `nu`, dos diagnósticos de `Hdot` y `w_eff`, y una comparación combinada del tiempo crítico con `alpha`. | [`05_graficos_phantom_ug.ipynb`](../notebooks/05_graficos_phantom_ug.ipynb) |
| [`figures/ug_vs_rg/`](figures/ug_vs_rg/) | 5 PDF | Comparación entre gravedad unimodular (UG) y relatividad general (RG) en cinco regímenes de `nu`. | [`A01_comparacion_ug_vs_rg.ipynb`](../notebooks/A01_comparacion_ug_vs_rg.ipynb) |

## Tablas

| Archivo | Dimensiones | Contenido | Notebook generador |
| --- | ---: | --- | --- |
| [`tables/barotropic/representative_cases.tex`](tables/barotropic/representative_cases.tex) | 16 casos | Versión LaTeX compacta del catálogo B1–F4, incluido el estado de cada diagnóstico. | `03_espacio_parametros_barotropico.ipynb` |
| [`tables/phantom/behavior_counts.csv`](tables/phantom/behavior_counts.csv) | 12 filas × 3 columnas | Conteos por `omega` y comportamiento final. | `04_resultados_phantom_barotropico.ipynb` |
| [`tables/phantom/analytic_corrections.csv`](tables/phantom/analytic_corrections.csv) | 88 filas × 6 columnas | Puntos cuyo diagnóstico numérico original contiene una corrección analítica explícita. | `04_resultados_phantom_barotropico.ipynb` |

## Regeneración y alcance

Las salidas se regeneran ejecutando los notebooks indicados desde la raíz del
repositorio. El notebook 03 carga por defecto
`data/processed/barotropic/parameter_scan_full.csv`; solo repite el barrido
completo si se activa `RUN_PARAMETER_SCAN`. El notebook 04 depende de
`data/processed/phantom_results_revised.csv`, cuya etapa completa de revisión no
está automatizada en este repositorio; sí vuelve a producir, de manera
reproducible a partir de ese artefacto, el mapa y las dos tablas phantom.

Los notebooks 05 y A01 generan sus figuras directamente a partir de las
expresiones e implementaciones del proyecto, sin depender de archivos externos.
Las diferencias menores de rasterizado, espaciado o últimos dígitos son posibles
entre versiones de Matplotlib y de las bibliotecas numéricas.

No edites manualmente una figura o tabla generada. Si requiere una corrección,
hazla en el notebook correspondiente y vuelve a ejecutar la celda de exportación
para mantener trazabilidad entre código y resultado.
