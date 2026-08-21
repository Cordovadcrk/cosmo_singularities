# Guía de notebooks

Los notebooks forman un recorrido curado: cada archivo tiene un propósito distinto, emplea rutas relativas a la raíz del repositorio y conserva sus salidas para que los resultados puedan inspeccionarse directamente en GitHub.

| Orden | Notebook | Entrada principal | Salida principal |
|---:|---|---|---|
| 1 | `01_modelo_cos_cosh.ipynb` | Parámetros Cos–Cosh declarados en el notebook | Figura de evolución y diagnóstico de convergencia. |
| 2 | `02_validacion_modelo_barotropico.ipynb` | Catálogo analítico B1–F4 | Validación de 16 casos y `representative_cases.csv`. |
| 3 | `03_espacio_parametros_barotropico.ipynb` | `parameter_scan_full.csv` o un barrido nuevo | Mapas, fronteras, tabla científica y casos representativos. |
| 4 | `04_resultados_phantom_barotropico.ipynb` | `phantom_results_revised.csv` | Conteos, correcciones analíticas y mapa phantom revisado. |
| 5 | `05_graficos_phantom_ug.ipynb` | Soluciones analíticas incorporadas | Ocho figuras finales sobre regímenes phantom en UG. |
| A1 | `A01_comparacion_ug_vs_rg.ipynb` | Soluciones analíticas incorporadas | Cinco figuras comparativas UG–RG. |

## Ejecución

Abre Jupyter desde la raíz del repositorio para que la detección de rutas sea inmediata:

```bash
jupyter lab
```

También se pueden ejecutar todos de forma no interactiva:

```bash
for notebook in notebooks/*.ipynb; do
  jupyter nbconvert --to notebook --execute --inplace "$notebook"
done
```

El notebook 03 usa `RUN_PARAMETER_SCAN = False` por defecto y lee el barrido completo versionado. Cambia ese interruptor a `True` solo si deseas reconstruir los 5425 puntos; durante ese proceso los checkpoints temporales se separan por modo en `data/processed/barotropic/checkpoints/pilot/` o `full/`, rutas excluidas de Git.

## Criterios de la selección

- Se eliminó la duplicación entre pruebas exploratorias y análisis finales, incluido un dataset phantom que mezclaba una taxonomía local con la evolución futura.
- Los 16 casos representativos permanecen en una validación compacta y trazable.
- Las rutas locales y los nombres dependientes del sistema operativo fueron reemplazados por rutas portables.
- Los gráficos de Big Rip muestran `ln(a)` cuando el factor de escala desborda numéricamente.
- Los gráficos finales se guardan en `results/`; los resultados tabulares, en `data/processed/` o `results/tables/` según su función.
