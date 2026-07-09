# TFM - Cribado de mamografías a partir de representaciones latentes de AsymMirai

**Trabajo Fin de Master en Inteligencia Artificial - Universidad Internacional de Valencia (VIU)**

- Autor: Victor Rodriguez Rodriguez
- Directora: Karen Lopez-Linares
- Defensa: julio de 2026

---

## Descripcion

Este repositorio contiene el codigo y los resultados experimentales del TFM. El trabajo aborda la **clasificacion binaria BI-RADS** sobre mamografia de cribado (sospechoso 4-5 frente a no sospechoso 1-3) reutilizando el modelo **AsymMirai** (Donnelly et al., 2024) como extractor de features congelado sobre el dataset publico **VinDr-Mammo** (Nguyen et al., 2023).

Se comparan sistematicamente 10 configuraciones de features (contenido A/B/AB x pooling GAP+GMP/pool 2x2 x granularidad estudio/mama) y 9 cabezas de clasificacion, lo que da **84 modelos base**, ampliados con **fusion tardia con densidad mamaria** (108 modelos) y **calibracion post-hoc** (Platt scaling e isotonica) sobre los dos finalistas.

El pipeline usa validacion cruzada de 5 folds con `StratifiedGroupKFold` a nivel mama (para evitar fugas entre las mamas L y R del mismo estudio) y hold-out interno 80/20 para busqueda de hiperparametros. Las metricas se reportan con **AUC + IC 95 % bootstrap, AP, Brier y ECE**, y las comparaciones pareadas usan el **test de DeLong**.

## Modelos finales

| Uso | Modelo | Metricas sobre test |
|-----|--------|---------------------|
| A nivel estudio (triaje) | `M_A_22 + XGBoost`, agregado con `max(L,R)`, calibrado con Platt | AUC = 0,689 [0,628 - 0,748] - AP = 0,363 - Brier = 0,080 - ECE = 0,019 |
| A nivel mama (localizacion) | `M_A_gg + MLP`, calibrado con Platt | AUC = 0,687 [0,629 - 0,742] - Brier = 0,044 - ECE = 0,008 |

## Estructura del repositorio

```
.
├── README.md                # Este archivo
├── requirements.txt         # Dependencias Python
├── .gitignore
├── LICENSE
├── notebooks/               # 13 notebooks + README propio con detalle experimental
│   ├── README.md
│   ├── 00_overview.ipynb
│   ├── 01_inspeccion_modelo.ipynb
│   ├── 02_extraccion_un_estudio.ipynb
│   ├── 03_extraccion_masiva.ipynb
│   ├── 04_extraccion_pool22.ipynb
│   ├── 05_pipeline_unificado.ipynb
│   ├── 06_evaluacion.ipynb
│   ├── 07_fusion_densidad.ipynb
│   ├── 07b_fusion_densidad_mama.ipynb
│   ├── 07c_fusion_densidad_exhaustivo.ipynb
│   ├── 08_calibracion_posthoc.ipynb
│   ├── 09_ejemplos_birads.ipynb
│   └── 10_analisis_sonda_edad.ipynb
├── src/
│   ├── tfm_pipeline.py      # Builders de las 9 cabezas, grid search, KFold unificado
│   └── tfm_eval.py          # Bootstrap, DeLong, ECE, agregacion mama a estudio
└── outputs/                 # Resultados generados por los notebooks
    ├── Features/            # Features extraidas por NB03 y NB04 (no versionadas)
    ├── Predicciones/        # Predicciones OOF y test, metricas y CSVs de evaluacion
    └── Plots/               # Figuras de la memoria
```
El orden de ejecucion esta documentado en `notebooks/README.md`.

## Requisitos

- Python 3.11
- GPU NVIDIA con al menos 8 GB de VRAM (desarrollo hecho sobre RTX 4070 Super). Solo el MLP y la extraccion de features usan CUDA; el resto de cabezas corren en CPU.
- 16 GB de RAM (32 GB recomendados para las configuraciones de mayor dimension).
- Probado en Windows 11 con Miniconda.

Dependencias en `requirements.txt`.

## Datos y pesos externos

Ni el dataset ni los pesos preentrenados se incluyen en el repositorio:

- **VinDr-Mammo** (~5.000 estudios DICOM): descarga desde [PhysioNet](https://physionet.org/content/vindr-mammo/1.0.0/) tras aceptar el DUA correspondiente.
- **Snapshot de AsymMirai** (`trained_asymmirai.pt`, ~46 MB): disponible en el [repositorio oficial](https://github.com/jdonnelly36/AsymMirai).

Estructura esperada de rutas por defecto (la raiz se autodetecta como la carpeta padre de `notebooks/`, o se sobrescribe con la variable de entorno `TFM_PROJECT_ROOT`):

```
<TFM_PROJECT_ROOT>/
├── AsymMirai/
│   └── snapshots/
│       └── trained_asymmirai.pt
├── Data/
│   └── vindr-mammo/
│       ├── breast-level_annotations.csv
│       ├── finding_annotations.csv
│       └── images/...
└── <este repositorio>
```

## Reproduccion

```bash
# 1. Clonar
git clone <url-del-repo>
cd <repo>

# 2. Entorno
conda create -n tfm python=3.11
conda activate tfm
pip install -r requirements.txt
# Para PyTorch con CUDA especifica, sigue https://pytorch.org/get-started/locally/

# 3. (Opcional) apuntar a una raiz distinta si los datos estan en otro disco
#    Windows CMD:      set TFM_PROJECT_ROOT=D:\ruta\proyecto
#    Windows PS:       $env:TFM_PROJECT_ROOT = "D:\ruta\proyecto"
#    Linux / macOS:    export TFM_PROJECT_ROOT=/ruta/proyecto

# 4. Ejecutar los notebooks en el orden indicado en notebooks/README.md
```

El pipeline completo desde cero tarda aproximadamente 12 horas en hardware similar al descrito, dominado por NB05. Si las features ya estan extraidas, el resto del pipeline se completa en menos de una hora.

## Citacion

```bibtex
@mastersthesis{rodriguez2026tfm,
  author = {Rodriguez Rodriguez, Victor},
  title  = {Cribado de mamografías a partir de representaciones latentes de AsymMirai},
  school = {Universidad Internacional de Valencia (VIU)},
  year   = {2026},
  type   = {Trabajo Fin de Master en Inteligencia Artificial}
}
```

## Licencia

Codigo distribuido bajo licencia MIT (ver `LICENSE`).

- El modelo **AsymMirai** mantiene su licencia original.
- El dataset **VinDr-Mammo** esta sujeto al Data Use Agreement de PhysioNet. Este repositorio no incluye los datos.

## Agradecimientos

A la directora Karen Lopez-Linares por la guia durante todo el desarrollo. A los autores de AsymMirai por publicar codigo y pesos, y al equipo de VinDr por publicar el dataset.
