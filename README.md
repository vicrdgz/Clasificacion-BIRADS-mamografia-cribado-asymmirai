# TFM — Clasificación binaria BI-RADS sobre mamografía de cribado con AsymMirai

> **Trabajo Fin de Máster en Inteligencia Artificial · Universidad Internacional de Valencia (UIV)**
>
> Autor: **Víctor Rodríguez Rodríguez** · Directora: **Karen López‑Linares** · Defensa: julio 2026

---

## Descripción del proyecto

Este repositorio contiene el código y los resultados experimentales del TFM **"Modelos de Riesgo de Cáncer de Mama a partir de Mamografía de Cribado"**, reformulado como un problema de **clasificación binaria BI‑RADS** (sospechoso 4–5 *vs* no sospechoso 1–3) sobre el dataset público **VinDr‑Mammo** (~5 000 estudios, prevalencia ≈ 10 %).

El enfoque experimental utiliza el modelo **AsymMirai** congelado como extractor de *features* (sin *fine‑tuning*) y compara sistemáticamente:

- Dos estrategias de **pooling** espacial de los mapas internos (GAP+GMP *vs* pool 2×2 adaptativo).
- Cinco tipos de **input**: por mama (M) o por estudio (E), con vistas (A), asimetría bilateral (B) o ambas (AB).
- Nueve **cabezas de clasificación** clásicas (LogReg L1/L2/EN, RandomForest, ExtraTrees, HistGradientBoosting, XGBoost, LightGBM, MLP).
- Estrategias de **fusión con densidad mamaria** (Platt, HistGB, MLP a nivel mama o estudio).
- **Calibración post‑hoc** (Platt scaling e isotonic regression) sobre los modelos finales.

El protocolo de evaluación combina **5‑fold StratifiedGroupKFold** con `study_id` como grupo (para evitar leak entre mamas L y R del mismo estudio), un **hold‑out interno 80/20 por fold** para búsqueda de hiperparámetros, y métricas **AUC + IC 95 % bootstrap, AP, Brier y ECE**, con **test pareado de DeLong** para todas las comparaciones significativas.

---

## Resultados principales

### Modelo final recomendado para uso clínico

**`M_A_22 + XGBoost`** (vistas por mama con pooling 2×2 + XGBoost), agregado a nivel estudio con `max(pred_L, pred_R)` y calibrado mediante **Platt scaling**:

| Métrica | Valor (IC 95 %) | Comentario |
|---|---|---|
| AUC test (estudio) | **0.6891 [0.6282 – 0.7479]** | Discriminación preservada exactamente tras calibrar |
| Brier score | **0.0802** | −43 % frente al base (0.1403) |
| ECE | **0.0193** | −93 % frente al base (0.2599) |
| AP | 0.3632 | Prevalencia test = 9.6 % |

Parámetros del calibrador: `a = 3.125, b = −3.501` (sigmoide `σ(a·p + b)`).

### Hallazgos metodológicos publicables

1. **GAP+GMP ≥ pool 2×2** (refuta hipótesis inicial). 4/42 comparaciones DeLong significativas, todas a favor de GAP+GMP. Probable causa: sobreajuste del pool espacial con 3 200–7 200 muestras de entrenamiento y 4× más dimensiones.
2. **Combinar A+B no mejora a A solo** (0/34 comparaciones DeLong significativas). La asimetría bilateral es función matemática de las vistas y no aporta información incremental.
3. **El pipeline unificado equivale al Hito 2 sin densidad** (M_A_gg+MLP=0.6866 vs M_A_mlp=0.6830, Δ=+0.0037, p=0.892). Validación independiente: las elecciones del Hito 2 eran óptimas dentro del espacio explorado.
4. **La fusión con densidad NO mejora significativamente** en ninguna de las **108 combinaciones evaluadas exhaustivamente** (NB11c). Solo 2/108 con mejora significativa, ambas sobre modelos base mediocres (AUC < 0.62) y sin alcanzar al top absoluto.
5. **Predecir a nivel mama y agregar con `max(L, R)` ≥ predecir a nivel estudio directamente** (patrón consistente en 9/9 comparaciones). Justifica la elección metodológica de operar a nivel mama.
6. **Platt scaling > isotonic regression** en este setting de baja prevalencia. Isotonic colapsa el rango de predicciones perdiendo discriminación significativamente (p<0.05); Platt preserva AUC exactamente por ser estrictamente monótono.

---

## Estructura del repositorio

```
.
├── README.md                    # Este archivo
├── requirements.txt             # Dependencias Python
├── .gitignore                   # Archivos/carpetas excluidos del repo
├── LICENSE                      # Licencia MIT
├── notebooks/                   # 15 notebooks (Hito 2 + experimental v2)
│   ├── README.md               # Guía de ejecución y descripción
│   ├── 00_overview.ipynb       # Introducción al setup
│   ├── 01_inspeccion_modelo.ipynb
│   ├── 02_extraccion_un_estudio.ipynb
│   ├── 03_extraccion_masiva.ipynb           # Features GAP+GMP
│   ├── 04_definicion_modelos.ipynb          # [legacy Hito 2]
│   ├── 05_entrenamiento_kfold.ipynb         # [legacy Hito 2]
│   ├── 06_evaluacion.ipynb                  # [legacy Hito 2]
│   ├── 07_fusion_densidad.ipynb             # [legacy Hito 2]
│   ├── 08_extraccion_pool22.ipynb           # Features pool 2×2
│   ├── 09_pipeline_unificado.ipynb          # Pipeline final: 84 corridas
│   ├── 10_evaluacion_v2.ipynb               # Evaluación rigurosa con DeLong
│   ├── 11_fusion_densidad_v2.ipynb          # Fusión densidad nivel estudio
│   ├── 11b_fusion_densidad_mama.ipynb       # Fusión densidad nivel mama
│   ├── 11c_fusion_densidad_exhaustivo.ipynb # 108 combinaciones evaluadas
│   └── 12_calibracion_posthoc.ipynb         # Calibración Platt + isotónica
├── src/                         # Módulos Python reutilizables
│   ├── README.md
│   ├── tfm_models.py            # Cabezas MLP / GBM (Hito 2)
│   ├── tfm_pipeline.py          # Pipeline de entrenamiento unificado
│   └── tfm_eval.py              # Bootstrap, DeLong, ECE, agregaciones
└── outputs/                     # Resultados generados (predicciones + métricas + gráficas)
    ├── README.md
    ├── Plots/                   # Gráficas del Hito 2 (legacy)
    ├── Predicciones/            # Predicciones del Hito 2 (legacy)
    └── Predicciones_v2/         # Resultados experimentales finales
        ├── calibracion_predicciones/        # Predicciones calibradas
        ├── fusion_densidad/                 # NB11
        ├── fusion_densidad_mama/            # NB11b
        └── fusion_exhaustivo_mama/          # NB11c (108 modelos)
```

---

## Requisitos

### Sistema

- **Python 3.11** (recomendado, probado con miniconda)
- **GPU NVIDIA con ≥ 8 GB VRAM** (RTX 4070 Super usada en el desarrollo). El MLP usa CUDA; el resto de cabezas se ejecutan en CPU.
- **RAM ≥ 16 GB** (32 GB recomendado para configuraciones de alta dimensión).
- **Sistema operativo**: probado en Windows 11 (paths con `r'...'` y separadores Windows en los notebooks legacy; los actuales usan `os.path.join` cross‑platform).

### Dependencias Python

Ver `requirements.txt`. Las principales:

```
torch>=2.0          # PyTorch con CUDA 12.x
scikit-learn>=1.5
xgboost>=2.0
lightgbm>=4.0
numpy, pandas, matplotlib, scipy
pylibjpeg, pylibjpeg-libjpeg, pylibjpeg-openjpeg   # para DICOM
nbformat, jupyter   # ejecución de notebooks
```

### Datos externos (no incluidos en el repo)

- **VinDr‑Mammo dataset** (~5 000 estudios DICOM). Descarga en [PhysioNet](https://physionet.org/content/vindr-mammo/1.0.0/) tras aceptar el DUA.
- **Snapshot de AsymMirai** entrenado (`trained_asymmirai.pt`, ~46 MB). Disponible en el [repositorio oficial](https://github.com/jondonas/asym-mirai).

Coloca los datos en una estructura compatible con la configuración por defecto:

```
TFM_PROJECT_ROOT/
├── AsymMirai/
│   └── snapshots/
│       └── trained_asymmirai.pt
├── Data/
│   └── vindr-mammo/
│       ├── breast-level_annotations.csv
│       ├── finding_annotations.csv
│       └── images/...
└── (este repositorio clonado en alguna subcarpeta)
```

---

## Cómo reproducir los resultados

### 1. Configuración inicial

```bash
# Clonar el repositorio
git clone https://github.com/<usuario>/tfm-bi-rads-mamografia.git
cd tfm-bi-rads-mamografia

# Crear entorno virtual
conda create -n tfm python=3.11
conda activate tfm
pip install -r requirements.txt

# (Opcional) si quieres usar una ruta personalizada para los datos:
export TFM_PROJECT_ROOT=/ruta/a/tu/proyecto
# En Windows PowerShell:
$env:TFM_PROJECT_ROOT = "C:\ruta\a\tu\proyecto"
```

Por defecto, los notebooks asumen que se ejecutan desde la carpeta `notebooks/` y usan la **carpeta padre** como raíz del proyecto. Sobrescribe con `TFM_PROJECT_ROOT` si tu estructura es diferente.

### 2. Pipeline de ejecución recomendado

Los notebooks deben ejecutarse en orden. El pipeline completo desde cero tarda **aproximadamente 12 horas** en hardware similar al descrito:

| Paso | Notebook | Tiempo aprox. | Salida principal |
|---|---|---|---|
| Inspección inicial del modelo | `00–02` | < 10 min | — |
| Extracción features (GAP+GMP) | `03_extraccion_masiva.ipynb` | ~ 90 min | `Outputs/Features/X_view.npy` |
| Extracción features (pool 2×2) | `08_extraccion_pool22.ipynb` | ~ 45 min | `Outputs/Features/X_view_22.npy` |
| **Pipeline experimental (84 corridas)** | `09_pipeline_unificado.ipynb` | **~ 10 h** | `Outputs/Predicciones_v2/*.npy` |
| Evaluación con bootstrap + DeLong | `10_evaluacion_v2.ipynb` | ~ 25 min | `eval_completo_v2.csv` |
| Fusión densidad (estudio) | `11_fusion_densidad_v2.ipynb` | < 5 min | `fusion_resultados.csv` |
| Fusión densidad (mama, simétrico Hito 2) | `11b_fusion_densidad_mama.ipynb` | < 10 min | `fusion_resultados_mama.csv` |
| Fusión densidad exhaustiva (108 modelos) | `11c_fusion_densidad_exhaustivo.ipynb` | ~ 15 min | `fusion_exhaustivo_mama.csv` |
| Calibración post‑hoc | `12_calibracion_posthoc.ipynb` | < 5 min | `calibracion_resultados.csv` |

### 3. Notebooks legacy (Hito 2)

Los notebooks `04–07` son del primer hito del TFM y se mantienen en el repo por **trazabilidad histórica y para la comparación cuantitativa contra el pipeline mejorado del Hito 3**. Sus resultados están en `outputs/Predicciones/` y se referencian en NB10 / NB11x para los test DeLong contra el Hito 2. **No es necesario ejecutarlos para reproducir los resultados finales** del Hito 3.

---

## Citación

Si usas este código o sus hallazgos, por favor cita:

```bibtex
@mastersthesis{rodriguez2026tfm,
  author = {Rodríguez Rodríguez, Víctor},
  title  = {Modelos de Riesgo de Cáncer de Mama a partir de Mamografía de Cribado},
  school = {Universidad Internacional de Valencia (UIV)},
  year   = {2026},
  type   = {Trabajo Fin de Máster en Inteligencia Artificial}
}
```

---

## Licencia

Este código se distribuye bajo licencia **MIT**. Ver `LICENSE`.

El modelo **AsymMirai** mantiene su propia licencia original; consulta el repositorio oficial.

El dataset **VinDr‑Mammo** está sujeto a su propio Data Use Agreement de PhysioNet.

---

## Agradecimientos

A la directora **Karen López‑Linares** por la guía técnica durante todo el desarrollo. A los autores de **AsymMirai** (Donas et al.) por liberar pesos y código del modelo. Al equipo de **VinDr** por publicar el dataset.
