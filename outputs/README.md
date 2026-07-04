# `outputs/` — Resultados generados por los notebooks

Este directorio contiene **predicciones**, **tablas de métricas** y **gráficas** producidas por la ejecución del pipeline.

Los archivos están organizados por hito experimental:

- `Plots/` y `Predicciones/` → outputs del **Hito 2** (legacy, conservados para comparación contra el Hito 3)
- `Predicciones_v2/` → outputs del **Hito 3** (pipeline definitivo)

---

## `Plots/` — Gráficas del Hito 2

| Archivo | Generado por | Descripción |
|---|---|---|
| `auc_ci_bootstrap.csv` | NB06 | IC 95 % bootstrap de los 8 modelos del Hito 2 |
| `auc_densidad_vs_sin.csv` | NB07 | Comparación AUC con/sin fusión densidad (Hito 2) |
| `delong_pareado.csv` | NB06 | Tests DeLong de los modelos del Hito 2 |
| `delong_pareado_densidad.csv` | NB07 | Tests DeLong fusión vs base (Hito 2) |
| `roc_comparacion.png` | NB06 | Curvas ROC de los 8 modelos Hito 2 |
| `roc_densidad.png` | NB07 | Curvas ROC con/sin densidad (Hito 2) |

---

## `Predicciones/` — Predicciones del Hito 2

Para cada uno de los 8 modelos del Hito 2 (4 base × 2 cabezas: MLP y GBM) se guardan dos arrays:

- `{config}_{head}_oof.npy` — predicciones OOF sobre el training pool
- `{config}_{head}_test.npy` — predicciones ensemble sobre el test pool

Configuraciones presentes: `E_A_{mlp,gbm}`, `E_B_{mlp,gbm}`, `M_A_{mlp,gbm}`, `M_A_{mlp,gbm}_dens` (con fusión densidad).

| Archivo | Descripción |
|---|---|
| `resumen_aucs.csv` | Tabla resumen con AUC OOF y test de cada modelo |

Estas predicciones son la **referencia "Hito 2"** usada en los tests DeLong de los NB10, NB11, NB11b y NB11c.

---

## `Predicciones_v2/` — Resultados del Hito 3 (definitivo)

### Predicciones base (NB09)

168 archivos `.npy` (84 modelos × 2: OOF + test) + 84 archivos `_meta.json` con hiperparámetros y métricas de cada corrida.

Naming: `{config}__{head}_{oof,test,meta}.{npy,json}` (doble guión bajo separa config de head).

Configuraciones: `M_A_{22,gg}`, `M_AB_{22,gg}`, `E_A_{22,gg}`, `E_B_{22,gg}`, `E_AB_{22,gg}` (10 configs).

Cabezas: `logreg_l1`, `logreg_l2`, `logreg_en` (solo mama), `histgb`, `lgbm`, `xgb`, `rf`, `extratrees`, `mlp` (9 cabezas).

### Tablas de evaluación (NB10)

| Archivo | Descripción |
|---|---|
| `resumen_v2.csv` | Resumen del pipeline NB09: 84 filas con AUC OOF, test, hiperparámetros |
| `eval_completo_v2.csv` | Métricas completas (AUC+IC, AP+IC, Brier, ECE) de las 84 corridas |
| `delong_pareado_v2.csv` | 76 tests DeLong pareados (pool_2x2_vs_gg, AB_vs_A) |
| `eval_densidad_v2.csv` | AUC estratificado por densidad (A/B/C/D) sobre top 5 por nivel |
| `post_hoc_mama_to_estudio.csv` | Análisis post‑hoc: agregar M→E con max vs E directo |
| `curvas_top.png` | Curvas ROC y PR del top 5 (mama y estudio) |

### Fusión con densidad — nivel estudio (NB11)

| Archivo | Descripción |
|---|---|
| `fusion_resultados.csv` | 3 candidatos × 3 modelos fusión, métricas + DeLong vs base |
| `fusion_vs_hito2.csv` | DeLong cada fusión NB11 contra `M_A_mlp_dens` Hito 2 |
| `fusion_curvas_roc.png` | Curvas ROC de base + 3 fusiones por candidato |
| `fusion_densidad/` | 9 archivos `.npy` con predicciones de fusión calibrada (test) |

### Fusión con densidad — nivel mama (NB11b, simétrico Hito 2)

| Archivo | Descripción |
|---|---|
| `fusion_resultados_mama.csv` | Métricas a nivel mama (réplica del setting NB07) |
| `fusion_mama_vs_hito2.csv` | DeLong vs Hito 2 (mismo nivel: mama) |
| `fusion_resultados_mama_agregadas.csv` | Fusiones del NB11b agregadas posteriormente a nivel estudio |
| `fusion_mama_curvas_roc.png` | Curvas ROC top con referencia Hito 2 |
| `fusion_densidad_mama/` | 9 archivos `.npy` con predicciones de fusión a nivel mama |

### Fusión exhaustiva (NB11c) — 108 modelos

| Archivo | Descripción |
|---|---|
| `fusion_exhaustivo_mama.csv` | 144 filas (36 base + 108 fusión) con todas las métricas |
| `fusion_exhaustivo_resumen.csv` | Agregados por cabeza, pooling, input, modelo de fusión |
| `fusion_exhaustivo_mama_vs_hito2.csv` | DeLong top 10 vs Hito 2 |
| `fusion_exhaustivo_top10.png` | Curvas ROC top 10 absoluto con referencia Hito 2 |
| `fusion_exhaustivo_mama/` | 108 archivos `.npy` con todas las predicciones de fusión |

### Calibración post‑hoc (NB12)

| Archivo | Descripción |
|---|---|
| `calibracion_resultados.csv` | Métricas antes/después de Platt e isotonic para los 2 candidatos finales |
| `calibracion_tabla_memoria.csv` | Versión compacta lista para incluir en la memoria |
| `calibracion_reliability.png` | Reliability diagrams + histogramas (binning por cuantiles) |
| `calibracion_predicciones/` | 4 archivos `.npy` con predicciones calibradas finales |

---

## Convenciones de naming

- `{config}__{head}_{oof,test}.npy` — doble guión bajo separa la configuración del nombre de la cabeza
- `*_oof.npy` — predicciones out‑of‑fold (sobre el training pool, 3 999 estudios o 7 998 mamas)
- `*_test.npy` — predicciones ensemble de los 5 folds sobre el test pool (1 000 estudios o 2 000 mamas)
- `*_meta.json` — hiperparámetros ganadores por fold y métricas detalladas
- `eval_*.csv`, `fusion_*.csv`, `calibracion_*.csv` — tablas con métricas y resultados estadísticos
- `*.png` — figuras finales (dpi=140, listas para memoria/paper)

## Reproducción

Para regenerar todos los outputs desde cero, ejecutar los notebooks en el orden indicado en `notebooks/README.md`. El pipeline completo tarda ~12 h en RTX 4070 Super.
