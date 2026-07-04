# Notebooks — guía de ejecución

Los **15 notebooks** de este directorio cubren todo el pipeline experimental del TFM. Se dividen en dos fases:

- **Notebooks 00–07 (Hito 2)**: trabajo exploratorio inicial — establece el pipeline base.
- **Notebooks 08–12 (Hito 3, definitivo)**: refactorización completa con protocolo unificado y evaluación exhaustiva.

Los notebooks legacy del Hito 2 (04–07) **no son necesarios** para reproducir los resultados finales, pero se mantienen porque:
1. Sus predicciones se usan como referencia en `outputs/Predicciones/` para los tests DeLong del Hito 3.
2. Documentan el proceso evolutivo del TFM (valor metodológico).

---

## Fase 1 — Setup y exploración inicial (Hito 2)

### `00_overview.ipynb`
Introducción de alto nivel al setup. Resumen de objetivos, arquitectura de AsymMirai, descripción de VinDr‑Mammo.

**Entrada**: ninguna  
**Salida**: solo documentación

### `01_inspeccion_modelo.ipynb`
Carga el snapshot pre‑entrenado de AsymMirai y explora su arquitectura: capas, parámetros, dimensiones de entrada/salida, comportamiento de la cabeza interna `mirai_localized_dif_head`.

**Entrada**: `AsymMirai/snapshots/trained_asymmirai.pt`  
**Salida**: información de modelo (impresa); no genera archivos

### `02_extraccion_un_estudio.ipynb`
Prueba de extracción de features sobre un único estudio de VinDr. Verifica el flujo completo: lectura DICOM → preprocesado → forward → extracción de las 4 vistas + asimetría bilateral.

**Entrada**: 1 estudio de `Data/vindr-mammo/images/`  
**Salida**: información (impresa); valida el pipeline antes de extracción masiva

### `03_extraccion_masiva.ipynb` — **importante**
Extracción masiva de features con **GAP+GMP** (Global Average + Max Pooling) sobre los ~5 000 estudios de VinDr‑Mammo.

- Cada estudio produce 4 vistas (L/R × CC/MLO) × 1024 dims = 4096 features de vista.
- Cada estudio produce 2 mapas de asimetría (CC/MLO) × 1024 dims = 2048 features de asimetría.
- Aplica normalización con estadísticas de EMBED (MIRAI_MEAN=7699.5, MIRAI_STD=11765.06 sobre escala 16‑bit).

**Entrada**: dataset completo VinDr‑Mammo  
**Salida**: 
- `outputs/Features/X_view.npy` `(N=4999, 4, 1024)` — vistas con GAP+GMP
- `outputs/Features/X_asym.npy` `(N=4999, 2, 1024)` — asimetría con GAP+GMP
- `outputs/Features/metadata.csv` — splits, etiquetas y densidad por estudio
**Tiempo**: ~90 min en RTX 4070 Super

### Notebooks 04–07 (legacy Hito 2)

> ⚠️ **No necesarios para reproducir los resultados finales** (Hito 3). Mantenidos para trazabilidad histórica.

- `04_definicion_modelos.ipynb` — Define cabezas MLP y LightGBM. Usa `src/tfm_models.py`.
- `05_entrenamiento_kfold.ipynb` — Entrena los 8 modelos del Hito 2 (M_A_mlp, M_A_gbm, E_A_mlp, etc.) con K=5.
- `06_evaluacion.ipynb` — Evaluación inicial con bootstrap y DeLong. Genera `Plots/roc_comparacion.png`.
- `07_fusion_densidad.ipynb` — Primer experimento de fusión con densidad. Generó AUC=0.7067 para M_A_mlp_dens (referencia que se compara en NB11/11b/11c).

---

## Fase 2 — Pipeline definitivo (Hito 3)

### `08_extraccion_pool22.ipynb`
Re‑extracción de features con **adaptive pool 2×2** (en lugar de GAP+GMP). Cada vista produce ahora 4×1024=4096 features, conservando información espacial.

**Entrada**: dataset VinDr‑Mammo, modelo AsymMirai  
**Salida**:
- `outputs/Features/X_view_22.npy` `(N=4999, 4, 4096)` — vistas con pool 2×2
- `outputs/Features/X_asym_22.npy` `(N=4999, 2, 4096)` — asimetría con pool 2×2  
**Tiempo**: ~45 min

### `09_pipeline_unificado.ipynb` — **NÚCLEO EXPERIMENTAL**
Entrena **84 modelos** bajo un protocolo unificado:

- **10 configuraciones** = 5 inputs × 2 poolings:
  - Inputs: `M_A`, `M_AB`, `E_A`, `E_B`, `E_AB`
  - Poolings: `gg` (GAP+GMP) y `22` (pool 2×2)
- **9 cabezas**: LogReg L1/L2/EN, RandomForest, ExtraTrees, HistGradBoosting, XGBoost, LightGBM, MLP
- (no todas las cabezas se entrenan en todas las configs por razones de coste computacional)

**Protocolo por configuración**:
1. K=5 StratifiedGroupKFold con `study_id` como grupo
2. Por fold: hold‑out interno 80/20 para grid search
3. Reentrenamiento con training pool completo
4. Predicción OOF + test ensemble

**Entrada**: `X_view*.npy`, `X_asym*.npy`, `metadata.csv`  
**Salida**: `outputs/Predicciones_v2/{config}__{head}_{oof,test,meta}.{npy,json}` + `resumen_v2.csv`  
**Tiempo**: ~10 h en RTX 4070 Super

### `10_evaluacion_v2.ipynb` — **Análisis estadístico riguroso**

Sobre las 84 predicciones del NB09:

1. AUC + IC 95 % bootstrap (n=1000), AP + IC, Brier, ECE para cada corrida
2. Tests DeLong pareados:
   - GAP+GMP vs pool 2×2 (42 comparaciones)
   - A solo vs A+B (34 comparaciones)
3. Comparación cuantitativa contra Hito 2 (M_A_mlp, M_A_mlp_dens, etc.)
4. Análisis estratificado por densidad mamaria (A/B/C/D) en hold‑out
5. Análisis post‑hoc M→E con `max(L,R)`
6. Curvas ROC y PR del top 5

**Entrada**: predicciones del NB09 + predicciones legacy del Hito 2  
**Salida**: `eval_completo_v2.csv`, `delong_pareado_v2.csv`, `eval_densidad_v2.csv`, `post_hoc_mama_to_estudio.csv`, `curvas_top.png`  
**Tiempo**: ~25 min

### `11_fusion_densidad_v2.ipynb`
Replica el experimento de fusión con densidad del Hito 2, pero **a nivel estudio**: las predicciones a nivel mama se agregan con `max(L, R)` antes de combinarlas con la densidad.

3 candidatos × 3 modelos de fusión (LogReg, HistGB, MLP) = 9 modelos.

**Entrada**: predicciones NB09  
**Salida**: `fusion_resultados.csv`, `fusion_vs_hito2.csv`, `fusion_curvas_roc.png`  
**Tiempo**: < 5 min

### `11b_fusion_densidad_mama.ipynb`
Versión simétrica al NB07 del Hito 2: la fusión se aplica **a nivel mama** (antes de agregar). Compara la mejor fusión NB11b directamente contra `M_A_mlp_dens` del Hito 2 (ambos al mismo nivel de evaluación).

**Entrada**: predicciones NB09  
**Salida**: `fusion_resultados_mama.csv`, `fusion_mama_vs_hito2.csv`, `fusion_resultados_mama_agregadas.csv`, `fusion_mama_curvas_roc.png`  
**Tiempo**: < 10 min

### `11c_fusion_densidad_exhaustivo.ipynb` — **rigor metodológico exhaustivo**
Evalúa la fusión con densidad sobre **las 36 configuraciones base** del NB09 (a nivel mama) × 3 modelos de fusión = **108 modelos**. Cierra cualquier sospecha de selección sesgada de candidatos en NB11/NB11b.

Análisis agregados por cabeza, pooling, input y modelo de fusión.

**Hallazgo clave**: solo 2/108 fusiones con mejora estadísticamente significativa, ambas sobre modelos base mediocres (AUC<0.62).

**Entrada**: predicciones NB09  
**Salida**: `fusion_exhaustivo_mama.csv`, `fusion_exhaustivo_resumen.csv`, `fusion_exhaustivo_mama_vs_hito2.csv`, `fusion_exhaustivo_top10.png`  
**Tiempo**: ~15 min

### `12_calibracion_posthoc.ipynb` — **modelo final**

Aplica **Platt scaling** y **regresión isotónica** a los dos modelos finales:

1. `M_A_22 + xgb` agregado a estudio — modelo clínico recomendado
2. `M_A_gg + mlp` a nivel mama — peor calibrado del top, caso didáctico

Métricas antes/después + DeLong para verificar preservación de AUC + reliability diagrams.

**Hallazgo**: Platt > isotónica en este setting de baja prevalencia (isotonic colapsa el rango de predicciones y pierde AUC significativamente).

**Entrada**: predicciones NB09 (mama)  
**Salida**: `calibracion_resultados.csv`, `calibracion_tabla_memoria.csv`, `calibracion_reliability.png`, predicciones calibradas  
**Tiempo**: < 5 min

---

## Configuración de rutas

Por defecto, los notebooks asumen que la **carpeta raíz del proyecto** es la carpeta padre de `notebooks/` (es decir, el directorio que contiene `notebooks/`, `src/`, `outputs/`). Esto se detecta automáticamente con:

```python
BASE = os.environ.get('TFM_PROJECT_ROOT',
                      os.path.abspath(os.path.join(os.getcwd(), '..')))
```

Si tu estructura de carpetas es diferente o quieres apuntar a otra ubicación (por ejemplo, los datos están en otro disco), define la variable de entorno antes de abrir Jupyter:

```bash
# Linux / macOS
export TFM_PROJECT_ROOT=/ruta/personalizada

# Windows PowerShell
$env:TFM_PROJECT_ROOT = "D:\datos\tfm"

# Windows CMD
set TFM_PROJECT_ROOT=D:\datos\tfm
```

---

## Orden de ejecución recomendado para reproducción completa

```
00 → 01 → 02 → 03 → 08 → 09 → 10 → 11 → 11b → 11c → 12
```

Los notebooks 04–07 (Hito 2) son opcionales — solo necesarios si se quieren regenerar las predicciones de referencia para los DeLong contra el Hito 2.
