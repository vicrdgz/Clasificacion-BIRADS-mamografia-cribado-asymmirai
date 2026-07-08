# Notebooks — guía de ejecución

Los **13 notebooks** de este directorio implementan el pipeline experimental del TFM: extracción de features con el backbone congelado de AsymMirai, entrenamiento de 84 configuraciones base sobre VinDr-Mammo, evaluación con bootstrap y DeLong, fusión con densidad mamaria, calibración post-hoc y análisis complementarios.

El **NB00** contiene una descripción de alto nivel del pipeline. Este README complementa esa descripción con los inputs y outputs concretos de cada notebook y la información de ejecución.

## `00_overview.ipynb`
Descripción del pipeline experimental: formulación del problema, estrategia de reutilización del backbone congelado, 10 configuraciones × 9 cabezas = 84 modelos base, protocolo de validación cruzada, extensiones (fusión con densidad y calibración post-hoc), y modelos finales.

**Inputs**: ninguno (documentación + sanity check de rutas).
**Outputs**: creación de las carpetas `outputs/{Features, Predicciones, Models, Plots}` si no existen.
**Tiempo**: < 1 s.

## `01_inspeccion_modelo.ipynb`
Carga del snapshot preentrenado de AsymMirai. Verifica accesibilidad al backbone, identifica los parámetros de `stretch` aprendidos, valida que `alignment_space = None` y confirma el shape del embedding `(1, 512, 52, 64)` por vista.

**Inputs**: `AsymMirai/snapshots/trained_asymmirai.pt`.
**Outputs**: información impresa (no genera archivos).
**Tiempo**: < 30 s.

## `02_extraccion_un_estudio.ipynb`
Validación del pipeline sobre un único estudio de VinDr-Mammo. Confirma que las etapas DICOM, preprocesado, backbone, Punto A, Punto B con stretch y reducción GAP+GMP producen los shapes esperados.

**Inputs**: `AsymMirai/snapshots/trained_asymmirai.pt`, un DICOM de `Data/vindr-mammo/images/`.
**Outputs**: `outputs/Features/preprocesado_etapas.png` (Ilustración 9 de la memoria).
**Tiempo**: < 1 min.

## `03_extraccion_masiva.ipynb`
Extracción masiva de features con **GAP + GMP** sobre los 5.000 estudios de VinDr-Mammo. Cada vista produce un vector de 1024 dimensiones (512 GAP + 512 GMP); cada par bilateral produce un mapa de asimetría con la misma dimensionalidad. Reanudación automática si el proceso se interrumpe.

**Inputs**: `Data/vindr-mammo/breast-level_annotations.csv`, DICOMs en `Data/vindr-mammo/images/`, `AsymMirai/snapshots/trained_asymmirai.pt`.
**Outputs**: `outputs/Features/X_view.npy` `(4999, 4, 1024)`, `X_asym.npy` `(4999, 2, 1024)`, `metadata.csv`, `done_studies.txt`, `caracterizacion_dataset.png` (Ilustración 5 de la memoria).
**Tiempo**: ~90 min en GPU.

## `08_extraccion_pool22.ipynb`
Re-extracción con **pool adaptativo 2×2** (AdaptiveAvg + AdaptiveMax). Preserva información espacial gruesa: cada vista pasa de 1024 dims (GAP+GMP) a 4096 dims (2×2×2). Misma lógica de reanudación que NB03.

**Inputs**: mismos que NB03 + `metadata.csv` producido por NB03 (opcional, se reutiliza).
**Outputs**: `outputs/Features/X_view_22.npy` `(4999, 4, 4096)`, `X_asym_22.npy` `(4999, 2, 4096)`, `done_studies_22.txt`.
**Tiempo**: ~45 min en GPU.

## `09_pipeline_unificado.ipynb`
**Núcleo experimental del TFM.** Entrena **84 modelos base**:

- **10 configuraciones** = 5 inputs × 2 poolings.
  - Inputs: `E_A`, `E_B`, `E_AB` (nivel estudio); `M_A`, `M_AB` (nivel mama).
  - Poolings: `gg` (GAP+GMP) y `22` (pool adaptativo 2×2).
- **9 cabezas**: `logreg_l1`, `logreg_l2`, `logreg_en` (ElasticNet, solo mama), `rf`, `extratrees`, `histgb`, `xgb`, `lgbm`, `mlp`.

Protocolo por configuración: `StratifiedKFold` (estudio) o `StratifiedGroupKFold` (mama) con K=5, hold-out interno 80/20 para grid search, reentrenamiento con el training pool completo, predicción OOF + ensemble sobre test.

**Inputs**: `X_view.npy`, `X_asym.npy`, `X_view_22.npy`, `X_asym_22.npy`, `metadata.csv`.
**Outputs**: en `outputs/Predicciones_v2/`, por cada corrida `{config}__{head}_oof.npy`, `_test.npy`, `_meta.json`; al final, `resumen_v2.csv`.
**Tiempo**: 5-8 h.

## `10_evaluacion_v2.ipynb`
Análisis detallado sobre las 84 corridas:

- Métricas con IC 95 % bootstrap: AUC, AP, Brier, ECE.
- DeLong pareado: `pool_2x2 vs GAP+GMP` (42 comparaciones) y `A+B vs A` (34 comparaciones).
- Análisis estratificado por densidad, edad y BI-RADS original.
- Matrices de confusión de los dos modelos finales calibrados.
- Análisis post-hoc de agregación mama a estudio.
- Curvas ROC y PR del top 5.

Produce los inputs para las Ilustraciones 11, 16, 17, 18, 19 y 20 de la memoria.

**Inputs**: predicciones de NB09, `metadata.csv`, `breast-level_annotations.csv` (para BI-RADS por vista y edad DICOM).
**Outputs**: en `outputs/Predicciones_v2/`: `eval_completo_v2.csv`, `delong_pareado_v2.csv`, `eval_densidad_v2.csv`, `eval_edad_v2.csv`, `auc_pairwise_birads.csv`, `eval_densidad_modelo_final.csv`, `eval_edad_modelo_final.csv`, `matrices_confusion_modelo_final.csv`, `post_hoc_mama_to_estudio.csv`. Figuras PNG en `outputs/Plots/`.
**Tiempo**: ~25 min.

## `11_fusion_densidad_v2.ipynb`
Fusión con densidad mamaria **a nivel estudio**: los 3 mejores candidatos del NB09 se agregan con `max(L,R)` y luego se combinan con la densidad codificada como one-hot (5 features totales).

3 candidatos × 3 modelos de fusión (LogReg, HistGB, MLP) = 9 modelos.

**Inputs**: predicciones NB09 de los 3 candidatos base.
**Outputs**: `fusion_resultados.csv`, `fusion_curvas_roc.png`, `fusion_densidad/{candidato}__fuse_{modelo}_test.npy`.
**Tiempo**: < 5 min.

## `11b_fusion_densidad_mama.ipynb`
Fusión con densidad **a nivel mama** (antes de agregar). Complementa NB11 permitiendo responder si el resultado observado es robusto al nivel donde se aplica la fusión.

**Inputs**: predicciones NB09 de los 3 candidatos base (nivel mama, sin agregar).
**Outputs**: `fusion_resultados_mama.csv`, `fusion_resultados_mama_agregadas.csv`, `fusion_mama_curvas_roc.png`, `fusion_densidad_mama/*.npy`.
**Tiempo**: < 10 min.

## `11c_fusion_densidad_exhaustivo.ipynb`
**Evaluación exhaustiva** de la fusión con densidad sobre las 36 configuraciones base a nivel mama (4 configs × 9 cabezas) × 3 modelos de fusión = **108 modelos**. Cierra cualquier sospecha de selección sesgada de candidatos en NB11 y NB11b.

Análisis agregados por cabeza base, pooling, input y modelo de fusión.

Hallazgo principal: 2/108 fusiones con mejora estadísticamente significativa; 20/108 con empeoramiento significativo. Reproduce la Tabla 15 y la Ilustración 14 de la memoria (sección 5.4.4).

**Inputs**: predicciones NB09 a nivel mama (36 corridas), `resumen_v2.csv`.
**Outputs**: `fusion_exhaustivo_mama.csv` (144 filas), `fusion_exhaustivo_resumen.csv`, `fusion_exhaustivo_top10.png`, `fusion_exhaustivo_mama/*.npy` (108 predicciones).
**Tiempo**: ~15 min.

## `12_calibracion_posthoc.ipynb`
Calibración **Platt scaling** e **isotónica** sobre los dos modelos finales:

1. `M_A_22 + xgb` agregado a estudio - modelo triaje recomendado.
2. `M_A_gg + mlp` a nivel mama - alternativo con calibración base más pobre.

Métricas antes/después + DeLong para verificar preservación del AUC + reliability diagrams. Reproduce la Tabla 16 y la Ilustración 15 de la memoria (sección 5.5).

Método recomendado en ambos casos: **Platt** (preserva AUC, minimiza ECE).

**Inputs**: predicciones NB09 de `M_A_22__xgb` y `M_A_gg__mlp`.
**Outputs**: `calibracion_resultados.csv`, `calibracion_tabla_memoria.csv`, `calibracion_reliability.png`, `calibracion_predicciones/*.npy`.
**Tiempo**: < 5 min.

## `13_ejemplos_birads.ipynb`
Genera las dos figuras ilustrativas del capítulo 3.1 de la memoria:

- Figura 1×5 con un ejemplo real de cada categoría BI-RADS (1 a 5), con bounding box superpuesto en las categorías sospechosas (Ilustración 2).
- Panel 2×2 con las cuatro vistas estándar de un examen de mamografía de cribado (Ilustración 1).

**Inputs**: `breast-level_annotations.csv`, `finding_annotations.csv`, DICOMs de `Data/vindr-mammo/images/`.
**Outputs**: `outputs/Features/ejemplos_birads.png`, `outputs/Features/mamografia_4_vistas.png`.
**Tiempo**: < 2 min.

## `14_analisis_sonda_edad.ipynb`
Análisis complementario que responde a la pregunta: ¿predice el modelo la edad implícitamente? Tres análisis:

1. **Sonda lineal** PCA(200) + Ridge sobre las features `M_A_22` para predecir edad.
2. **Correlación** entre el score del modelo final calibrado y la edad.
3. **Base rate + clasificador trivial** "solo edad" (LogReg con edad como única covariable) comparado con el modelo final por rango de edad.

Reproduce la Ilustración 18 de la memoria y constituye la subsección 5.7.1.

**Inputs**: `X_view_22.npy`, `X_asym_22.npy`, `metadata.csv`, scores calibrados del NB12, `breast-level_annotations.csv` (para edad DICOM).
**Outputs**: `resumen_analisis_edad.csv` y 3 figuras PNG en `outputs/Plots/`.
**Tiempo**: ~5 min.

---

## Configuración de rutas

Por defecto, los notebooks asumen que la carpeta raíz del proyecto es la carpeta padre de `notebooks/` (es decir, el directorio que contiene `notebooks/`, `src/`, `outputs/`). Esto se detecta automáticamente con:

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

## Orden de ejecución recomendado

```
00 -> 01 -> 02 -> 03 -> 08 -> 09 -> 10 -> 11 -> 11b -> 11c -> 12 -> 13 -> 14
```

NB09 es el paso más costoso (5-8 h). El resto del pipeline se ejecuta en minutos si los archivos de features (`X_view.npy`, `X_asym.npy`, `X_view_22.npy`, `X_asym_22.npy`) y las predicciones de `outputs/Predicciones_v2/` ya existen.
