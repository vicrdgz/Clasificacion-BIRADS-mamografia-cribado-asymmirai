# `src/` — Módulos Python reutilizables

Tres módulos que encapsulan la lógica del pipeline experimental:

---

## `tfm_models.py` (legacy — Hito 2)

Definiciones de las cabezas MLP y LightGBM usadas en los notebooks `04_definicion_modelos.ipynb` y `05_entrenamiento_kfold.ipynb`. Se mantiene únicamente para reproducir las predicciones del Hito 2.

**Contenido principal**: clases `MLPHead`, `GBMHead` con interfaces `.fit()` y `.predict_proba()`.

**Usado por**: notebooks 04, 05, 06, 07.

---

## `tfm_pipeline.py` (Hito 3 — definitivo)

Pipeline unificado de entrenamiento usado por el `09_pipeline_unificado.ipynb`. Encapsula:

- Definición del **zoo de 9 cabezas** con sus hiperparámetros (`HEAD_FACTORIES`, `HEAD_GRIDS`).
- Funciones `train_logreg`, `train_rf`, `train_extratrees`, `train_histgb`, `train_xgb`, `train_lgbm`, `train_mlp` (cada una con sus particularidades de configuración).
- **Hold‑out interno 80/20** para grid search (`holdout_grid_search`).
- **K‑fold CV** con `StratifiedGroupKFold` (`train_kfold_unified`).
- Helpers de serialización (`save_predictions`).

**Constantes clave**:
- `SEED = 42` — semilla global
- `HEAD_NAMES` — orden canónico de las cabezas
- `HEAD_GRIDS` — grids reducidos basados en evidencia empírica (ej. L1 sin `C=10.0` porque nunca ganó)

**Notas técnicas**:
- LogReg L1 usa `saga` con `tol=5e-3` (más robusto que liblinear en alta dimensionalidad con muchas features irrelevantes; documentado en NB09).
- ElasticNet usa `saga` con `tol=1e-3` pero **solo se evaluó a nivel mama** por coste computacional (4 corridas × ~74 min cada una).
- XGBoost se configura con `verbose=False` y `early_stopping_rounds=30`.

**Usado por**: notebook 09.

---

## `tfm_eval.py` (Hito 3 — definitivo)

Funciones de evaluación estadística rigurosa usadas por los notebooks 10, 11, 11b, 11c y 12.

### Funciones públicas

```python
bootstrap_metric_ci(y_true, y_pred, metric_fn, n_boot=1000, ci=0.95, seed=42)
    # IC% bootstrap percentil para cualquier métrica binaria.
    # Devuelve (point_estimate, ci_lower, ci_upper).

ece_score(y_true, y_pred, n_bins=10)
    # Expected Calibration Error con binning equiespaciado.

delong_test(y_true, pred_a, pred_b)
    # Test pareado de DeLong: H0 AUC_a == AUC_b sobre los mismos datos.
    # Implementación O(n²) de Sun & Xu (2014).
    # Devuelve dict con auc_a, auc_b, delta, se_delta, z, p_value.

aggregate_breast_to_study(pred_breast, study_ids, agg='max')
    # Agrega predicciones nivel mama → nivel estudio.
    # pred_breast viene ordenado [L_1..L_N, R_1..R_N].
    # agg in {'max', 'mean'}.

compute_full_metrics(y_true, y_pred, n_boot=1000, seed=42)
    # Calcula AUC+IC, AP+IC, Brier y ECE en una llamada.
```

### Constantes

```python
SEED = 42
```

**Usado por**: notebooks 10, 11, 11b, 11c, 12.

---

## Convenciones internas

- **Orden de predicciones a nivel mama**: siempre `[L_1, L_2, ..., L_N, R_1, R_2, ..., R_N]` (todas las izquierdas primero, luego todas las derechas). Esta convención debe respetarse al cargar predicciones para que `aggregate_breast_to_study` funcione correctamente.
- **`StratifiedGroupKFold` con `groups=study_id`** a nivel mama: garantiza que L y R del mismo estudio caen en el mismo fold, evitando leak.
- **Bootstrap**: muestrea con reemplazo manteniendo el tamaño de la muestra; descarta réplicas donde toda la muestra es de una sola clase.
