"""
tfm_eval - Funciones de evaluacion para el TFM.

TFM - Master Universitario en Inteligencia Artificial - VIU 2025-2026
Victor Rodriguez Rodriguez

Componentes:
    - bootstrap_metric_ci: IC 95% por bootstrap percentil para cualquier metrica.
    - ece_score: Expected Calibration Error.
    - delong_test: test pareado de DeLong para comparar dos AUCs sobre los mismos datos.
    - aggregate_breast_to_study: agregar predicciones a nivel mama hacia nivel estudio.
    - compute_full_metrics: bloque de metricas (AUC, AP, Brier, ECE) con IC bootstrap.

Todas las funciones operan sobre arrays numpy de y_true (0/1) y y_pred (probabilidades).
"""

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
import scipy.stats as st

SEED = 42


# ============================================================================
# BOOTSTRAP IC
# ============================================================================

def bootstrap_metric_ci(y_true, y_pred, metric_fn, n_boot=1000, ci=0.95, seed=SEED):
    """
    Calcula intervalo de confianza percentil para una métrica vía bootstrap.

    Parámetros
    ----------
    y_true : array (n,) — etiquetas binarias 0/1
    y_pred : array (n,) — probabilidades o scores continuos
    metric_fn : callable (y_true, y_pred) -> float
    n_boot : int — número de remuestreos (default 1000)
    ci : float — nivel de confianza (default 0.95)
    seed : int — semilla para reproducibilidad

    Devuelve
    --------
    (point_estimate, ci_lower, ci_upper)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rng = np.random.RandomState(seed)
    n = len(y_true)
    scores = []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        try:
            scores.append(metric_fn(y_true[idx], y_pred[idx]))
        except Exception:
            continue
    scores = np.array(scores)
    point = metric_fn(y_true, y_pred)
    alpha = 1 - ci
    lower = np.percentile(scores, 100 * alpha / 2)
    upper = np.percentile(scores, 100 * (1 - alpha / 2))
    return point, lower, upper


# ============================================================================
# CALIBRACIÓN: ECE
# ============================================================================

def ece_score(y_true, y_pred, n_bins=10):
    """
    Expected Calibration Error con binning equiespaciado.

    ECE = sum_{b} (|B_b| / N) · |acc(B_b) - conf(B_b)|

    donde acc(B_b) es la fracción de positivos en el bin b y conf(B_b)
    es la probabilidad media predicha en ese bin.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = (y_pred > lo) & (y_pred <= hi)
        if i == 0:
            in_bin = (y_pred >= lo) & (y_pred <= hi)
        n_in_bin = in_bin.sum()
        if n_in_bin == 0:
            continue
        acc_in_bin = y_true[in_bin].mean()
        conf_in_bin = y_pred[in_bin].mean()
        ece += (n_in_bin / n) * abs(acc_in_bin - conf_in_bin)
    return float(ece)


# ============================================================================
# DELONG TEST PAREADO
# ============================================================================

def _delong_structural_components(y_true, y_pred):
    """
    Componentes de DeLong: vectores V10 (sobre positivos) y V01 (sobre negativos).
    Implementación O(n²) — adecuada para n ≤ 10k. Para datasets más grandes habría
    que usar la versión con midrank de Sun & Xu (2014).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    pos = y_pred[y_true == 1]
    neg = y_pred[y_true == 0]
    n_pos, n_neg = len(pos), len(neg)

    # V10: para cada positivo i, fracción de negativos por debajo (con empates 0.5)
    v10 = np.empty(n_pos)
    for i in range(n_pos):
        v10[i] = (np.sum(neg < pos[i]) + 0.5 * np.sum(neg == pos[i])) / n_neg

    # V01: para cada negativo j, fracción de positivos por encima (con empates 0.5)
    v01 = np.empty(n_neg)
    for j in range(n_neg):
        v01[j] = (np.sum(pos > neg[j]) + 0.5 * np.sum(pos == neg[j])) / n_pos

    auc = v10.mean()  # equivalente a 1 - v01.mean()
    return auc, v10, v01


def delong_test(y_true, pred_a, pred_b):
    """
    Test pareado de DeLong para H0: AUC_a == AUC_b sobre los mismos datos.

    Devuelve
    --------
    dict con auc_a, auc_b, delta = auc_a - auc_b, se_delta, z, p_value.
    """
    y_true = np.asarray(y_true)
    pred_a = np.asarray(pred_a)
    pred_b = np.asarray(pred_b)

    auc_a, v10_a, v01_a = _delong_structural_components(y_true, pred_a)
    auc_b, v10_b, v01_b = _delong_structural_components(y_true, pred_b)

    n_pos = (y_true == 1).sum()
    n_neg = (y_true == 0).sum()

    # Matrices de covarianza de V10 y V01 entre los dos modelos
    var_a = v10_a.var(ddof=1) / n_pos + v01_a.var(ddof=1) / n_neg
    var_b = v10_b.var(ddof=1) / n_pos + v01_b.var(ddof=1) / n_neg
    cov_10 = np.cov(v10_a, v10_b, ddof=1)[0, 1]
    cov_01 = np.cov(v01_a, v01_b, ddof=1)[0, 1]
    cov_ab = cov_10 / n_pos + cov_01 / n_neg

    var_diff = var_a + var_b - 2 * cov_ab
    delta = auc_a - auc_b

    if var_diff <= 0:
        # Mismas predicciones o caso degenerado
        return {
            'auc_a': float(auc_a), 'auc_b': float(auc_b),
            'delta': float(delta), 'se_delta': 0.0, 'z': 0.0,
            'p_value': 1.0,
        }
    se = np.sqrt(var_diff)
    z = delta / se
    p_value = 2 * (1 - st.norm.cdf(abs(z)))
    return {
        'auc_a': float(auc_a), 'auc_b': float(auc_b),
        'delta': float(delta), 'se_delta': float(se), 'z': float(z),
        'p_value': float(p_value),
    }


# ============================================================================
# AGREGACIÓN MAMA -> ESTUDIO
# ============================================================================

def aggregate_breast_to_study(pred_breast, study_ids, agg='max'):
    """
    Agrega predicciones a nivel mama hacia nivel estudio.

    Parámetros
    ----------
    pred_breast : array (n_mamas,) — predicciones a nivel mama, en orden [L₁, L₂, ..., R₁, R₂, ...]
                  (mitad de las filas son mama izquierda, mitad derecha, mismo orden que study_ids)
    study_ids : array (n_estudios,) — IDs de estudio en el orden original
    agg : 'max' | 'mean' — función de agregación

    Devuelve
    --------
    (pred_estudio, study_ids_orden) — array (n_estudios,) con la predicción agregada por estudio
    """
    n = len(study_ids)
    if len(pred_breast) != 2 * n:
        raise ValueError(f'pred_breast debe tener 2*{n}={2*n} elementos, tiene {len(pred_breast)}')
    pred_L = pred_breast[:n]
    pred_R = pred_breast[n:]
    if agg == 'max':
        pred_estudio = np.maximum(pred_L, pred_R)
    elif agg == 'mean':
        pred_estudio = (pred_L + pred_R) / 2
    else:
        raise ValueError(f'agg desconocido: {agg}')
    return pred_estudio, study_ids


# ============================================================================
# TABLA DE MÉTRICAS COMPLETAS
# ============================================================================

def compute_full_metrics(y_true, y_pred, n_boot=1000, seed=SEED):
    """
    Calcula todas las métricas para un par (y_true, y_pred):
    AUC + IC95%, AP + IC95%, Brier, ECE.

    Devuelve dict con las métricas.
    """
    auc, auc_lo, auc_hi = bootstrap_metric_ci(y_true, y_pred, roc_auc_score, n_boot=n_boot, seed=seed)
    ap, ap_lo, ap_hi = bootstrap_metric_ci(y_true, y_pred, average_precision_score, n_boot=n_boot, seed=seed)
    brier = brier_score_loss(y_true, y_pred)
    ece = ece_score(y_true, y_pred)
    return {
        'auc': auc, 'auc_lo': auc_lo, 'auc_hi': auc_hi,
        'ap':  ap,  'ap_lo':  ap_lo,  'ap_hi':  ap_hi,
        'brier': brier,
        'ece': ece,
    }
