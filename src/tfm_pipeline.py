"""
tfm_pipeline - Pipeline unificado para el TFM.

TFM - Master Universitario en Inteligencia Artificial - VIU 2025-2026
Victor Rodriguez Rodriguez

Componentes:
    - Builders de las 9 cabezas con sus grids de hiperparametros.
    - Funcion `holdout_grid_search`: hold-out interno 80/20 dentro del training de un fold.
    - Funcion `train_fold_unified`: scaler + grid search + reentrenamiento + prediccion.
    - Funcion `train_kfold_unified`: aplica train_fold_unified a los 5 folds y guarda predicciones.

Filosofia:
    - StandardScaler aplicado por fold (sin leakage), sobre TODOS los modelos uniformemente.
    - Grid search interno con hold-out estratificado 80/20 dentro del training
      (mas rapido que CV anidada).
    - Tras elegir los mejores hiperparametros, REENTRENA con el training completo del fold.
    - Misma semilla en todos los modelos (SEED=42).
    - Tratamiento del desbalance uniforme: class_weight='balanced' o pos_weight=n_neg/n_pos.
"""

import copy, time, warnings
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
import lightgbm as lgb
import xgboost as xgb
from itertools import product

# Silenciar warnings repetidos de sklearn (deprecaciones de la 1.8, no-convergencia de saga, etc.)
# No afectan a los resultados, sólo ensucian el output.
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=ConvergenceWarning)

SEED = 42

# ============================================================================
# GRIDS DE HIPERPARÁMETROS
# Mantenidos pequeños y razonables para que el tiempo total sea manejable.
# ============================================================================

GRIDS = {
    'logreg_l1':   [{'C': c} for c in [0.1, 1.0]],
    'logreg_l2':   [{'C': c} for c in [0.1, 1.0, 10.0]],
    'logreg_en':   [{'C': c, 'l1_ratio': r} for c, r in product([1.0], [0.3, 0.5, 0.7])],
    'rf':          [{'n_estimators': n, 'max_depth': d}
                    for n, d in product([300, 500], [None, 20])],
    'extratrees':  [{'n_estimators': n, 'max_depth': d}
                    for n, d in product([300, 500], [None, 20])],
    'histgb':      [{'max_leaf_nodes': l, 'learning_rate': lr}
                    for l, lr in product([31, 63], [0.05, 0.1])],
    'xgb':         [{'max_depth': d, 'learning_rate': lr}
                    for d, lr in product([3, 5], [0.05, 0.1])],
    'lgbm':        [{'num_leaves': l, 'learning_rate': lr}
                    for l, lr in product([31, 63], [0.05, 0.1])],
    'mlp':         [{'hidden': h, 'dropout': d}
                    for h, d in product([128, 256], [0.3])],
}

HEAD_NAMES = ['logreg_l1', 'logreg_l2', 'logreg_en',
              'rf', 'extratrees', 'histgb',
              'xgb', 'lgbm', 'mlp']

# ============================================================================
# MLP — definido aquí para evitar dependencias externas
# ============================================================================

class MLPHead(nn.Module):
    def __init__(self, in_dim, hidden=128, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden, 1),
        )
    def forward(self, x):
        return self.net(x)

# ============================================================================
# TRAINERS POR CABEZA
# Cada trainer recibe (X_tr, y_tr, X_val, y_val, X_test) escalados y SPW (scale_pos_weight).
# Devuelve (val_probs, test_probs, val_auc).
# Para modelos con grid search interno reciben además hparams.
# ============================================================================

def _spw(y_tr):
    n_pos = max(int(y_tr.sum()), 1)
    n_neg = max(int(len(y_tr) - y_tr.sum()), 1)
    return n_neg / n_pos


def train_logreg(X_tr, y_tr, X_val, y_val, X_test, hparams, penalty):
    """
    LogReg con penalty='l1', 'l2' o 'elasticnet'.
    Solver elegido por velocidad y robustez según la penalty:
      - L1: saga con tol=5e-3 (liblinear era inestable en alta dim con estos datos)
      - L2: lbfgs (rápido y robusto)
      - ElasticNet: saga con tol=1e-3 (única opción que soporta EN)
    """
    if penalty == 'l1':
        clf = LogisticRegression(
            penalty='l1', C=hparams['C'], solver='saga',
            max_iter=300, tol=5e-3,
            random_state=SEED, class_weight='balanced', n_jobs=-1)
    elif penalty == 'l2':
        clf = LogisticRegression(
            penalty='l2', C=hparams['C'], solver='lbfgs',
            max_iter=1000, random_state=SEED, class_weight='balanced',
            n_jobs=-1)
    elif penalty == 'elasticnet':
        clf = LogisticRegression(
            penalty='elasticnet', C=hparams['C'], l1_ratio=hparams['l1_ratio'],
            solver='saga', max_iter=500, tol=1e-3,
            random_state=SEED, class_weight='balanced', n_jobs=-1)
    else:
        raise ValueError(f'penalty desconocido: {penalty}')
    clf.fit(X_tr, y_tr)
    val_p  = clf.predict_proba(X_val)[:, 1]
    test_p = clf.predict_proba(X_test)[:, 1] if X_test is not None else None
    return val_p, test_p, roc_auc_score(y_val, val_p), clf


def train_rf(X_tr, y_tr, X_val, y_val, X_test, hparams):
    clf = RandomForestClassifier(
        n_estimators=hparams['n_estimators'], max_depth=hparams['max_depth'],
        class_weight='balanced', random_state=SEED, n_jobs=-1)
    clf.fit(X_tr, y_tr)
    val_p  = clf.predict_proba(X_val)[:, 1]
    test_p = clf.predict_proba(X_test)[:, 1] if X_test is not None else None
    return val_p, test_p, roc_auc_score(y_val, val_p), clf


def train_extratrees(X_tr, y_tr, X_val, y_val, X_test, hparams):
    clf = ExtraTreesClassifier(
        n_estimators=hparams['n_estimators'], max_depth=hparams['max_depth'],
        class_weight='balanced', random_state=SEED, n_jobs=-1)
    clf.fit(X_tr, y_tr)
    val_p  = clf.predict_proba(X_val)[:, 1]
    test_p = clf.predict_proba(X_test)[:, 1] if X_test is not None else None
    return val_p, test_p, roc_auc_score(y_val, val_p), clf


def train_histgb(X_tr, y_tr, X_val, y_val, X_test, hparams):
    # HistGradientBoosting acepta class_weight; usamos sample_weight por equivalencia limpia con boosting
    spw = _spw(y_tr)
    sw  = np.where(y_tr == 1, spw, 1.0)
    clf = HistGradientBoostingClassifier(
        max_leaf_nodes=hparams['max_leaf_nodes'],
        learning_rate=hparams['learning_rate'],
        max_iter=300, early_stopping=True, validation_fraction=0.1,
        n_iter_no_change=30, random_state=SEED)
    clf.fit(X_tr, y_tr, sample_weight=sw)
    val_p  = clf.predict_proba(X_val)[:, 1]
    test_p = clf.predict_proba(X_test)[:, 1] if X_test is not None else None
    return val_p, test_p, roc_auc_score(y_val, val_p), clf


def train_xgb(X_tr, y_tr, X_val, y_val, X_test, hparams):
    clf = xgb.XGBClassifier(
        max_depth=hparams['max_depth'], learning_rate=hparams['learning_rate'],
        n_estimators=300, scale_pos_weight=_spw(y_tr),
        eval_metric='auc', early_stopping_rounds=30,
        random_state=SEED, n_jobs=-1, verbosity=0)
    clf.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    val_p  = clf.predict_proba(X_val)[:, 1]
    test_p = clf.predict_proba(X_test)[:, 1] if X_test is not None else None
    return val_p, test_p, roc_auc_score(y_val, val_p), clf


def train_lgbm(X_tr, y_tr, X_val, y_val, X_test, hparams):
    clf = lgb.LGBMClassifier(
        num_leaves=hparams['num_leaves'], learning_rate=hparams['learning_rate'],
        n_estimators=300, scale_pos_weight=_spw(y_tr),
        random_state=SEED, n_jobs=-1, verbosity=-1)
    clf.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric='auc', callbacks=[lgb.early_stopping(30, verbose=False)])
    val_p  = clf.predict_proba(X_val)[:, 1]
    test_p = clf.predict_proba(X_test)[:, 1] if X_test is not None else None
    return val_p, test_p, roc_auc_score(y_val, val_p), clf


def train_mlp(X_tr, y_tr, X_val, y_val, X_test, hparams, device='cuda', max_epochs=120, batch_size=128, lr=1e-3, weight_decay=1e-4, patience=15):
    Xt = torch.from_numpy(X_tr).float()
    yt = torch.from_numpy(y_tr).float().unsqueeze(1)
    Xv = torch.from_numpy(X_val).float().to(device)
    Xte = torch.from_numpy(X_test).float().to(device) if X_test is not None else None

    model = MLPHead(in_dim=X_tr.shape[1], hidden=hparams['hidden'], dropout=hparams['dropout']).to(device)
    pos_weight = torch.tensor([_spw(y_tr)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    dl = DataLoader(TensorDataset(Xt, yt), batch_size=batch_size, shuffle=True)

    best_auc, best_state, no_improve = -1.0, None, 0
    for epoch in range(max_epochs):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            vp = torch.sigmoid(model(Xv)).cpu().numpy().ravel()
        try: val_auc = roc_auc_score(y_val, vp)
        except ValueError: val_auc = float('nan')
        scheduler.step(val_auc)
        if val_auc > best_auc:
            best_auc = val_auc; best_state = copy.deepcopy(model.state_dict()); no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience: break

    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        val_probs  = torch.sigmoid(model(Xv)).cpu().numpy().ravel()
        test_probs = torch.sigmoid(model(Xte)).cpu().numpy().ravel() if Xte is not None else None
    return val_probs, test_probs, best_auc, model


HEAD_TRAINERS = {
    'logreg_l1':  lambda *a, **k: train_logreg(*a, **k, penalty='l1'),
    'logreg_l2':  lambda *a, **k: train_logreg(*a, **k, penalty='l2'),
    'logreg_en':  lambda *a, **k: train_logreg(*a, **k, penalty='elasticnet'),
    'rf':         train_rf,
    'extratrees': train_extratrees,
    'histgb':     train_histgb,
    'xgb':        train_xgb,
    'lgbm':       train_lgbm,
    'mlp':        train_mlp,
}

# ============================================================================
# HOLD-OUT INTERNO + GRID SEARCH
# Para cada combinación del grid: entrena en sub-train, evalúa en sub-val,
# devuelve los mejores hiperparámetros según AUC en sub-val.
# ============================================================================

def holdout_grid_search(head_name, X_tr, y_tr, mlp_device='cuda'):
    """
    Dentro del training de un fold:
    1. Split 80/20 estratificado en sub-train, sub-val
    2. Probar todas las combinaciones del grid de la cabeza
    3. Elegir la mejor por AUC en sub-val
    Devuelve los mejores hparams (dict).
    """
    grid = GRIDS[head_name]
    trainer = HEAD_TRAINERS[head_name]
    if len(grid) == 1:
        return grid[0], None  # nada que buscar

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    sub_tr_idx, sub_val_idx = next(sss.split(X_tr, y_tr))
    X_sub_tr, y_sub_tr = X_tr[sub_tr_idx], y_tr[sub_tr_idx]
    X_sub_val, y_sub_val = X_tr[sub_val_idx], y_tr[sub_val_idx]

    best_auc, best_hp = -1.0, None
    aucs_log = []
    for hp in grid:
        kwargs = {'hparams': hp}
        if head_name == 'mlp':
            kwargs['device'] = mlp_device
        _, _, auc, _ = trainer(X_sub_tr, y_sub_tr, X_sub_val, y_sub_val, None, **kwargs)
        aucs_log.append((hp, auc))
        if auc > best_auc:
            best_auc, best_hp = auc, hp
    return best_hp, aucs_log


# ============================================================================
# ENTRENAMIENTO DE UN FOLD CON PROTOCOLO UNIFICADO
# ============================================================================

def train_fold_unified(head_name, X_tr, y_tr, X_val, y_val, X_test, mlp_device='cuda'):
    """
    Pipeline completo por fold:
    1. StandardScaler ajustado en X_tr
    2. Grid search interno sobre X_tr (hold-out 80/20)
    3. Reentrenamiento con X_tr completo usando los mejores hparams
    4. Predicción sobre X_val y X_test
    Devuelve (val_probs, test_probs, val_auc, best_hp, search_log).
    """
    scaler = StandardScaler()
    X_tr_s   = scaler.fit_transform(X_tr).astype(np.float32)
    X_val_s  = scaler.transform(X_val).astype(np.float32)
    X_test_s = scaler.transform(X_test).astype(np.float32) if X_test is not None else None

    best_hp, search_log = holdout_grid_search(head_name, X_tr_s, y_tr, mlp_device=mlp_device)

    trainer = HEAD_TRAINERS[head_name]
    kwargs = {'hparams': best_hp}
    if head_name == 'mlp':
        kwargs['device'] = mlp_device
    val_p, test_p, val_auc, _ = trainer(X_tr_s, y_tr, X_val_s, y_val, X_test_s, **kwargs)
    return val_p, test_p, val_auc, best_hp, search_log


# ============================================================================
# KFOLD UNIFICADO
# ============================================================================

def train_kfold_unified(head_name, X, y, is_train, is_test, groups=None, n_splits=5, mlp_device='cuda', verbose=True):
    """
    Aplica train_fold_unified a los 5 folds.
    Si groups se pasa, usa StratifiedGroupKFold (necesario a nivel mama).
    Devuelve dict con oof_preds, test_preds, métricas y log de hparams elegidos.
    """
    X_tr_pool = X[is_train]; y_tr_pool = y[is_train]
    X_te_pool = X[is_test];  y_te_pool = y[is_test]

    if groups is None:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
        split_iter = splitter.split(X_tr_pool, y_tr_pool)
    else:
        g_tr = groups[is_train]
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
        split_iter = splitter.split(X_tr_pool, y_tr_pool, groups=g_tr)

    oof_preds  = np.zeros(len(y_tr_pool), dtype=np.float32)
    test_preds = np.zeros((n_splits, len(y_te_pool)), dtype=np.float32)
    fold_aucs  = []
    fold_hps   = []

    for fold, (tr_idx, val_idx) in enumerate(split_iter, start=1):
        t0 = time.time()
        val_p, test_p, val_auc, best_hp, _ = train_fold_unified(
            head_name,
            X_tr_pool[tr_idx], y_tr_pool[tr_idx],
            X_tr_pool[val_idx], y_tr_pool[val_idx],
            X_te_pool, mlp_device=mlp_device,
        )
        oof_preds[val_idx] = val_p
        test_preds[fold-1] = test_p
        fold_aucs.append(val_auc)
        fold_hps.append(best_hp)
        if verbose:
            print(f'fold {fold}: val_AUC={val_auc:.4f} hp={best_hp} ({time.time()-t0:.1f}s)')

    oof_auc  = roc_auc_score(y_tr_pool, oof_preds)
    test_ens = test_preds.mean(axis=0)
    test_auc = roc_auc_score(y_te_pool, test_ens)

    if verbose:
        print(f'-> fold AUCs: {[f"{a:.4f}" for a in fold_aucs]}')
        print(f'-> mean fold AUC: {np.mean(fold_aucs):.4f}  (std {np.std(fold_aucs):.4f})')
        print(f'-> OOF AUC: {oof_auc:.4f}     Test AUC: {test_auc:.4f}')

    return {
        'oof_preds': oof_preds, 'test_preds': test_ens,
        'oof_auc': oof_auc, 'test_auc': test_auc,
        'mean_fold_auc': float(np.mean(fold_aucs)),
        'std_fold_auc':  float(np.std(fold_aucs)),
        'fold_aucs': fold_aucs, 'fold_hps': fold_hps,
    }
