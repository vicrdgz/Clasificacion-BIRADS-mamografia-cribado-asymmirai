"""
tfm_models — cabezas de clasificación del TFM.
Uso:
    from tfm_models import Head, build_mlp, build_gbm
"""
import torch.nn as nn
import lightgbm as lgb

class Head(nn.Module):
    def __init__(self, in_dim, hidden=None, dropout=0.3):
        super().__init__()
        if hidden is None:
            hidden = max(64, in_dim // 16)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden, 1),
        )
        self.in_dim, self.hidden = in_dim, hidden
    def forward(self, x):
        return self.net(x)

def build_mlp(in_dim):
    return Head(in_dim=in_dim, dropout=0.3)

def build_gbm(scale_pos_weight=1.0, seed=42):
    return lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=31,
        min_child_samples=20, reg_alpha=0.1, reg_lambda=0.1,
        scale_pos_weight=scale_pos_weight, random_state=seed,
        n_jobs=-1, verbosity=-1,
    )
