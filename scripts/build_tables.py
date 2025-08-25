from pathlib import Path
import pandas as pd
import yaml
from io_parse import parse_header, read_signal_block
from features import current_feats, vibration_feats
import numpy as np
from scipy.stats import skew, kurtosis
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import Counter

def _iter_files(root):
    for p in Path(root).rglob("*.csv"):
        yield p

def label_from(cfg, raw, path):
    if raw:
        return cfg["label_map"].get(raw, raw)
    for k in cfg["status_normal_keywords"]:
        if k in str(path):
            return cfg["label_map"].get(k, k)
    lab = path.parent.name
    return cfg["label_map"].get(lab, lab)

def build_modal_table(root, modal, cfg):
    rows=[]
    for p in _iter_files(root):
        lines=Path(p).read_text(encoding="utf-8",errors="ignore").splitlines()
        meta, first= parse_header(lines)
        fs = meta.get("sample_rate", 4000.0 if modal=="vibration" else 2000.0)
        if modal=="vibration":
            sig = read_signal_block(p, first, n_cols=1)
            feats = vibration_feats(sig, fs, cfg["bands_hz"], cfg.get("use_envelope",True))
        else:
            sig = read_signal_block(p, first, n_cols=3)
            feats = current_feats(sig, fs, cfg["bands_hz"], cfg.get("use_envelope",True))
        # 경로 규칙: <root>/<power>/<equipment>/<label>/*.csv
        feats["equipment_id"] = p.parent.parent.name
        feats["power"] = p.parent.parent.parent.name
        feats["label"] = label_from(cfg, meta.get("data_label",""), p)
        rows.append(feats)
    return pd.DataFrame(rows)

def extract_features(values):
    if isinstance(values, np.ndarray) and values.ndim == 1:
        values = values.reshape(-1, 1)
    feats = []
    for i in range(values.shape[1]):
        v = values[:, i]
        feats.extend([
            float(np.mean(v)),
            float(np.std(v, ddof=1)) if v.size > 1 else 0.0,
            float(np.sqrt(np.mean(v ** 2))),
            float(np.max(np.abs(v))),
            float(skew(v)) if v.size > 2 else 0.0,
            float(kurtosis(v)) if v.size > 3 else 0.0,
        ])
    return np.array(feats, dtype=float)

def anomaly_detection_on_table(table, contamination=0.02, threshold_method="zero", test_size=0.1, random_state=0):
    # table: DataFrame with 'label' and feature columns
    normal_label = '정상'
    X_normal = table[table['label'] == normal_label].drop('label', axis=1).values
    if X_normal.shape[0] == 0:
        return None
    iso = IsolationForest(contamination=contamination, random_state=random_state)
    iso.fit(X_normal)
    scores = iso.decision_function(table.drop('label', axis=1).values)
    if threshold_method == "quantile":
        thr = np.quantile(scores, 1.0 - contamination)
    else:
        thr = 0.0
    pseudo = np.where(scores >= thr, '정상', '비정상_의사')
    cnt_pseudo = Counter(pseudo)
    y_pseudo = (pseudo == '비정상_의사').astype(int)
    feats_valid = table.drop('label', axis=1).values
    class_counts = Counter(y_pseudo)
    if len(class_counts) < 2 or min(class_counts.values()) < 2:
        clf_pseudo = RandomForestClassifier(n_estimators=300, random_state=random_state, class_weight='balanced')
        clf_pseudo.fit(feats_valid, y_pseudo)
        acc_pseudo = None
    else:
        X_tr, X_te, y_tr, y_te = train_test_split(
            feats_valid, y_pseudo, test_size=test_size,
            random_state=random_state, stratify=y_pseudo
        )
        clf_pseudo = RandomForestClassifier(n_estimators=300, random_state=random_state, class_weight='balanced')
        clf_pseudo.fit(X_tr, y_tr)
        acc_pseudo = accuracy_score(y_te, clf_pseudo.predict(X_te))
    supervised_accuracy_real = None
    true_valid = table['label'].values
    if len(set(true_valid)) > 1:
        cnt_true = Counter(true_valid)
        if min(cnt_true.values()) < 2:
            clf_real = RandomForestClassifier(n_estimators=200, random_state=random_state)
            clf_real.fit(feats_valid, true_valid)
        else:
            X_tr2, X_te2, y_tr2, y_te2 = train_test_split(
                feats_valid, true_valid, test_size=test_size,
                random_state=random_state, stratify=true_valid
            )
            clf_real = RandomForestClassifier(n_estimators=200, random_state=random_state)
            clf_real.fit(X_tr2, y_tr2)
            supervised_accuracy_real = accuracy_score(y_te2, clf_real.predict(X_te2))
    return {
        'unsupervised_samples': int(X_normal.shape[0]),
        'pseudo_counts': dict(cnt_pseudo),
        'pseudo_split_acc': acc_pseudo,
        'supervised_accuracy_true_labels': supervised_accuracy_real
    }

if __name__=="__main__":
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train.yaml")
    ap.add_argument("--out-prefix", default="artifacts")
    args=ap.parse_args()
    cfg=yaml.safe_load(open(args.config,"r",encoding="utf-8"))
    Path(args.out_prefix).mkdir(parents=True, exist_ok=True)
    df_cur = build_modal_table(cfg["current_root"], "current", cfg)
    df_vib = build_modal_table(cfg["vibration_root"], "vibration", cfg)
    df_cur.to_parquet(Path(args.out_prefix)/"current_feats.parquet")
    df_vib.to_parquet(Path(args.out_prefix)/"vibration_feats.parquet")
    print("[OK] feature tables saved.")
