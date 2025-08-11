from pathlib import Path
import pandas as pd
import yaml
from io_parse import parse_header, read_signal_block
from features import current_feats, vibration_feats

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
