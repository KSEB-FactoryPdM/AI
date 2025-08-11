import argparse, yaml, joblib, numpy as np, pandas as pd
from pathlib import Path
from ae_train import AE, encode
from xgb_train import train_xgb, save_json
import torch, json
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import xgboost as xgb

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train.yaml")
    ap.add_argument("--equipment-id", required=True)
    ap.add_argument("--power", required=True)
    ap.add_argument("--allow-single-modal", action="store_true")
    ap.add_argument("--inventory", default="artifacts/inventory.json")
    ap.add_argument("--feats-current", default="artifacts/current_feats.parquet")
    ap.add_argument("--feats-vibration", default="artifacts/vibration_feats.parquet")
    ap.add_argument("--out-dir", default="models")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    inv = json.loads(Path(args.inventory).read_text(encoding="utf-8"))
    inv_key = f"{args.equipment_id}::{args.power}"
    if inv_key not in inv:
        raise SystemExit(f"[ERR] inventory entry not found: {inv_key}")
    case = inv[inv_key]["case"]

    out = Path(args.out_dir) / args.equipment_id / args.power / cfg["model_version"]
    if not out.exists():
        raise SystemExit(f"[ERR] output folder not found (run finetune first): {out}")

    # metadata로 사용 모달 확인
    meta = json.loads((out/"metadata.json").read_text(encoding="utf-8"))
    modalities = meta.get("modalities", ["current","vibration"])
    normal_label = cfg["label_map"].get("정상", "normal")

    # 피처 로드 + 필터
    dfc = pd.read_parquet(args.feats_current)
    dfv = pd.read_parquet(args.feats_vibration)
    dfc = dfc[(dfc["equipment_id"] == args.equipment_id) & (dfc["power"] == args.power)]
    dfv = dfv[(dfv["equipment_id"] == args.equipment_id) & (dfv["power"] == args.power)]

    use_cur = "current" in modalities and not dfc.empty
    use_vib = "vibration" in modalities and not dfv.empty
    if not (use_cur or use_vib):
        raise SystemExit(f"[ERR] no rows for selected modalities at {args.equipment_id}@{args.power}")
    if (not use_cur or not use_vib) and (not args.allow_single_modal):
        raise SystemExit(f"[ERR] single-modal at train but not allowed. Pass --allow-single-modal.")

    # 케이스 A면 분류기 없음
    if case == "A":
        print("[INFO] Case A: gate only (no classifier).")
        meta["case"] = "A"; meta["faults"] = []
        (out / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        return

    # 컬럼
    cur_cols = [c for c in dfc.columns if c.startswith("cur_")]
    vib_cols = [c for c in dfv.columns if c.startswith("vib_")]

    # 라벨 인코딩
    map_label = cfg["label_map"]
    enc_label = lambda x: map_label.get(x, x)
    if use_cur: dfc["label"] = dfc["label"].map(enc_label)
    if use_vib: dfv["label"] = dfv["label"].map(enc_label)

    # 샘플 수 맞추기
    if use_cur and use_vib:
        n = min(len(dfc), len(dfv))
        if n < 5:
            raise SystemExit(f"[ERR] too few samples for classifier: n={n}")
        dfc = dfc.sample(n=n, random_state=42).reset_index(drop=True)
        dfv = dfv.sample(n=n, random_state=42).reset_index(drop=True)
    elif use_cur:
        n = len(dfc)
    else:
        n = len(dfv)
    if n < 5:
        raise SystemExit(f"[ERR] too few samples for classifier: n={n}")

    # 잠재 만들기
    Z_parts = []
    if use_cur:
        sc_c = joblib.load(out/"standard_scaler_current.joblib")
        ae_c = AE(len(cur_cols), cfg["ae_current"]["hidden"], cfg["ae_current"]["latent_dim"])
        ae_c.load_state_dict(torch.load(out/"ae_current.pt", map_location="cpu")); ae_c.eval()
        Xc = sc_c.transform(dfc[cur_cols].astype(float).values)
        Zc = encode(ae_c, Xc); Z_parts.append(Zc)

    if use_vib:
        sc_v = joblib.load(out/"standard_scaler_vibration.joblib")
        ae_v = AE(len(vib_cols), cfg["ae_vibration"]["hidden"], cfg["ae_vibration"]["latent_dim"])
        ae_v.load_state_dict(torch.load(out/"ae_vibration.pt", map_location="cpu")); ae_v.eval()
        Xv = sc_v.transform(dfv[vib_cols].astype(float).values)
        Zv = encode(ae_v, Xv); Z_parts.append(Zv)

    Z = Z_parts[0] if len(Z_parts)==1 else np.hstack(Z_parts)

    # y
    y_str = (dfc if use_cur else dfv)["label"].values
    classes = sorted(list(set(y_str)))
    if normal_label in classes:
        classes.remove(normal_label); classes = [normal_label] + classes
    class_to_id = {c: i for i, c in enumerate(classes)}

    y = np.array([class_to_id[s] for s in y_str], dtype=int)
    n_classes = len(set(y))
    if n_classes == 1:
        print("[WARN] only one class present; skip classifier.")
        return

    # split
    X_tr, X_te, y_tr, y_te = train_test_split(Z, y, test_size=0.2, random_state=42, stratify=y)

    # 학습/저장
    booster = train_xgb(X_tr, y_tr, cfg["xgb"], n_classes=n_classes)
    save_json(booster, str(out / "xgb.json"))

    # 평가
    proba = np.asarray(booster.predict(xgb.DMatrix(X_te)))
    if proba.ndim == 1:  # binary
        y_hat = (proba >= 0.5).astype(int)
    else:
        y_hat = proba.argmax(axis=1)

    from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
    acc = float(accuracy_score(y_te, y_hat))
    f1_macro = float(f1_score(y_te, y_hat, average="macro"))
    class_map = {i: c for c, i in class_to_id.items()}
    target_names = [class_map[i] for i in range(len(class_map))] if len(class_map) else None
    report_dict = classification_report(y_te, y_hat, target_names=target_names,
                                        zero_division=0, output_dict=True)
    cm = confusion_matrix(y_te, y_hat).tolist()

    # 메타데이터 갱신
    meta["power"] = args.power
    meta["case"] = "B" if n_classes == 2 else "C"
    meta["faults"] = [c for c in classes if c != normal_label]
    meta["class_map"] = {int(i): c for i, c in class_map.items()}
    meta.setdefault("metrics", {})
    meta["metrics"].update({
        "accuracy": acc,
        "f1_macro": f1_macro,
        "classification_report": report_dict,
        "confusion_matrix": cm
    })
    (out / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[METRICS] acc={acc:.4f}  f1_macro={f1_macro:.4f}")
    print(f"[OK] classifier saved -> {out/'xgb.json'}")

if __name__ == "__main__":
    main()
