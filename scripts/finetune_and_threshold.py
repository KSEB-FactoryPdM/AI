import argparse, yaml, joblib, numpy as np, pandas as pd
from pathlib import Path
from ae_train import fit_ae, recon_error

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train.yaml")
    ap.add_argument("--equipment-id", required=True)
    ap.add_argument("--power", required=True, help="예: 3.7kW, 5.5kW, 11kW ...")
    ap.add_argument("--allow-single-modal", action="store_true", help="하나의 모달만 있어도 진행")
    ap.add_argument("--feats-current", default="artifacts/current_feats.parquet")
    ap.add_argument("--feats-vibration", default="artifacts/vibration_feats.parquet")
    ap.add_argument("--out-dir", default="models")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    q = cfg["ae_threshold_quantile"]

    dfc = pd.read_parquet(args.feats_current)
    dfv = pd.read_parquet(args.feats_vibration)
    dfc = dfc[(dfc["equipment_id"] == args.equipment_id) & (dfc["power"] == args.power)]
    dfv = dfv[(dfv["equipment_id"] == args.equipment_id) & (dfv["power"] == args.power)]

    normal_label = cfg["label_map"].get("정상", "normal")
    dfc_n = dfc[dfc["label"] == normal_label]
    dfv_n = dfv[dfv["label"] == normal_label]

    has_cur = not dfc_n.empty
    has_vib = not dfv_n.empty
    if not (has_cur or has_vib):
        raise SystemExit(f"[ERR] no NORMAL rows for equipment={args.equipment_id}, power={args.power}")
    if (not has_cur or not has_vib) and (not args.allow_single_modal):
        raise SystemExit(f"[ERR] missing modality (current={has_cur}, vibration={has_vib}). Use --allow-single-modal.")

    out = Path(args.out_dir) / args.equipment_id / args.power / cfg["model_version"]
    out.mkdir(parents=True, exist_ok=True)

    modalities = []
    from sklearn.preprocessing import StandardScaler
    import torch, json, hashlib, yaml as _y

    # current
    if has_cur:
        cur_cols = [c for c in dfc.columns if c.startswith("cur_")]
        Xc = dfc_n[cur_cols].astype(float).values
        sc_c = StandardScaler().fit(Xc)
        Xc_s = sc_c.transform(Xc)
        from ae_train import fit_ae, recon_error
        ae_c = fit_ae(Xc_s, cfg["ae_current"], epochs_key="finetune_epochs")
        th_c = float(np.quantile(recon_error(ae_c, Xc_s), q))
        joblib.dump(sc_c, out/"standard_scaler_current.joblib")
        torch.save(ae_c.state_dict(), out/"ae_current.pt")
        modalities.append("current")
    else:
        th_c = None; cur_cols=[]

    # vibration
    if has_vib:
        vib_cols = [c for c in dfv.columns if c.startswith("vib_")]
        Xv = dfv_n[vib_cols].astype(float).values
        sc_v = StandardScaler().fit(Xv)
        Xv_s = sc_v.transform(Xv)
        ae_v = fit_ae(Xv_s, cfg["ae_vibration"], epochs_key="finetune_epochs")
        th_v = float(np.quantile(recon_error(ae_v, Xv_s), q))
        joblib.dump(sc_v, out/"standard_scaler_vibration.joblib")
        torch.save(ae_v.state_dict(), out/"ae_vibration.pt")
        modalities.append("vibration")
    else:
        th_v = None; vib_cols=[]

    # feature_spec / metadata
    spec = {
        "features": [*cur_cols, *vib_cols],
        "modal_map": {m: (cur_cols if m=="current" else vib_cols) for m in modalities},
        "scaling": "standard_separate",
    }
    (out/"feature_spec.yaml").write_text(_y.safe_dump(spec, sort_keys=False, allow_unicode=True), encoding="utf-8")

    def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    files = [p for p in out.glob("*") if p.is_file()]
    meta = {
        "equipment_id": args.equipment_id,
        "power": args.power,
        "model_version": cfg["model_version"],
        "modalities": modalities,
        "case": "A",
        "faults": [],
        "th_ae": { "current": th_c, "vibration": th_v },
        "files_sha256": {p.name: sha(p) for p in files}
    }
    (out/"metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] AE/thresholds ({'+'.join(modalities)}) -> {out}")

if __name__ == "__main__":
    main()
