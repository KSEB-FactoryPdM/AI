import json
from pathlib import Path
import yaml
import pandas as pd

def main():
    cfg = yaml.safe_load(open("configs/train.yaml","r",encoding="utf-8"))
    root = Path(cfg["out_dir"])
    rows = []
    for eid_dir in root.iterdir():
        if not eid_dir.is_dir(): continue
        for power_dir in eid_dir.iterdir():
            if not power_dir.is_dir(): continue
            for ver_dir in power_dir.iterdir():
                if not ver_dir.is_dir(): continue
                meta_p = ver_dir / "metadata.json"
                if not meta_p.exists(): continue
                meta = json.loads(meta_p.read_text(encoding="utf-8"))
                metrics = meta.get("metrics", {})
                row = {
                    "equipment_id": meta.get("equipment_id", eid_dir.name),
                    "power": meta.get("power", power_dir.name),
                    "model_version": meta.get("model_version", ver_dir.name),
                    "modalities": "+".join(meta.get("modalities", [])),
                    "case": meta.get("case", ""),
                    "accuracy": metrics.get("accuracy"),
                    "f1_macro": metrics.get("f1_macro"),
                }
                rep = metrics.get("classification_report", {})
                if isinstance(rep, dict):
                    for cls, vals in rep.items():
                        if not isinstance(vals, dict): continue
                        row[f"prec_{cls}"] = vals.get("precision")
                        row[f"rec_{cls}"]  = vals.get("recall")
                        row[f"f1_{cls}"]   = vals.get("f1-score")
                rows.append(row)
    df = pd.DataFrame(rows)
    out = Path("artifacts/metrics_summary.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"[OK] saved metrics summary -> {out}")

if __name__=="__main__":
    main()
