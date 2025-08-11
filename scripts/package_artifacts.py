import argparse, json, hashlib
from pathlib import Path
import yaml

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train.yaml")
    ap.add_argument("--equipment-id", required=True)
    ap.add_argument("--power", required=True)
    ap.add_argument("--out-dir", default="models")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    out = Path(args.out_dir) / args.equipment_id / args.power / cfg["model_version"]
    if not out.exists():
        raise SystemExit(f"[ERR] output folder not found: {out}")

    files = [p for p in out.glob("*") if p.is_file()]
    meta = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
    meta["power"] = args.power
    meta["files_sha256"] = {p.name: sha(p) for p in files}
    (out / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] packaged (sha256 updated) -> {out}")

if __name__ == "__main__":
    main()
