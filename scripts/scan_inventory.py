import argparse, json
from pathlib import Path
from collections import defaultdict
import yaml

def scan_modal(root):
    # root/<power>/<equipment>/<label>/*.csv
    stats = defaultdict(lambda: {"powers": set(), "labels_by_power": defaultdict(set)})
    root = Path(root)
    for power_dir in root.iterdir():
        if not power_dir.is_dir(): continue
        power = power_dir.name
        for eq_dir in power_dir.iterdir():
            if not eq_dir.is_dir(): continue
            eid = eq_dir.name
            stats[eid]["powers"].add(power)
            for lab_dir in eq_dir.iterdir():
                if not lab_dir.is_dir(): continue
                stats[eid]["labels_by_power"][power].add(lab_dir.name)
    return stats

def fit_ae(X, cfg, epochs_key="epochs"):
    import numpy as np
    import torch
    from torch.utils.data import TensorDataset, DataLoader
    import torch.optim as optim
    import torch.nn as nn

    # 안전 캐스팅
    _to_float = lambda v: float(v)
    _to_int = lambda v: int(float(v))

    X = np.asarray(X, dtype=np.float32)

    hidden = [_to_int(h) for h in cfg["hidden"]]
    latent_dim = _to_int(cfg["latent_dim"])
    batch_size = _to_int(cfg["batch_size"])
    lr = _to_float(cfg["lr"])
    epochs = _to_int(cfg.get(epochs_key, cfg["epochs"]))

    ds = TensorDataset(torch.from_numpy(X))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    ae = AE(X.shape[1], hidden, latent_dim)
    opt = optim.Adam(ae.parameters(), lr=lr)
    crit = nn.MSELoss()

    ae.train()
    for _ in range(epochs):
        for (xb,) in dl:
            xr, _ = ae(xb)
            loss = crit(xr, xb)
            opt.zero_grad()
            loss.backward()
            opt.step()
    ae.eval()
    return ae

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train.yaml")
    ap.add_argument("--out", default="artifacts/inventory.json")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config,"r",encoding="utf-8"))
    cur = scan_modal(cfg["current_root"])
    vib = scan_modal(cfg["vibration_root"])

    normals = set(cfg["status_normal_keywords"])
    eids = sorted(set(cur.keys()) | set(vib.keys()))
    inv = {}

    for eid in eids:
        powers = sorted(list(cur.get(eid,{}).get("powers",set()) | vib.get(eid,{}).get("powers",set())))
        for power in powers:
            labels = set()
            labels |= set(cur.get(eid,{}).get("labels_by_power",{}).get(power,set()))
            labels |= set(vib.get(eid,{}).get("labels_by_power",{}).get(power,set()))
            faults = sorted([l for l in labels if l not in normals])
            if   len(faults)==0: case="A"
            elif len(faults)==1: case="B"
            else:                case="C"
            key = f"{eid}::{power}"
            inv[key] = {"equipment_id": eid, "power": power, "case": case, "faults": faults}

    Path("artifacts").mkdir(parents=True, exist_ok=True)
    json.dump(inv, open(args.out,"w",encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"[OK] inventory -> {args.out} (entries={len(inv)})")

if __name__=="__main__":
    main()
