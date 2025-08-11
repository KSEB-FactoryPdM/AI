# scripts/run_all_equipments.py
import json, subprocess, time
from pathlib import Path

CONFIG = "configs/train.yaml"
INV_PATH = "artifacts/inventory.json"

def run(cmd):
    print("[RUN]", " ".join(cmd)); t0=time.time()
    subprocess.run(cmd, check=True)
    print(f"[OK] {time.time()-t0:.1f}s")

def main():
    inv = json.loads(Path(INV_PATH).read_text(encoding="utf-8"))
    keys = sorted(inv.keys())  # "EID::POWER"
    for key in keys:
        info = inv[key]
        eid, power, case = info["equipment_id"], info["power"], info["case"]
        print(f"\n===== {eid} @ {power} (case {case}) =====")
                # 1) AE 미세튜닝 + 임계값
        run(["python","scripts/finetune_and_threshold.py","--config",CONFIG,
             "--equipment-id",eid,"--power",power,"--allow-single-modal"])
        # 2) 분류기 (B/C만)
        if case != "A":
            run(["python","scripts/train_by_equipment.py","--config",CONFIG,
                 "--equipment-id",eid,"--power",power,"--inventory",INV_PATH,"--allow-single-modal"])
        # 3) 패키징(체크섬)
        run(["python","scripts/package_artifacts.py","--config",CONFIG,
             "--equipment-id",eid,"--power",power])

if __name__=="__main__":
    main()
