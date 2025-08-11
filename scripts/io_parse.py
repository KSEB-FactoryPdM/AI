from pathlib import Path
import csv, io, re
import numpy as np

NUMERIC_RE = re.compile(r'^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$')

def _only_numeric_tokens(tokens):
    tokens = [t.strip() for t in tokens if t.strip() != ""]
    if not tokens:
        return False
    return all(NUMERIC_RE.match(t) is not None for t in tokens)

def _split_tokens(s):
    # 콤마/스페이스 혼합 대응, 빈 토큰 제거
    return [t for t in re.split(r'[,\s]+', s.strip()) if t != ""]

def parse_header(lines):
    meta = {}
    first_data = None
    for ln in lines:
        s = ln.strip()
        if not s: 
            continue
        if s.startswith("Date,"):          meta["date"] = s.split(",",1)[1]
        elif s.startswith("Filename,"):    meta["filename"] = s.split(",",1)[1]
        elif s.startswith("Data Label,"):  meta["data_label"] = s.split(",",1)[1]
        elif s.startswith("Label No,"):    meta["label_no"] = s.split(",",1)[1]
        elif s.startswith("Motor Spec,"):  meta["motor_spec"] = s.split(",",1)[1]
        elif s.startswith("Period,"):      meta["period"] = s.split(",",1)[1]
        elif s.startswith("Sample Rate,"):
            try: meta["sample_rate"] = float(s.split(",",1)[1])
            except: pass
        elif s.startswith("RMS,"):         meta["rms"] = s.split(",",1)[1]
        elif s.startswith("Data Length,"):
            try: meta["data_len"] = int(s.split(",",1)[1])
            except: pass
        else:
            toks = _split_tokens(s)
            if _only_numeric_tokens(toks):
                first_data = s
                break
            else:
                continue
    return meta, first_data

def read_signal_block(path: Path, first_data_line: str, n_cols: int):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        raw = f.read().splitlines()

    # n_cols 또는 n_cols+1(선두 시간열) 모두 허용
    def fits_row(toks):
        if not _only_numeric_tokens(toks):
            return False
        L = len(toks)
        return (
            L >= n_cols and (
                L == n_cols or
                L == n_cols + 1 or
                L % (n_cols + 1) == 0 or
                L % n_cols == 0
            )
        )

    start_idx = None
    seed_toks = None
    if first_data_line:
        t = _split_tokens(first_data_line)
        if fits_row(t):
            start_idx = -1
            seed_toks = t

    if start_idx is None:
        for i, ln in enumerate(raw):
            t = _split_tokens(ln)
            if fits_row(t):
                start_idx = i
                seed_toks = t
                break

    if start_idx is None:
        raise ValueError(f"Data start not found in file: {path}")

    rows = []

    def push_tokens(toks):
        L = len(toks)
        if L == n_cols:
            rows.append([float(x) for x in toks])
        elif L == n_cols + 1:
            rows.append([float(x) for x in toks[1:]])
        elif L > n_cols + 1 and L % (n_cols + 1) == 0:
            for k in range(0, L, n_cols + 1):
                rows.append([float(x) for x in toks[k+1:k+1+n_cols]])
        elif L > n_cols and L % n_cols == 0:
            for k in range(0, L, n_cols):
                rows.append([float(x) for x in toks[k:k+n_cols]])
        # 그 외는 무시

    if start_idx == -1:
        push_tokens(seed_toks)
        data_iter = raw
        start_j = 1
    else:
        data_iter = raw[start_idx:]
        push_tokens(seed_toks)
        start_j = 1

    for ln in data_iter[start_j:]:
        toks = _split_tokens(ln)
        if not _only_numeric_tokens(toks):
            continue
        push_tokens(toks)

    return np.asarray(rows, dtype=float)