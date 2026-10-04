"""Recompute the `score` column of existing score parts from `raw_output` with the current parse_score (no GPU needed).

    python -m reproduce.monitors.reparse_scores [--scores <dir>] [--dry-run]

Needed once for thinking columns scored before 2026-09-19: the old parser read a number out of a reasoning trace that
was cut off by max_tokens; such samples are now NaN (= unparsed). Prints how many scores change per config.
"""
import argparse, glob, os
import numpy as np, pandas as pd
from reproduce.monitors.score_matrix import parse_score
from gammacert.core import out_dir

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--scores", default=None); ap.add_argument("--dry-run", action="store_true"); a = ap.parse_args()
    for d in sorted(glob.glob(os.path.join(a.scores or out_dir("scores"), "*"))):
        think = "__think" in os.path.basename(d); n = ch = lost = 0
        for p in glob.glob(os.path.join(d, "*.parquet")):
            s = pd.read_parquet(p); new = s.raw_output.map(lambda t: parse_score(t, think)).astype(float)
            diff = ~((new == s.score) | (new.isna() & s.score.isna())); n += len(s); ch += int(diff.sum()); lost += int((new.isna() & s.score.notna()).sum())
            if diff.any() and not a.dry_run: s["score"] = new; s.to_parquet(p + ".tmp", index=False); os.replace(p + ".tmp", p)
        print(f"{os.path.basename(d)}: {n} samples, {ch} scores changed ({lost} became NaN)" + (" [dry run]" if a.dry_run else ""))
