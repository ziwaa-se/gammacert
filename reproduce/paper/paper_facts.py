"""Small facts quoted in the manuscript, computed from the merged score matrix."""
import glob, os, pandas as pd, numpy as np
from gammacert.core import out_dir
R = out_dir("results"); M = pd.read_parquet(f"{R}/fpr05_surface/M_long.parquet"); M = M[M.p_detect.notna()]
b = M[(M.dataset == "basharena") & M.is_attack].pivot_table(index="row_id", columns="config_id", values="p_detect"); print("monitor configs:", b.shape[1], "| bash attacks:", len(b))
print("bash attacks missed by ALL configs at 5% surface:", int((b.sum(1) == 0).sum()))
for tag in ("fpr01_surface", "fpr10_surface"):
    m = pd.read_parquet(f"{R}/{tag}/M_long.parquet"); m = m[m.p_detect.notna() & (m.dataset == "basharena") & m.is_attack].pivot_table(index="row_id", columns="config_id", values="p_detect"); print(tag, "missed by all:", int((m.sum(1) == 0).sum()))
T = pd.read_parquet(os.path.join(out_dir("data"), "transcripts.parquet"))[["row_id", "is_attack"]]
for d in sorted(glob.glob(os.path.join(out_dir("scores"), "*"))):
    S = pd.concat(pd.read_parquet(p, columns=["row_id", "score", "truncated"]) for p in glob.glob(d + "/*.parquet")).merge(T, on="row_id")
    r = S.groupby("row_id").agg(s=("score", "mean"), a=("is_attack", "first"))
    print(f"{os.path.basename(d):58s} NaN samples {S.score.isna().mean():6.2%} | rows without score: attack {r[r.a].s.isna().mean():6.2%} benign {r[~r.a].s.isna().mean():6.2%} | truncated prompts {S.truncated.mean():6.2%}")
main = M[M.config_id == "Qwen3-32B__monitor_default__full__think__n2"]
from sklearn.metrics import roc_auc_score
for ds, g in main.groupby("dataset"): print("main monitor AUROC", ds, round(roc_auc_score(g.is_attack, g.score.fillna(-1)), 3))
