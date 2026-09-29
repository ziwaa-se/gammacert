"""One column of the score matrix M: a monitor configuration (model x prompt x view) scores every transcript.

    python -m reproduce.monitors.score_matrix --monitor Qwen/Qwen3-8B --prompt monitor_default --view full \
           [--data outputs/data/transcripts.parquet] [--shard 0/4] [--n-samples 4 --temperature 0.7]   (4 samples averaged: integer scores tie at the 1% FPR threshold otherwise) [--datasets sleight apps]

Offline vLLM batch inference. Resumable: rows already present in the output directory are skipped, so a job that is
pre-empted or times out is simply resubmitted. Output parquet parts in <out>/<config_id>/ with columns
    row_id, config_id, monitor, prompt_id, view, sample_idx, score, raw_output, n_prompt_tokens, truncated
The same script runs the acting-side selectors of Experiment 3 (--prompt selector_default).
"""
import argparse, glob, os, re, time
import numpy as np, pandas as pd

SCORE_RE = re.compile(r"<score>\s*(\d{1,3}(?:\.\d+)?)\s*</score>", re.I)


FINAL_RE = re.compile(r"<\|channel\|>\s*final\s*<\|message\|>|assistantfinal")      # gpt-oss harmony: start of the final channel (with / without special tokens)


def parse_score(txt, think=False):
    """Never read a score out of the reasoning trace. If the reasoning was cut off by max_tokens (no end-of-reasoning
    marker), the answer is missing: return NaN instead of letting the fallback pick a number out of the trace."""
    txt = txt or ""; fin = list(FINAL_RE.finditer(txt))
    if fin: txt = txt[fin[-1].end():]
    elif "</think>" in txt: txt = txt.split("</think>")[-1]
    elif think or "<think>" in txt or re.match(r"\s*(<\|channel\|>)?\s*analysis", txt): return np.nan
    m = SCORE_RE.findall(txt)
    if not m: m = re.findall(r"(?<![\d.])(\d{1,3}(?:\.\d+)?)(?![\d.])", (txt or "")[-200:])     # fallback: last number in the tail
    if not m: return np.nan
    v = float(m[-1]); return v if 0 <= v <= 100 else np.nan


def build_body(row, view):
    if view == "action_only": return "(not shown to the monitor)", row.action
    if view == "task_action": return row.task or "(no task description available)", row.action
    ctx = f"{row.context}\n\n--- action under review ---\n" if row.context else ""
    return row.task or "(no task description available)", ctx + row.action


def fit(tok, text, budget):
    """Middle-out truncation to `budget` tokens: keeps the start (setup) and the end (the action under review)."""
    ids = tok(text, add_special_tokens=False).input_ids
    if len(ids) <= budget: return text, len(ids), False
    head = budget // 3; return tok.decode(ids[:head]) + "\n...[middle of the record truncated]...\n" + tok.decode(ids[-(budget - head):]), budget, True


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--monitor", required=True); ap.add_argument("--prompt", default="monitor_default"); ap.add_argument("--view", default="full", choices=["full", "action_only", "task_action"])
    ap.add_argument("--data", default=None); ap.add_argument("--out", default=None); ap.add_argument("--shard", default="0/1"); ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--attacks-only", action="store_true"); ap.add_argument("--n-samples", type=int, default=4); ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-model-len", type=int, default=16384); ap.add_argument("--max-new-tokens", type=int, default=256); ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--gpu-mem", type=float, default=0.90); ap.add_argument("--quantization", default=None); ap.add_argument("--tp", type=int, default=1); ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--think", action="store_true", help="let the monitor reason before scoring (Qwen3 thinking mode); use with --max-new-tokens 2048")
    ap.add_argument("--subsample", type=int, default=None, help="random N rows per (dataset, is_attack) cell, fixed seed: quick comparisons of monitor configs")
    a = ap.parse_args()
    from gammacert.core import out_dir, transcripts_path
    data = a.data or transcripts_path()
    cfg = f"{a.monitor.split('/')[-1]}__{a.prompt}__{a.view}" + ("__think" if a.think else "") + (f"__n{a.n_samples}" if a.n_samples != 4 else ""); odir = os.path.join(a.out or out_dir("scores"), cfg); os.makedirs(odir, exist_ok=True)
    df = pd.read_parquet(data)
    if a.datasets: df = df[df.dataset.isin(a.datasets)]
    if a.attacks_only: df = df[df.is_attack]
    if a.subsample: df = df.groupby(["dataset", "is_attack"], group_keys=False).apply(lambda g: g.sample(min(a.subsample, len(g)), random_state=0))
    k, K = map(int, a.shard.split("/")); df = df.sort_values("n_chars").iloc[k::K]            # length-sorted striping balances shards
    done = set(); [done.update(pd.read_parquet(p, columns=["row_id"]).row_id) for p in glob.glob(os.path.join(odir, "*.parquet"))]
    df = df[~df.row_id.isin(done)]
    if a.limit: df = df.iloc[: a.limit]
    print(f"[{cfg}] shard {a.shard}: {len(df)} rows to score, {len(done)} already done", flush=True)
    if df.empty: raise SystemExit

    from vllm import LLM, SamplingParams
    llm = LLM(a.monitor, max_model_len=a.max_model_len, gpu_memory_utilization=a.gpu_mem, enable_prefix_caching=True, quantization=a.quantization,
              tensor_parallel_size=a.tp, seed=0)
    tok = llm.get_tokenizer(); tmpl = open(os.path.join(os.path.dirname(__file__), "..", "prompts", a.prompt + ".txt")).read()
    sp = SamplingParams(n=a.n_samples, temperature=a.temperature, top_p=1.0 if a.temperature == 0 else 0.95, max_tokens=a.max_new_tokens, seed=0)
    budget = a.max_model_len - a.max_new_tokens - len(tok(tmpl).input_ids) - 600           # 600: chat-template overhead + task-statement slack
    tmpl_kwargs = {"enable_thinking": a.think, "reasoning_effort": "medium" if a.think else "low"}                        # Qwen3 / gpt-oss switches; ignored by other templates
    t0 = time.time(); ntok = 0
    for ci, start in enumerate(range(0, len(df), a.chunk)):
        part = df.iloc[start:start + a.chunk]; msgs, meta = [], []
        for row in part.itertuples():
            task, body = build_body(row, a.view); task, nt, _ = fit(tok, task, 1500); body, nb, tr = fit(tok, body, budget - nt)
            msgs.append([{"role": "user", "content": tmpl.replace("{task}", task).replace("{transcript}", body)}]); meta.append((row.row_id, nt + nb, tr))
        try: outs = llm.chat(msgs, sp, chat_template_kwargs=tmpl_kwargs, use_tqdm=False)
        except TypeError: outs = llm.chat(msgs, sp, use_tqdm=False)
        rows = [dict(row_id=rid, config_id=cfg, monitor=a.monitor, prompt_id=a.prompt, view=a.view, sample_idx=si, score=parse_score(c.text, a.think), raw_output=c.text, n_prompt_tokens=n, truncated=tr)
                for (rid, n, tr), o in zip(meta, outs) for si, c in enumerate(o.outputs)]
        fn = os.path.join(odir, f"part-{k:03d}of{K:03d}-{int(time.time())}-{ci:04d}.parquet")
        pd.DataFrame(rows).to_parquet(fn + ".tmp", index=False); os.replace(fn + ".tmp", fn)      # atomic: a job killed mid-write must not leave a corrupt part that breaks resume
        ntok += sum(m[1] for m in meta); bad = np.mean([np.isnan(r["score"]) for r in rows])
        print(f"  chunk {ci}: {len(part)} rows | {ntok / (time.time() - t0):.0f} prompt tok/s | unparsed {bad:.1%}", flush=True)
    print(f"[{cfg}] done in {(time.time() - t0) / 60:.1f} min")
