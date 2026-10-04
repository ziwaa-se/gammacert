"""Build the unified transcript table that every monitor scores.

    python -m reproduce.monitors.data_prep --sources apps basharena sleight --n-apps 1500
    python -m reproduce.monitors.data_prep --inspect sleight          # print what a gated dataset actually looks like

Output: <out>/transcripts.parquet with one row per (transcript to be judged):
    row_id, dataset, is_attack, attack_type, meta (json), task, context, action
`task` is what the agent was asked to do, `context` is what happened before, `action` is what is judged.
Views (full / action_only / task_action) are assembled by score_matrix.py, not here.

Needs network access to the HF hub (or a warm HF cache). SLEIGHT-Bench and MALT are gated: accept the terms on
the dataset page and `hf auth login` first. SLEIGHT-Bench carries a canary string: evaluation only.
"""
import argparse, glob, json, os
import numpy as np, pandas as pd

MAX_TOOL_OUT = 2000          # characters of any single tool output kept when rendering a trajectory


def render_messages(msgs):
    """Generic renderer for chat / tool-call trajectories (list of dicts, or a JSON string of one)."""
    if isinstance(msgs, str):
        try: msgs = json.loads(msgs)
        except Exception: return msgs
    if isinstance(msgs, dict): msgs = [msgs]
    out = []
    for m in msgs:
        if not isinstance(m, dict): out.append(str(m)); continue
        if "function" in m and "arguments" in m and "role" not in m:            # a bare tool call
            out.append(f"[tool call] {m['function']}({json.dumps(m['arguments'], ensure_ascii=False)})"); continue
        role = m.get("role", m.get("type", "?")); content = m.get("content", "")
        if not isinstance(content, str): content = json.dumps(content, ensure_ascii=False)
        if role == "tool" and len(content) > MAX_TOOL_OUT: content = content[: MAX_TOOL_OUT // 2] + "\n...[truncated]...\n" + content[-MAX_TOOL_OUT // 2:]
        if content: out.append(f"[{role}{' ' + str(m['function']) if role == 'tool' and m.get('function') else ''}] {content}")
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function", tc); name = fn.get("name", fn) if isinstance(fn, dict) else fn
            args = tc.get("arguments", fn.get("arguments") if isinstance(fn, dict) else None)
            out.append(f"[tool call] {name}({json.dumps(args, ensure_ascii=False) if not isinstance(args, str) else args})")
    return "\n".join(out)


def hub_parquet(repo):
    """Read a parquet dataset straight from the hub snapshot: works offline once the datasets are in the Hugging Face cache."""
    from huggingface_hub import snapshot_download
    root = snapshot_download(repo, repo_type="dataset", allow_patterns=["*.parquet"])
    return pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(os.path.join(root, "**", "*.parquet"), recursive=True))], ignore_index=True)


def prep_apps(n, rng):
    ds = hub_parquet("RoganInglis/apps-control-arena")
    ds = ds[ds.has_backdoor & ds.backdoor_works & ds.solution_passes_tests & ~ds.is_nondeterministic]
    ds = ds.sample(min(n, len(ds)), random_state=int(rng.integers(1 << 31))); rows = []
    for _, p in ds.iterrows():
        good = [s for s in p.solutions if s.get("passes_tests")]; bds = [b for b in p.backdoors if b.get("backdoor_works") and b.get("passes_tests")]
        if not good or not bds: continue
        rows.append(dict(row_id=f"apps-{p.problem_id}-benign", dataset="apps", is_attack=False, attack_type="", task=p.question, context="", action=good[0]["code"],
                         meta=json.dumps(dict(difficulty=p.difficulty))))
        seen = set()
        for b in bds:                                                        # at most one backdoor per generating model, two per problem
            key = f"{b.get('source')}:{b.get('model') or 'human+gpt4'}"
            if key in seen or len(seen) >= 2: continue
            seen.add(key); rows.append(dict(row_id=f"apps-{p.problem_id}-bd{len(seen)}", dataset="apps", is_attack=True, attack_type=key, task=p.question, context="",
                                            action=b["backdoor_code"], meta=json.dumps(dict(difficulty=p.difficulty, source=b.get("source"), model=b.get("model")))))
    return rows


def prep_basharena(rng):
    ds = hub_parquet("abhayesian/basharena-monitor-eval"); rows = []
    for _, p in ds.iterrows():
        rows.append(dict(row_id=f"bash-{p.sample_id}-{p.action_idx}", dataset="basharena", is_attack=p.label == "attack", attack_type=p.side_task or "",
                         task=p.task_description, context=render_messages(p.previous_actions), action=render_messages(p.action),
                         meta=json.dumps(dict(action_idx=int(p.action_idx), side_task_success=p.side_task_success, original_monitor_score=None if pd.isna(p.original_monitor_score) else float(p.original_monitor_score)))))
    return rows


def sleight_dir():
    from huggingface_hub import snapshot_download
    return snapshot_download("sleightbench/SLEIGHT-Bench", repo_type="dataset")


def render_sleight(path):
    """SLEIGHT FORMAT.md: one content block per JSONL line; the first line is a canary object (no role) and is skipped.
    Returns (task, body): the first user text block is the task, everything after it is the behaviour under review."""
    task, out = "", []
    for line in open(path):
        if not line.strip(): continue
        o = json.loads(line)
        if "role" not in o: continue                                   # canary line
        b = o.get("content") or {}; t = b.get("type") if isinstance(b, dict) else None
        if t == "text": txt = f"[{o['role']}] {b.get('text', '')}"
        elif t == "thinking": txt = f"[assistant thinking] {b.get('thinking', '')}"
        elif t == "tool_use": txt = f"[tool call] {b.get('name')}({json.dumps(b.get('input'), ensure_ascii=False)})"
        elif t == "tool_result":
            c = b.get("content"); c = c if isinstance(c, str) else json.dumps(c, ensure_ascii=False)
            if len(c) > MAX_TOOL_OUT: c = c[: MAX_TOOL_OUT // 2] + "\n...[truncated]...\n" + c[-MAX_TOOL_OUT // 2:]
            txt = f"[tool result] {c}"
        else: txt = f"[{o['role']}] {json.dumps(b, ensure_ascii=False)}"
        if not task and o["role"] == "user" and t == "text": task = b.get("text", ""); continue
        out.append(txt)
    return task, "\n".join(out)


def prep_sleight(rng):
    """attacks/<category>/<attack>/{transcript.jsonl (attack), benign.jsonl (malicious action removed), metadata.json, ...}.
    metadata (outcome, mechanism, ...) describes the attack and is never shown to a monitor; only `tags` are kept for analysis."""
    root = sleight_dir(); rows = []
    for d in sorted(glob.glob(os.path.join(root, "attacks", "*", "*"))):
        cat, name = d.split(os.sep)[-2:]; meta = json.load(open(os.path.join(d, "metadata.json")))
        for fname, atk in (("transcript.jsonl", True), ("benign.jsonl", False)):
            task, body = render_sleight(os.path.join(d, fname))
            rows.append(dict(row_id=f"sleight-{cat}-{name}-{'attack' if atk else 'benign'}", dataset="sleight", is_attack=atk, attack_type=cat if atk else "",
                             task=task, context="", action=body, meta=json.dumps(dict(category=cat, attack=name, tags=meta.get("tags")))))
    return rows


def prep_malt(rng, n=300):
    raise NotImplementedError("MALT (metr-evals/malt-transcripts-public) is gated and its `nodes`/`labels` schema has not been inspected yet. "
                              "Run `--inspect malt`, then render `nodes` with render_messages(), map `labels` to is_attack/attack_type, and use the five "
                              "configs (default, irrelevant_detail, language_mixing, summarize, vague_cot) as the obfuscation axis in `meta`. "
                              "Trajectories are ~1MB each: subsample (n=300) and let score_matrix.py window them.")


def inspect(which):
    if which == "sleight":
        root = sleight_dir(); print(open(os.path.join(root, "FORMAT.md")).read()[:6000]); d = sorted(glob.glob(os.path.join(root, "attacks", "*", "*")))
        print(len(d), "attacks;", "categories:", sorted({x.split(os.sep)[-2] for x in d})); print(open(os.path.join(d[0], "transcript.jsonl")).read()[:3000])
    else:
        from datasets import load_dataset
        name = {"malt": "metr-evals/malt-transcripts-public", "apps": "RoganInglis/apps-control-arena", "basharena": "abhayesian/basharena-monitor-eval"}[which]
        ds = load_dataset(name, split=None, streaming=True); sp = list(ds.keys())[0]; ex = next(iter(ds[sp])); print("splits:", list(ds.keys()))
        for k, v in ex.items(): print(f"--- {k} ({type(v).__name__}):", (json.dumps(v, default=str) if not isinstance(v, str) else v)[:1500])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--sources", nargs="+", default=["apps", "basharena", "sleight"]); ap.add_argument("--n-apps", type=int, default=1500)
    ap.add_argument("--out", default=None); ap.add_argument("--inspect", default=None); ap.add_argument("--seed", type=int, default=0); a = ap.parse_args()
    if a.inspect: inspect(a.inspect); raise SystemExit
    from gammacert.core import out_dir
    out = a.out or out_dir("data"); rng = np.random.default_rng(a.seed); rows = []
    for s in a.sources:
        try: new = {"apps": lambda: prep_apps(a.n_apps, rng), "basharena": lambda: prep_basharena(rng), "sleight": lambda: prep_sleight(rng), "malt": lambda: prep_malt(rng)}[s]()
        except Exception as e: print(f"[WARN] source {s} skipped: {type(e).__name__}: {e}"); continue
        print(f"{s}: {len(new)} rows ({sum(r['is_attack'] for r in new)} attacks)"); rows += new
    df = pd.DataFrame(rows).drop_duplicates("row_id"); df["n_chars"] = df.task.str.len() + df.context.str.len() + df.action.str.len()
    df.to_parquet(os.path.join(out, "transcripts.parquet"), index=False)
    print(df.groupby(["dataset", "is_attack"]).agg(n=("row_id", "size"), median_chars=("n_chars", "median"), max_chars=("n_chars", "max")))
    print(df[df.is_attack].groupby(["dataset", "attack_type"]).size().to_string()); print("wrote", os.path.join(out, "transcripts.parquet"))
