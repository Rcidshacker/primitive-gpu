"""Summarise primitive -trace JSONL output: where time goes and what a GPU port faces."""
import json
import statistics as st
import sys


def load(path):
    with open(path) as fp:
        return [json.loads(line) for line in fp]


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(len(xs) * p / 100))]


def phase(rows, name):
    if not rows:
        return
    search = sum(r["search_ms"] for r in rows)
    commit = sum(r["commit_ms"] for r in rows)
    evals = sum(r["evals"] for r in rows)
    climbs = [c for r in rows for c in r["climb_evals"]]
    px = sum(r["eval_pixels"] for r in rows)
    ln = sum(r["eval_lines"] for r in rows)
    print(f"  {name:<10} steps={len(rows):<4} step_ms={(search+commit)/len(rows):7.2f} "
          f"commit%={100*commit/(search+commit):4.1f} evals/step={evals/len(rows):7.0f} "
          f"climb p50/p90/max={pct(climbs,50)}/{pct(climbs,90)}/{max(climbs)} "
          f"cand span={px/ln:5.1f}px lines/cand={ln/evals:4.1f} px/cand={px/evals:6.1f} "
          f"commit span={st.mean(r['pixels']/max(r['lines'],1) for r in rows):5.1f}px")


def main():
    for path in sys.argv[1:]:
        rows = load(path)
        n = len(rows)
        print(f"{path}: {n} steps, final score {rows[-1]['score']:.4f}")
        phase(rows[: n // 3], "early")
        phase(rows[n // 3 : 2 * n // 3], "mid")
        phase(rows[2 * n // 3 :], "late")
        phase(rows, "all")
        tot = sum(r["search_ms"] + r["commit_ms"] for r in rows)
        ev = sum(r["evals"] for r in rows)
        print(f"  total {tot/1000:.2f}s, {ev} evals => {ev/(tot/1000)/1000:.0f}k evals/s wall")


main()
