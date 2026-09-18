"""Turn the per-mask CSVs into the markdown tables the report quotes.

Writes ``summary.md`` next to the CSVs in ``experiments_out/sam_compare/``:
overall, by size bucket, by label, and the prompt/model timing table, plus the
concept-prompt tables when ``concepts.csv`` exists.

Run::

    D:\\Anaconda\\envs\\tda\\python.exe -m experiments.sam_compare.aggregate
"""
from __future__ import annotations

import pandas as pd

from experiments.sam_compare import _env
from experiments.sam_compare.common import PROMPTS, SIZE_NAMES

#: Labels the tool cares about most -- small hardware an annotator clicks all day.
FOCUS = ("Screw", "Retention Clip", "Connector")


def load() -> pd.DataFrame:
    frames = []
    for name in ("interactive_sam21.csv", "interactive_sam3.csv"):
        path = _env.OUT_ROOT / name
        if path.is_file():
            frames.append(pd.read_csv(path))
    if not frames:
        raise SystemExit("no interactive_*.csv found; run run_interactive.py first")
    df = pd.concat(frames, ignore_index=True)
    return df[df["status"] == "ok"].copy()


def _stats(group: pd.DataFrame) -> pd.Series:
    return pd.Series({
        "n": len(group),
        "IoU_mean": group["iou"].mean(),
        "IoU_med": group["iou"].median(),
        "bF1_mean": group["boundary_f1"].mean(),
        "IoU>=0.75": (group["iou"] >= 0.75).mean(),
        "IoU>=0.9": (group["iou"] >= 0.9).mean(),
        "IoU_oracle": group["iou_oracle"].mean(),
        "decode_ms_med": group["decode_ms"].median(),
    })


#: Columns that are counts, so they print without decimals.
_INT_COLS = {"n", "support", "n_ref", "n_pred", "n_match"}


def _cell(value, floatfmt: str, integral: bool = False) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "-"
    if isinstance(value, float):
        return f"{value:.0f}" if integral else floatfmt.format(value)
    return str(value)


def _md(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    """Render a (possibly MultiIndex) frame as a GitHub markdown table.

    Hand-rolled rather than ``DataFrame.to_markdown`` because that needs
    ``tabulate``, which is not in this environment's pinned requirements and is
    not worth adding for four tables.
    """
    out = df.reset_index()
    header = [" / ".join(str(p) for p in c if p != "") if isinstance(c, tuple) else str(c)
              for c in out.columns]
    integral = [h.split(" / ")[0] in _INT_COLS for h in header]
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join("---" for _ in header) + "|"]
    for _, row in out.iterrows():
        lines.append("| " + " | ".join(
            _cell(v, floatfmt, i) for v, i in zip(row, integral)) + " |")
    return "\n".join(lines)


def table_overall(df: pd.DataFrame) -> pd.DataFrame:
    t = df.groupby(["model", "prompt"], sort=False).apply(_stats, include_groups=False)
    return t.reindex(pd.MultiIndex.from_product(
        [sorted(df["model"].unique()), list(PROMPTS)], names=["model", "prompt"]))


def table_by_size(df: pd.DataFrame) -> pd.DataFrame:
    t = df.groupby(["size_bucket", "model", "prompt"], sort=False).apply(
        _stats, include_groups=False)
    return t.reindex(pd.MultiIndex.from_product(
        [list(SIZE_NAMES), sorted(df["model"].unique()), list(PROMPTS)],
        names=["size_bucket", "model", "prompt"])).dropna(how="all")


def table_focus(df: pd.DataFrame) -> pd.DataFrame:
    """Small-hardware families: screws, RAM clips, connectors."""
    rows = []
    for family in FOCUS:
        sub = df[df["label"].str.contains(family, case=False, na=False)]
        if sub.empty:
            continue
        for model in sorted(sub["model"].unique()):
            for prompt in PROMPTS:
                g = sub[(sub["model"] == model) & (sub["prompt"] == prompt)]
                if g.empty:
                    continue
                stats = _stats(g)
                stats["family"] = family
                stats["model"] = model
                stats["prompt"] = prompt
                rows.append(stats)
    return pd.DataFrame(rows).set_index(["family", "model", "prompt"])


def table_by_label(df: pd.DataFrame) -> pd.DataFrame:
    best = df[df["prompt"] == "point_box"]
    t = best.groupby(["label", "model"], sort=False).apply(_stats, include_groups=False)
    return t.unstack("model")[["IoU_mean", "bF1_mean", "n"]].sort_values(
        ("n", "sam2.1"), ascending=False)


def table_timing(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in sorted(df["model"].unique()):
        sub = df[df["model"] == model]
        rows.append({
            "model": model,
            "embed_ms_med": sub["embed_ms"].median(),
            "embed_ms_p90": sub["embed_ms"].quantile(0.9),
            **{f"decode_{p}_ms_med": sub[sub["prompt"] == p]["decode_ms"].median()
               for p in PROMPTS},
        })
    return pd.DataFrame(rows).set_index("model")


def concepts_tables() -> str:
    path = _env.OUT_ROOT / "concepts.csv"
    if not path.is_file():
        return "\n## 6. Concept prompts\n\n_not run_\n"
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:  # still being written by a running job
        return "\n## 6. Concept prompts\n\n_incomplete_\n"
    out = ["\n## 3. SAM 3 concept (text) prompts\n"]
    agg = df.groupby(["framing", "concept", "threshold"], sort=False).agg(
        n_ref=("n_ref", "sum"), n_pred=("n_pred", "sum"), n_match=("n_match", "sum"),
        ms=("ms", "median"))
    agg["recall"] = agg["n_match"] / agg["n_ref"].replace(0, pd.NA)
    agg["precision_lb"] = agg["n_match"] / agg["n_pred"].replace(0, pd.NA)
    out.append(_md(agg.reset_index().set_index(["framing", "concept", "threshold"])))
    out.append("\n`precision_lb` is a *lower bound*: the reference only covers what the "
               "annotators drew, so a correct detection of an undrawn part counts "
               "against it.\n")
    return "\n".join(out)


def main() -> int:
    df = load()
    parts = [
        "# SAM 2.1 vs SAM 3 -- interactive prompt comparison\n",
        f"Rows: {len(df)} ok "
        f"(masks {df['index'].nunique()}, frames {df.groupby(['desktop', 'step']).ngroups}, "
        f"desktops {df['desktop'].nunique()}).\n",
        "\n## 1. Overall\n", _md(table_overall(df)),
        "\n\n## 2. By size bucket (reference area, native px)\n", _md(table_by_size(df)),
        "\n\n## 3. Small-hardware families\n", _md(table_focus(df)),
        "\n\n## 4. Timing (ms, median)\n", _md(table_timing(df), "{:.1f}"),
        "\n\n## 5. Per label (point+box prompt)\n", _md(table_by_label(df)),
        concepts_tables(),
    ]
    text = "\n".join(parts)
    out = _env.OUT_ROOT / "summary.md"
    out.write_text(text, encoding="utf-8")
    print(text)
    print(f"\n[aggregate] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
