"""Check configs/taxonomy_map.yaml against every raw step name in the dataset.

Reads ``<raw_logs_dir>/drive/steps_long.csv`` (2,830 rows), runs each
``sequence_name_raw`` + ``target_nest_group`` through ``parse_raw_name`` and
reports:

* rows that hit no rule (``matched_rule == "fallback"``) -- target < 1%;
* rows whose ``canon_group`` disagrees with the csv's own ``target_canon``
  column, which is the earlier analysis pass we are porting.

Both lists are written to ``configs/taxonomy_map_unmatched.txt``.

    python scripts/taxonomy_coverage.py
"""
from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tda.core.taxonomy import FALLBACK_RULE, parse_raw_name  # noqa: E402

OUT_PATH = REPO_ROOT / "configs" / "taxonomy_map_unmatched.txt"


def steps_csv_path() -> Path:
    with open(REPO_ROOT / "configs" / "paths.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return Path(cfg["raw_logs_dir"]) / "drive" / "steps_long.csv"


def main() -> int:
    path = steps_csv_path()
    if not path.exists():
        print(f"steps_long.csv not found at {path}", file=sys.stderr)
        return 2

    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    unmatched: Counter[tuple[str, str]] = Counter()
    mismatched: Counter[tuple[str, str, str]] = Counter()
    for row in rows:
        raw = row["sequence_name_raw"]
        nest = row.get("target_nest_group", "") or ""
        parsed = parse_raw_name(raw, nest)
        if parsed.matched_rule == FALLBACK_RULE:
            unmatched[(raw, nest)] += 1
        elif parsed.canon_group != row["target_canon"]:
            mismatched[(raw, parsed.canon_group, row["target_canon"])] += 1

    total = len(rows)
    n_unmatched = sum(unmatched.values())
    n_mismatched = sum(mismatched.values())
    rate = 100.0 * n_unmatched / total if total else 0.0

    print(f"rows                : {total}")
    print(f"unmatched (fallback): {n_unmatched} ({rate:.2f}%) over {len(unmatched)} distinct names")
    print(f"canon_group mismatch: {n_mismatched} over {len(mismatched)} distinct names")
    for (raw, nest), n in unmatched.most_common():
        print(f"  UNMATCHED x{n:<4} {raw!r} nest={nest!r}")
    for (raw, got, want), n in mismatched.most_common():
        print(f"  MISMATCH  x{n:<4} {raw!r}: got {got!r}, csv says {want!r}")

    lines = [
        "# Raw step names configs/taxonomy_map.yaml does not resolve.",
        "# Regenerate with: python scripts/taxonomy_coverage.py",
        f"# rows={total} unmatched={n_unmatched} ({rate:.2f}%) "
        f"canon_group_mismatch={n_mismatched}",
        "",
        "## unmatched (matched_rule == fallback)",
    ]
    lines += [f"{n}\t{raw}\tnest={nest}" for (raw, nest), n in unmatched.most_common()] or ["(none)"]
    lines += ["", "## canon_group differs from steps_long.csv target_canon"]
    lines += [
        f"{n}\t{raw}\tgot={got}\tcsv={want}"
        for (raw, got, want), n in mismatched.most_common()
    ] or ["(none)"]
    OUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT_PATH}")

    return 0 if rate < 1.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
