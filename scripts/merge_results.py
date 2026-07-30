#!/usr/bin/env python3
"""Merge multiple results.csv files (Phase 6 + Phase 7) into one.

Unions columns across files. Warns on duplicate (arm,dataset,seq_len,pred_len,
seed) keys and keeps the LAST occurrence (later --csvs win).
"""

import argparse
import csv
import os
from collections import OrderedDict

KEY = ["arm", "dataset", "seq_len", "pred_len", "seed"]


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csvs", required=True, help="comma-separated results.csv paths")
    p.add_argument("--output", required=True)
    args = p.parse_args()

    paths = [x.strip() for x in args.csvs.split(",") if x.strip()]
    fields, merged, dupes = [], OrderedDict(), 0
    for path in paths:
        if not os.path.isfile(path):
            print(f"[merge] WARNING: missing {path}, skipping")
            continue
        for r in read_rows(path):
            for k in r:
                if k not in fields:
                    fields.append(k)
            key = tuple(r.get(k, "") for k in KEY)
            if key in merged:
                dupes += 1
            merged[key] = r  # last wins

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in merged.values():
            w.writerow({k: r.get(k, "") for k in fields})
    if dupes:
        print(f"[merge] WARNING: {dupes} duplicate key rows (kept last occurrence)")
    print(f"rows={len(merged)} cols={len(fields)} output={args.output}")


if __name__ == "__main__":
    main()
