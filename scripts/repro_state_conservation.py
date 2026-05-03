#!/usr/bin/env python3
"""Smoke test: assert that every cell in the /api/infer response satisfies
the per-cell fraction conservation invariant (sum-to-1).

A failure here means the env's per-cell modifiable mass changed during
inference, which usually points to ``env.samples`` not actually pointing at
the requested study area's data — the original symptom of the F3.3 bug
(cells with sum=2.0 from water-cell built-area creation).

Usage:
    python scripts/repro_state_conservation.py \
        --backend http://127.0.0.1:8000 \
        --lat -14.033739 --lng 34.525155 \
        --experiment exp1
"""
import argparse
import json
import sys
import urllib.request


def post(backend: str, path: str, body: dict) -> dict:
    # Bypass any system-level proxy — we're hitting localhost.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    req = urllib.request.Request(
        f"{backend.rstrip('/')}{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    return json.loads(opener.open(req, timeout=180).read())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", default="http://127.0.0.1:8000")
    p.add_argument("--lat", type=float, default=-14.033739)
    p.add_argument("--lng", type=float, default=34.525155)
    p.add_argument("--experiment", default="exp1")
    p.add_argument("--tolerance", type=float, default=0.05,
                   help="cells with |sum - 1| above this count as violations")
    args = p.parse_args()

    print(f"POST {args.backend}/api/infer  ({args.lat}, {args.lng}, {args.experiment})")
    r = post(args.backend, "/api/infer",
             {"lat": args.lat, "lng": args.lng, "experiment": args.experiment})

    n_rows = len(r["before"])
    n_cols = len(r["before"][0])
    n_total = n_rows * n_cols

    def violations(grid):
        return [
            (r, c, sum(grid[r][c].values()))
            for r in range(n_rows) for c in range(n_cols)
            if abs(sum(grid[r][c].values()) - 1.0) > args.tolerance
        ]

    bad_before = violations(r["before"])
    bad_after = violations(r["after"])

    print(f"  before-grid violations: {len(bad_before)}/{n_total}")
    print(f"  after-grid  violations: {len(bad_after)}/{n_total}")
    if bad_after:
        print(f"  first 5 after violations: {bad_after[:5]}")

    if bad_before or bad_after:
        sys.exit(1)
    print("OK — invariant holds for every cell.")


if __name__ == "__main__":
    main()
