#!/usr/bin/env python3
"""Assert that an opt-in target+draft FlashHead trace has no full-vocabulary Q6_K head."""

import json
import sqlite3
import sys
from pathlib import Path


def table(conn: sqlite3.Connection, pattern: str) -> str:
    row = conn.execute(
        "select name from sqlite_master where type='table' and name like ?", (pattern,)
    ).fetchone()
    if not row:
        raise RuntimeError(f"missing table {pattern}")
    return row[0]


def main() -> int:
    if len(sys.argv) != 3:
        print(f"usage: {Path(sys.argv[0]).name} TRACE.db whole-analysis.json", file=sys.stderr)
        return 2
    analysis = json.loads(Path(sys.argv[2]).read_text())
    conn = sqlite3.connect(sys.argv[1])
    dispatch = table(conn, "rocpd_kernel_dispatch_%")
    symbol = table(conn, "rocpd_info_kernel_symbol_%")
    rows = conn.execute(
        f"""
        select count(*), sum(d.end-d.start)/1e6
          from {dispatch} d join {symbol} k on k.id=d.kernel_id
         where d.end > ? and d.start < ?
           and d.grid_size_x = 7946240
           and k.display_name like '%mul_mat_vec_q<(ggml_type)14,%false, false>%'
        """,
        (analysis["request_start_ns"], analysis["request_end_ns"]),
    ).fetchone()
    count, milliseconds = int(rows[0]), float(rows[1] or 0)
    if count:
        print(f"FAIL: {count} full-vocabulary Q6_K launches remain ({milliseconds:.3f} ms)")
        return 1
    print("PASS: no full-vocabulary Q6_K LM-head launches")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
