"""
SQL analysis of the retail-sales series
=======================================

Loads the monthly retail data into an in-memory SQLite database and runs the
analysis queries in sql/queries.sql — yearly totals, the seasonal index,
record months, and year-over-year growth (using a SQL window function).

This shows the same series answered with SQL rather than pandas — the two are
complementary tools every data analyst is expected to know.

Run:  python sql_analysis.py

Author: Muhammad Nasiruddin
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).parent / "data" / "us_retail_sales.csv"
SQL_PATH = Path(__file__).parent / "sql" / "queries.sql"
OUT_DIR = Path(__file__).parent / "sql_output"


def load_queries(path: Path = SQL_PATH) -> dict[str, str]:
    """Parse queries.sql into {name: sql} using the `-- name:` headers."""
    text = path.read_text()
    queries: dict[str, str] = {}
    name = None
    buf: list[str] = []
    for line in text.splitlines():
        header = re.match(r"--\s*name:\s*(\w+)", line)
        if header:
            if name:
                queries[name] = "\n".join(buf).strip()
            name, buf = header.group(1), []
        elif name is not None:
            buf.append(line)
    if name:
        queries[name] = "\n".join(buf).strip()
    return queries


def build_db(csv_path: Path = DATA_PATH) -> sqlite3.Connection:
    """Load the CSV into an in-memory SQLite table `retail_sales`."""
    df = pd.read_csv(csv_path)
    df = df.rename(columns={"observation_date": "date", "RSXFSN": "sales"})
    conn = sqlite3.connect(":memory:")
    df.to_sql("retail_sales", conn, index=False)
    return conn


def run(conn: sqlite3.Connection, sql: str) -> pd.DataFrame:
    """Run a query and return the result as a DataFrame."""
    return pd.read_sql_query(sql, conn)


def main() -> None:
    conn = build_db()
    queries = load_queries()
    OUT_DIR.mkdir(exist_ok=True)

    # Run every query, print a preview, and save the full result as a CSV so the
    # output is viewable directly in the repo (GitHub renders CSVs as tables).
    for name, sql in queries.items():
        result = run(conn, sql)
        result.to_csv(OUT_DIR / f"{name}.csv", index=False)
        print(f"=== {name} ({len(result)} rows) ===")
        print(result.head(12).to_string(index=False), "\n")

    conn.close()
    print(f"Saved {len(queries)} query results to {OUT_DIR.name}/")


if __name__ == "__main__":
    main()
