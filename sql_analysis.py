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

    print("=== Year-over-year growth (last 8 years) ===")
    print(run(conn, queries["yoy_growth"]).tail(8).to_string(index=False), "\n")

    print("=== Seasonal index (100 = average month) ===")
    print(run(conn, queries["seasonal_index"]).to_string(index=False), "\n")

    print("=== Five highest-sales months on record ===")
    print(run(conn, queries["top_months"]).to_string(index=False), "\n")

    yearly = run(conn, queries["yearly_totals"])
    print(f"Years covered: {len(yearly)}  "
          f"({yearly['year'].iloc[0]}–{yearly['year'].iloc[-1]})")
    conn.close()


if __name__ == "__main__":
    main()
