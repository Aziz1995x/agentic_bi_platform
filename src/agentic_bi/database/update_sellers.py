"""
Writes seller -> employee assignments onto the already-loaded `sellers` table.

This is an UPDATE, not a COPY/INSERT, because the seller rows already exist
(loaded with employee_id = NULL by load_sellers). Uses execute_values with an
UPDATE ... FROM (VALUES ...) pattern for a single bulk round-trip instead of
one UPDATE per row.
"""

import pandas as pd
from psycopg2.extras import execute_values


def update_seller_employee_assignments(assignments_df: pd.DataFrame, conn) -> None:
    """
    assignments_df must have columns: seller_id, employee_id
    (the output of generate_employees.assign_employees_to_sellers)
    """
    records = list(assignments_df.itertuples(index=False, name=None))

    sql = """
        UPDATE sellers AS s
        SET employee_id = v.employee_id
        FROM (VALUES %s) AS v(seller_id, employee_id)
        WHERE s.seller_id = v.seller_id
    """

    with conn.cursor() as cur:
        execute_values(cur, sql, records)

    conn.commit()


def get_seller_ids(conn) -> list[str]:
    """Fetch real seller_ids already loaded, needed as input to
    assign_employees_to_sellers before calling the update above."""
    with conn.cursor() as cur:
        cur.execute("SELECT seller_id FROM sellers")
        return [row[0] for row in cur.fetchall()]
