"""
Loads already-generated synthetic DataFrames (employees, inventory, returns)
into Postgres via COPY — same buffer pattern as the real-table loaders.
"""

import io
import pandas as pd


def load_employees_df(employees_df: pd.DataFrame, conn) -> None:
    df = employees_df[["employee_id", "employee_name", "role", "region_id", "hire_date"]]
    df = df.where(pd.notna(df), "")

    buffer = io.StringIO()
    df.to_csv(buffer, index=False, header=False, sep="\t")
    buffer.seek(0)

    with conn.cursor() as cur:
        cur.copy_expert(
            "COPY employees FROM STDIN WITH (FORMAT csv, DELIMITER E'\\t', NULL '')",
            buffer
        )
    conn.commit()


def load_inventory_df(inventory_df: pd.DataFrame, conn) -> None:
    df = inventory_df[["product_id", "stock_quantity", "restock_date"]]
    df = df.where(pd.notna(df), "")

    buffer = io.StringIO()
    df.to_csv(buffer, index=False, header=False, sep="\t")
    buffer.seek(0)

    with conn.cursor() as cur:
        cur.copy_expert(
            "COPY inventory FROM STDIN WITH (FORMAT csv, DELIMITER E'\\t', NULL '')",
            buffer
        )
    conn.commit()


def load_returns_df(returns_df: pd.DataFrame, conn) -> None:
    df = returns_df[["return_id", "order_id", "order_item_id", "return_reason", "return_date", "refund_amount"]]
    df = df.where(pd.notna(df), "")

    buffer = io.StringIO()
    df.to_csv(buffer, index=False, header=False, sep="\t")
    buffer.seek(0)

    with conn.cursor() as cur:
        cur.copy_expert(
            "COPY returns FROM STDIN WITH (FORMAT csv, DELIMITER E'\\t', NULL '')",
            buffer
        )
    conn.commit()


def fetch_seller_ids(conn) -> list[str]:
    with conn.cursor() as cur:
        cur.execute("SELECT seller_id FROM sellers")
        return [row[0] for row in cur.fetchall()]


def fetch_product_ids(conn) -> list[str]:
    with conn.cursor() as cur:
        cur.execute("SELECT product_id FROM products")
        return [row[0] for row in cur.fetchall()]

def load_order_items_for_returns(conn) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.execute("SELECT order_id, order_item_id, price FROM order_items")
        rows = cur.fetchall()
        columns = [desc[0] for desc in cur.description]

    return pd.DataFrame(rows, columns=columns)


def load_orders_for_returns(conn) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.execute("SELECT order_id, order_purchase_timestamp FROM orders")
        rows = cur.fetchall()
        columns = [desc[0] for desc in cur.description]

    return pd.DataFrame(rows, columns=columns)


def fetch_order_items(conn) -> pd.DataFrame:
    # return pd.read_sql("SELECT order_id, order_item_id, price FROM order_items", conn)
    return load_order_items_for_returns(conn)


def fetch_orders(conn) -> pd.DataFrame:
    # return pd.read_sql("SELECT order_id, order_purchase_timestamp FROM orders", conn)
    return load_orders_for_returns(conn)
