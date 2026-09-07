"""
CSV-to-Postgres loader functions for the 9 real (Olist-derived) tables.

Pattern used throughout:
1. Read raw CSV with pandas
2. Select/reorder columns to match the target table's schema exactly
3. Convert pandas NaN -> empty string (so COPY's `NULL ''` maps it correctly)
4. Write to an in-memory buffer as tab-delimited CSV (no header, no index)
5. COPY the buffer into Postgres via psycopg's copy_expert

Each function commits its own transaction — per-table atomicity, so a
failure in one loader does not roll back tables already loaded successfully.
"""

import io
import pandas as pd


def load_customers(csv_path: str, conn) -> None:
    """Loads both `customers` (deduplicated) and `customer_crosswalk` (full) from
    the same raw customers.csv, since Olist stores both IDs in one file."""
    df = pd.read_csv(csv_path)

    # customers: one row per real person
    customers_df = df[[
        "customer_unique_id",
        "customer_zip_code_prefix",
        "customer_city",
        "customer_state"
    ]].drop_duplicates(subset="customer_unique_id", keep="first")

    # customer_crosswalk: every order-scoped customer_id preserved, no dedup
    crosswalk_df = df[[
        "customer_id",
        "customer_unique_id"
    ]]

    customers_df = customers_df.where(pd.notna(customers_df), "")
    crosswalk_df = crosswalk_df.where(pd.notna(crosswalk_df), "")

    with conn.cursor() as cur:
        # customers must load first — crosswalk has an FK to it
        buffer = io.StringIO()
        customers_df.to_csv(buffer, index=False, header=False, sep="\t")
        buffer.seek(0)
        cur.copy_expert(
            "COPY customers FROM STDIN WITH (FORMAT csv, DELIMITER E'\\t', NULL '')",
            buffer
        )

        buffer = io.StringIO()
        crosswalk_df.to_csv(buffer, index=False, header=False, sep="\t")
        buffer.seek(0)
        cur.copy_expert(
            "COPY customer_crosswalk FROM STDIN WITH (FORMAT csv, DELIMITER E'\\t', NULL '')",
            buffer
        )

    conn.commit()


def load_category_translation(csv_path: str, conn) -> None:
    df = pd.read_csv(csv_path)
    df = df[["product_category_name", "product_category_name_english"]]

    buffer = io.StringIO()
    df.to_csv(buffer, index=False, header=False, sep="\t")
    buffer.seek(0)

    with conn.cursor() as cur:
        cur.copy_expert(
            "COPY product_category_name_translation FROM STDIN WITH (FORMAT csv, DELIMITER E'\\t', NULL '')",
            buffer
        )
    conn.commit()


def load_products(csv_path: str, conn) -> None:
    """Assumes product_name_lenght / product_description_lenght have already
    been manually corrected to product_name_length / product_description_length
    in the raw CSV (project-specific decision — see data/raw/README.md note)."""
    df = pd.read_csv(csv_path)

    df = df[[
        "product_id",
        "product_category_name",
        "product_name_length",
        "product_description_length",
        "product_photos_qty",
        "product_weight_g",
        "product_length_cm",
        "product_height_cm",
        "product_width_cm"
    ]]

    # These columns are read as float64 by pandas when the source has any
    # missing values (NaN forces the whole column to float). Cast to the
    # nullable Int64 dtype so real values print as "40" not "40.0" — Postgres's
    # INTEGER columns reject the latter. Int64 (capital I) can hold <NA>
    # internally while formatting whole numbers correctly on to_csv.
    #
    # Note: no separate NaN->"" step needed here — to_csv's default na_rep=''
    # already converts any missing value (NaN or Int64's <NA>) to an empty
    # string, which is what COPY's NULL '' expects. Calling .where() on top
    # of an Int64 column actually raises a TypeError, since you can't assign
    # a plain string into a masked integer array that way.
    for int_col in ["product_name_length", "product_description_length", "product_photos_qty"]:
        df[int_col] = df[int_col].astype("Int64")

    buffer = io.StringIO()
    df.to_csv(buffer, index=False, header=False, sep="\t")
    buffer.seek(0)

    with conn.cursor() as cur:
        cur.copy_expert(
            "COPY products FROM STDIN WITH (FORMAT csv, DELIMITER E'\\t', NULL '')",
            buffer
        )
    conn.commit()


def load_sellers(csv_path: str, conn) -> None:
    """employee_id is synthetic and not present in the raw CSV — explicit
    column list tells COPY to leave employee_id NULL for every row."""
    df = pd.read_csv(csv_path)

    df = df[[
        "seller_id",
        "seller_zip_code_prefix",
        "seller_city",
        "seller_state"
    ]]

    df = df.where(pd.notna(df), "")

    buffer = io.StringIO()
    df.to_csv(buffer, index=False, header=False, sep="\t")
    buffer.seek(0)

    with conn.cursor() as cur:
        cur.copy_expert(
            "COPY sellers (seller_id, seller_zip_code_prefix, seller_city, seller_state) "
            "FROM STDIN WITH (FORMAT csv, DELIMITER E'\\t', NULL '')",
            buffer
        )
    conn.commit()


def load_orders(csv_path: str, conn) -> None:
    df = pd.read_csv(csv_path)

    df = df[[
        "order_id",
        "customer_id",
        "order_status",
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date"
    ]]

    df = df.where(pd.notna(df), "")

    buffer = io.StringIO()
    df.to_csv(buffer, index=False, header=False, sep="\t")
    buffer.seek(0)

    with conn.cursor() as cur:
        cur.copy_expert(
            "COPY orders FROM STDIN WITH (FORMAT csv, DELIMITER E'\\t', NULL '')",
            buffer
        )
    conn.commit()


def load_order_items(csv_path: str, conn) -> None:
    df = pd.read_csv(csv_path)

    df = df[[
        "order_id",
        "order_item_id",
        "product_id",
        "seller_id",
        "shipping_limit_date",
        "price",
        "freight_value"
    ]]

    df = df.where(pd.notna(df), "")

    buffer = io.StringIO()
    df.to_csv(buffer, index=False, header=False, sep="\t")
    buffer.seek(0)

    with conn.cursor() as cur:
        cur.copy_expert(
            "COPY order_items FROM STDIN WITH (FORMAT csv, DELIMITER E'\\t', NULL '')",
            buffer
        )
    conn.commit()


def load_order_payments(csv_path: str, conn) -> None:
    df = pd.read_csv(csv_path)

    df = df[[
        "order_id",
        "payment_sequential",
        "payment_type",
        "payment_installments",
        "payment_value"
    ]]

    df = df.where(pd.notna(df), "")

    buffer = io.StringIO()
    df.to_csv(buffer, index=False, header=False, sep="\t")
    buffer.seek(0)

    with conn.cursor() as cur:
        cur.copy_expert(
            "COPY order_payments FROM STDIN WITH (FORMAT csv, DELIMITER E'\\t', NULL '')",
            buffer
        )
    conn.commit()


def load_order_reviews(csv_path: str, conn) -> None:
    df = pd.read_csv(csv_path)

    df = df[[
        "review_id",
        "order_id",
        "review_score",
        "review_comment_title",
        "review_comment_message",
        "review_creation_date",
        "review_answer_timestamp"
    ]]

    df = df.where(pd.notna(df), "")

    buffer = io.StringIO()
    df.to_csv(buffer, index=False, header=False, sep="\t")
    buffer.seek(0)

    with conn.cursor() as cur:
        cur.copy_expert(
            "COPY order_reviews FROM STDIN WITH (FORMAT csv, DELIMITER E'\\t', NULL '')",
            buffer
        )
    conn.commit()
