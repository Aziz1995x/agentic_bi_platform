"""
Orchestrates the full CSV-to-Postgres load in strict FK-dependency order.

Design decisions locked in during review:
- One shared connection, but each load_* function commits its own transaction
  (per-table atomicity — a failure partway through does not undo tables
  already loaded successfully).
- On any exception, stop immediately and re-raise (no silent partial success).
- No auto-truncate/reset here by design — see reset_database() (separate,
  not yet written) for deliberately wiping tables before a clean rerun.
- regions / state_region_map are NOT included here — they are static seed
  data loaded once via seed_regions.sql, not part of this Python pipeline.
- employees / inventory / returns are also NOT included here yet — they
  depend on sellers/products/order_items already being loaded, and need
  their own follow-up step (generate -> load) after this pipeline succeeds.
"""

import psycopg2

from src.agentic_bi.config.logging import get_logger
from src.agentic_bi.database.loaders import (
    load_customers,
    load_category_translation,
    load_products,
    load_sellers,
    load_orders,
    load_order_items,
    load_order_payments,
    load_order_reviews,
)

logger = get_logger(__name__)


def get_connection(config):
    """Uses the individual Settings fields directly, NOT config.database_url —
    that property is built as postgresql+asyncpg://... for the app's async
    SQLAlchemy engine, which psycopg2.connect() cannot parse. This batch
    script is deliberately sync (see design discussion), so it needs a
    plain psycopg2-style connection instead."""
    return psycopg2.connect(
        host=config.postgres_host,
        port=config.postgres_port,
        dbname=config.postgres_db,
        user=config.postgres_user,
        password=config.postgres_password,
    )


def run_pipeline(config) -> None:
    conn = get_connection(config)

    loaders = [
        ("customers", lambda: load_customers(config.customers_csv, conn)),
        ("product_category_name_translation", lambda: load_category_translation(config.category_translation_csv, conn)),
        ("products", lambda: load_products(config.products_csv, conn)),
        ("sellers", lambda: load_sellers(config.sellers_csv, conn)),
        ("orders", lambda: load_orders(config.orders_csv, conn)),
        ("order_items", lambda: load_order_items(config.order_items_csv, conn)),
        ("order_payments", lambda: load_order_payments(config.order_payments_csv, conn)),
        ("order_reviews", lambda: load_order_reviews(config.order_reviews_csv, conn)),
    ]

    table_name = None
    try:
        for table_name, loader in loaders:
            logger.info("Loading table: %s", table_name)
            loader()
            logger.info("Successfully loaded table: %s", table_name)
    except Exception:
        logger.exception("Pipeline failed while loading table: %s", table_name)
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    from src.agentic_bi.config.settings import get_settings
    run_pipeline(get_settings())
