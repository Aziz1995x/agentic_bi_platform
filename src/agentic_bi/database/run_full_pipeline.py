"""
Full end-to-end database build for the Agentic BI Platform.

Order of operations:
1. Run seed_regions.sql (static regions + state_region_map — no generation)
2. Run the real-table pipeline (customers -> ... -> order_reviews)
3. Generate + load employees (needs regions already seeded)
4. Update sellers.employee_id (needs employees + sellers already loaded)
5. Generate + load inventory (needs products already loaded)
6. Generate + load returns (needs order_items + orders already loaded)

Run once against a freshly created (schema.sql already applied) database.
Re-running against a non-empty database will fail on primary key conflicts —
this script does not truncate/reset anything by design (see prior discussion
on keeping reset logic separate from load logic).
"""

from pathlib import Path

from src.agentic_bi.config.logging import get_logger
from src.agentic_bi.config.settings import get_settings
from src.agentic_bi.database.pipeline import get_connection, run_pipeline
from src.agentic_bi.database.synthetic_loaders import (
    load_employees_df,
    load_inventory_df,
    load_returns_df,
    fetch_seller_ids,
    fetch_product_ids,
    fetch_order_items,
    fetch_orders,
)
from src.agentic_bi.database.update_sellers import update_seller_employee_assignments
from src.agentic_bi.database.generate_employees import generate_employees, assign_employees_to_sellers
from src.agentic_bi.database.generate_inventory import generate_inventory
from src.agentic_bi.database.generate_returns import generate_returns

logger = get_logger(__name__)

SEED_REGIONS_SQL = Path("data/seed_regions.sql")


def run_seed_regions(conn) -> None:
    logger.info("Seeding regions and state_region_map")
    with conn.cursor() as cur, open(SEED_REGIONS_SQL) as f:
        cur.execute(f.read())
    conn.commit()
    logger.info("Seed complete")


def run_full_pipeline() -> None:
    config = get_settings()

    # Step 1 + 2: seed + real tables (run_pipeline opens/closes its own connection)
    conn = get_connection(config)
    try:
        run_seed_regions(conn)
    finally:
        conn.close()

    run_pipeline(config)  # loads customers -> ... -> order_reviews, per-table commits

    # Steps 3-6 share one connection since they read back what was just loaded
    conn = get_connection(config)
    try:
        # Step 3: employees (depends on regions being seeded + sellers being loaded)
        logger.info("Generating and loading employees")
        seller_ids = fetch_seller_ids(conn)
        employees_df = generate_employees(seller_count=len(seller_ids))
        load_employees_df(employees_df, conn)

        # Step 4: assign each seller to a random employee, write back to sellers
        logger.info("Assigning employees to sellers")
        assignments_df = assign_employees_to_sellers(seller_ids, employees_df)
        update_seller_employee_assignments(assignments_df, conn)

        # Step 5: inventory (depends on products being loaded)
        logger.info("Generating and loading inventory")
        product_ids = fetch_product_ids(conn)
        inventory_df = generate_inventory(product_ids)
        load_inventory_df(inventory_df, conn)

        # Step 6: returns (depends on order_items + orders being loaded)
        logger.info("Generating and loading returns")
        order_items_df = fetch_order_items(conn)
        orders_df = fetch_orders(conn)
        returns_df = generate_returns(order_items_df, orders_df)
        load_returns_df(returns_df, conn)

        logger.info("Full pipeline complete")
    except Exception:
        logger.exception("Synthetic generation/loading step failed")
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    run_full_pipeline()
