"""
Generate synthetic inventory: one row per real product_id.

Design decisions made:
- stock_quantity: random int, weighted toward low-to-moderate stock
  (0-500 range; occasional stockouts at 0 are intentional/realistic)
- restock_date: random recent date within the last 90 days
"""

import random
import pandas as pd
from faker import Faker

fake = Faker()
Faker.seed(42)
random.seed(42)


def generate_inventory(product_ids: list[str]) -> pd.DataFrame:
    rows = []
    for pid in product_ids:
        rows.append({
            "product_id": pid,
            "stock_quantity": random.randint(0, 500),
            "restock_date": fake.date_between(start_date="-90d", end_date="today"),
        })
    return pd.DataFrame(rows)
