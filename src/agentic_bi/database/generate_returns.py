"""
Generate synthetic returns for a random subset of real order_items.

Design decisions made:
- ~4% of order_items get a return (plausible e-commerce return rate)
- return_date is a random offset (1-30 days) AFTER the order's purchase date,
  since a return can never happen before the item was ordered
- refund_amount = the item's actual price (full refund assumed for simplicity)
"""

import random
from datetime import timedelta
import pandas as pd
from faker import Faker

fake = Faker()
Faker.seed(42)
random.seed(42)

RETURN_RATE = 0.04
REASONS = ["damaged", "wrong item", "changed mind", "not as described", "defective"]


def generate_returns(order_items_df: pd.DataFrame, orders_df: pd.DataFrame) -> pd.DataFrame:
    """
    order_items_df must have columns: order_id, order_item_id, price
    orders_df must have columns: order_id, order_purchase_timestamp
    """
    purchase_dates = orders_df.set_index("order_id")["order_purchase_timestamp"]

    eligible = order_items_df.sample(frac=RETURN_RATE, random_state=42)

    rows = []
    for i, (_, item) in enumerate(eligible.iterrows(), start=1):
        purchase_date = pd.to_datetime(purchase_dates.get(item["order_id"]))
        if pd.isna(purchase_date):
            continue  # skip if we can't find a valid purchase date

        return_date = purchase_date + timedelta(days=random.randint(1, 30))

        rows.append({
            "return_id": f"RET{i:06d}",
            "order_id": item["order_id"],
            "order_item_id": item["order_item_id"],
            "return_reason": random.choice(REASONS),
            "return_date": return_date.date(),
            "refund_amount": item["price"],
        })

    return pd.DataFrame(rows)
