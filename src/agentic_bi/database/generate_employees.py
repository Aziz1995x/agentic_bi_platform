"""
Generate synthetic employees, then randomly assign each real seller to one.

Design decisions made:
- ~1 employee per 40 sellers (plausible account-manager ratio; adjust as needed)
- employees are NOT restricted to their own region when managing sellers
  (simplification — real account-management assignment logic is out of scope)
- employee_id format: EMP0001, EMP0002, ...
"""

import random
import pandas as pd
from faker import Faker

fake = Faker()
Faker.seed(42)   # reproducible generation across runs/teammates
random.seed(42)

SELLERS_PER_EMPLOYEE = 40
ROLES = ["Account Manager", "Senior Account Manager", "Regional Lead"]
REGION_IDS = [1, 2, 3, 4, 5]  # matches seed_regions.sql


def generate_employees(seller_count: int) -> pd.DataFrame:
    """Generate a DataFrame of synthetic employees sized relative to seller_count."""
    num_employees = max(1, seller_count // SELLERS_PER_EMPLOYEE)

    rows = []
    for i in range(1, num_employees + 1):
        rows.append({
            "employee_id": f"EMP{i:04d}",
            "employee_name": fake.name(),
            "role": random.choice(ROLES),
            "region_id": random.choice(REGION_IDS),
            "hire_date": fake.date_between(start_date="-5y", end_date="today"),
        })

    return pd.DataFrame(rows)


def assign_employees_to_sellers(seller_ids: list[str], employees_df: pd.DataFrame) -> pd.DataFrame:
    """
    Randomly assign each seller_id to one employee_id.
    Returns a DataFrame with columns [seller_id, employee_id] for updating `sellers`.
    """
    employee_ids = employees_df["employee_id"].tolist()
    assignments = [random.choice(employee_ids) for _ in seller_ids]
    return pd.DataFrame({"seller_id": seller_ids, "employee_id": assignments})
