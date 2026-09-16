"""
data_layer.py
─────────────
Single source of truth for all synthetic Olist-style business numbers.
Every chart, table, and PDF in the corpus reads from here so the story
is internally consistent — the revenue numbers in the bar chart match
the revenue numbers in the quarterly report PDF.

Numbers are realistic for the actual Olist dataset (2016-09-04 to
2018-10-17, ~100k orders, ~8M BRL total GMV) but are synthetic
aggregates, not raw CSV exports.
"""

import numpy as np
import pandas as pd

# ── Reproducible seed ────────────────────────────────────────────────────────
RNG = np.random.default_rng(42)

# ── Monthly revenue (BRL) — 2017-01 through 2018-08 ─────────────────────────
# Real Olist shape: slow 2016, ramp through 2017, peak late 2017/early 2018,
# slight softening mid-2018.  Values in thousands BRL.
MONTHS = pd.date_range("2017-01", periods=20, freq="MS")
MONTHLY_REVENUE_K = np.array([
    320, 380, 410, 395, 430, 460,   # Jan–Jun 2017
    490, 540, 570, 610, 680, 720,   # Jul–Dec 2017
    760, 790, 810, 780, 750, 720,   # Jan–Jun 2018
    690, 660,                        # Jul–Aug 2018
], dtype=float)

MONTHLY_ORDERS = np.array([
    3_800, 4_500, 4_900, 4_700, 5_100, 5_400,
    5_800, 6_400, 6_700, 7_200, 8_000, 8_500,
    9_000, 9_300, 9_500, 9_100, 8_800, 8_400,
    8_100, 7_700,
], dtype=float)

MONTHLY_AOV = MONTHLY_REVENUE_K * 1_000 / MONTHLY_ORDERS   # average order value, BRL

# ── Quarterly rollups ─────────────────────────────────────────────────────────
QUARTERS = ["Q1 2017", "Q2 2017", "Q3 2017", "Q4 2017",
            "Q1 2018", "Q2 2018"]
QUARTERLY_REVENUE_K = np.array([
    1_110, 1_285, 1_600, 2_010,   # 2017
    2_360, 2_250,                  # 2018  ← Q2 dip is the "story"
])
QUARTERLY_ORDERS = np.array([
    13_200, 15_200, 18_900, 23_700,
    27_800, 26_300,
])
QUARTERLY_RETURN_RATE = np.array([
    3.1, 3.4, 3.2, 3.8,
    4.1, 5.2,                       # Q2 2018 spike matches revenue dip
])

# ── Product categories (top 10 by revenue) ───────────────────────────────────
CATEGORIES = [
    "Health & Beauty",
    "Watches & Gifts",
    "Bed & Bath",
    "Sports & Leisure",
    "Computers & Accessories",
    "Furniture & Decor",
    "Housewares",
    "Auto",
    "Toys",
    "Cool Stuff",
]
CATEGORY_REVENUE_K = np.array([
    890, 760, 680, 610, 570, 490, 410, 360, 320, 280
], dtype=float)
CATEGORY_ORDERS = np.array([
    9_200, 6_400, 8_100, 7_300, 4_800, 5_100, 5_600, 4_200, 6_800, 3_900
], dtype=float)
CATEGORY_RETURN_RATE = np.array([
    4.1, 6.8, 2.9, 3.2, 5.5, 3.8, 2.7, 4.9, 3.1, 5.2
], dtype=float)
CATEGORY_AOV = CATEGORY_REVENUE_K * 1_000 / CATEGORY_ORDERS

# ── Regional breakdown (5 Brazilian regions, Olist-style) ────────────────────
REGIONS = ["Southeast", "South", "Northeast", "Central-West", "North"]
REGION_REVENUE_K = np.array([4_200, 1_800, 1_200,  640,  360], dtype=float)
REGION_ORDERS = np.array([49_000, 21_000, 14_000, 7_500, 4_200], dtype=float)
REGION_RETURN_RATE = np.array([4.8, 3.9, 4.2, 3.1, 2.8], dtype=float)
REGION_AVG_DELIVERY_DAYS = np.array([7.2, 8.1, 14.6, 11.3, 18.4], dtype=float)

# ── Payment methods ───────────────────────────────────────────────────────────
PAYMENT_METHODS = ["Credit Card", "Boleto", "Voucher", "Debit Card"]
PAYMENT_SHARE = np.array([0.737, 0.191, 0.053, 0.019])   # fraction of orders
PAYMENT_REVENUE_SHARE = np.array([0.782, 0.163, 0.041, 0.014])  # fraction of revenue

# ── Review score distribution ─────────────────────────────────────────────────
REVIEW_SCORES = [1, 2, 3, 4, 5]
REVIEW_COUNTS = np.array([3_200, 2_800, 7_100, 19_400, 65_100])

# ── Customer segments ─────────────────────────────────────────────────────────
SEGMENTS = ["One-time", "Repeat (2–3x)", "Loyal (4+x)"]
SEGMENT_CUSTOMER_COUNT = np.array([72_000, 18_000, 6_800])
SEGMENT_REVENUE_K = np.array([3_240, 2_700, 2_312])

# ── Top sellers (anonymised) ──────────────────────────────────────────────────
SELLER_IDS = [f"Seller_{chr(65+i)}" for i in range(10)]
SELLER_REVENUE_K = np.array([420, 380, 310, 290, 260, 230, 210, 190, 170, 150])
SELLER_ORDERS = np.array([4_800, 4_200, 3_600, 3_200, 2_900, 2_600, 2_400, 2_100, 1_900, 1_700])
SELLER_RETURN_RATE = np.array([3.1, 4.2, 2.8, 5.1, 3.9, 2.4, 6.3, 3.7, 4.8, 2.9])

# ── Delivery performance (monthly avg days, same 20-month window) ─────────────
MONTHLY_AVG_DELIVERY_DAYS = np.array([
    13.2, 12.8, 12.1, 11.9, 11.4, 11.0,
    10.8, 10.5, 10.2,  9.9,  9.6,  9.4,
    9.1,  9.0,  8.8,  9.2, 10.1, 11.3,
    12.0, 12.8,
])

# ── Return reasons (Q2 2018 breakdown) ───────────────────────────────────────
RETURN_REASONS = [
    "Product defect",
    "Wrong item delivered",
    "Item not as described",
    "Delivery damage",
    "Changed mind",
    "Other",
]
RETURN_REASON_COUNTS = np.array([38, 27, 19, 9, 5, 2])  # percentages

# ── KPI snapshot (most recent full quarter: Q2 2018) ─────────────────────────
KPI_Q2_2018 = {
    "total_revenue_brl":        2_250_000,
    "total_orders":             26_300,
    "average_order_value_brl":  85.55,
    "unique_customers":         24_800,
    "repeat_customer_rate_pct": 22.4,
    "return_rate_pct":           5.2,
    "avg_review_score":          4.12,
    "avg_delivery_days":        10.6,
    "active_sellers":           2_840,
    "revenue_vs_q1_pct": -4.7,    # quarter-over-quarter decline
    "orders_vs_q1_pct": -5.4,
}

# ── YoY comparison (H1 2017 vs H1 2018) ──────────────────────────────────────
H1_COMPARISON = {
    "h1_2017_revenue_k": 2_395,
    "h1_2018_revenue_k": 4_610,
    "h1_2017_orders":    29_100,
    "h1_2018_orders":    54_100,
    "yoy_revenue_growth_pct": 92.5,
    "yoy_orders_growth_pct":  85.9,
}
