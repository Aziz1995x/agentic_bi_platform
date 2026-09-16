"""
generate_tables.py
──────────────────
Produces 5 standalone table images (PNG) — the kind a BI team would
screenshot from a dashboard or embed in a report.  Tables are rendered
using matplotlib's table renderer so they look like genuine dashboard
exports, not HTML.

Table inventory
───────────────
T01  kpi_summary_table.png           — Q2 2018 KPI snapshot vs Q1 2018
T02  category_performance_table.png  — all 10 categories, 4 metrics each
T03  regional_performance_table.png  — 5 regions, 5 metrics each
T04  top_sellers_table.png           — top 10 sellers, revenue + returns
T05  quarterly_comparison_table.png  — 6 quarters, revenue/orders/AOV/return rate
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import data_layer as D
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

matplotlib.use("Agg")

OUT = Path("documents/multimodal/tables")
OUT.mkdir(parents=True, exist_ok=True)

BRAND_BLUE = "#1A56DB"
BRAND_GREY = "#F3F4F6"
HEADER_BG = "#1E3A5F"
HEADER_FG = "white"
ALT_ROW = "#EEF2FF"
ALERT_BG = "#FEE2E2"

metadata = {}


def make_table_fig(col_labels, row_labels, cell_data, title,
                   col_widths=None, figsize=None, alert_rows=None,
                   fmt_cols=None):
    """
    Render a styled table as a matplotlib figure.
    alert_rows: set of row indices (0-based) to colour red
    fmt_cols: dict {col_idx: format_string}
    """
    n_rows = len(row_labels)
    n_cols = len(col_labels)
    if figsize is None:
        figsize = (max(8, n_cols * 1.8), max(3, n_rows * 0.52 + 1.2))

    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # title
    ax.set_title(title, fontsize=12, fontweight="bold", color="#1E3A5F",
                 pad=12, loc="left")

    tbl = ax.table(
        cellText=cell_data,
        colLabels=col_labels,
        rowLabels=row_labels,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)

    if col_widths:
        for ci, w in enumerate(col_widths):
            tbl.auto_set_column_width([ci])
            for ri in range(-1, n_rows):
                tbl[ri, ci].set_width(w)

    # style header row
    for ci in range(n_cols):
        cell = tbl[0, ci]
        cell.set_facecolor(HEADER_BG)
        cell.set_text_props(color=HEADER_FG, fontweight="bold")
        cell.set_edgecolor("white")
        cell.set_linewidth(0.5)

    # style row label column
    for ri in range(n_rows):
        cell = tbl[ri + 1, -1]
        cell.set_facecolor("#E8ECEF")
        cell.set_text_props(fontweight="bold", color="#1E3A5F")
        cell.set_edgecolor("white")

    # style data cells
    for ri in range(n_rows):
        bg = ALERT_BG if (alert_rows and ri in alert_rows) else (ALT_ROW if ri % 2 == 0 else "white")
        for ci in range(n_cols):
            cell = tbl[ri + 1, ci]
            cell.set_facecolor(bg)
            cell.set_edgecolor("#D1D5DB")
            cell.set_linewidth(0.3)

    tbl.scale(1, 1.5)
    fig.tight_layout(pad=0.5)
    return fig


def savefig(fig, name):
    path = OUT / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved → {path.name}")


# ── T01  KPI summary ──────────────────────────────────────────────────────────
def table_01():
    kpi = D.KPI_Q2_2018
    rows = [
        ("Total Revenue",         f"R${kpi['total_revenue_brl']:,.0f}",   f"{kpi['revenue_vs_q1_pct']:+.1f}%"),
        ("Total Orders",          f"{kpi['total_orders']:,}",              f"{kpi['orders_vs_q1_pct']:+.1f}%"),
        ("Avg Order Value",       f"R${kpi['average_order_value_brl']:.2f}",  "—"),
        ("Unique Customers",      f"{kpi['unique_customers']:,}",          "—"),
        ("Repeat Customer Rate",  f"{kpi['repeat_customer_rate_pct']:.1f}%", "—"),
        ("Return Rate",           f"{kpi['return_rate_pct']:.1f}%",       "+1.1pp vs Q1"),
        ("Avg Review Score",      f"{kpi['avg_review_score']:.2f} / 5.00", "—"),
        ("Avg Delivery Days",     f"{kpi['avg_delivery_days']:.1f} days", "+1.5d vs Q1"),
        ("Active Sellers",        f"{kpi['active_sellers']:,}",            "—"),
    ]
    col_labels = ["Metric", "Q2 2018 Value", "vs Q1 2018"]
    row_labels = [r[0] for r in rows]
    cell_data = [[r[1], r[2]] for r in rows]

    fig = make_table_fig(
        col_labels=["Q2 2018 Value", "vs Q1 2018"],
        row_labels=row_labels,
        cell_data=cell_data,
        title="Olist Platform — Q2 2018 KPI Snapshot",
        alert_rows={0, 1, 5, 7},   # highlight the declining / at-risk metrics
        figsize=(9, 5.5),
    )
    savefig(fig, "T01_kpi_summary_table")
    metadata["T01_kpi_summary_table.png"] = {
        "title": "Olist Platform — Q2 2018 KPI Snapshot",
        "table_type": "KPI dashboard table",
        "key_insight": (
            "Q2 2018 shows revenue of R$2,250,000 (down 4.7% vs Q1), total orders "
            "26,300 (down 5.4% vs Q1), return rate 5.2% (up 1.1 percentage points), "
            "and average delivery days 10.6 (up 1.5 days vs Q1). "
            "Repeat customer rate is 22.4% and average order value is R$85.55."
        ),
        "data_source": "synthetic — modelled on Olist Brazilian E-Commerce dataset",
        "time_period": "Q2 2018 (April – June 2018)",
    }


# ── T02  Category performance table ──────────────────────────────────────────
def table_02():
    row_labels = D.CATEGORIES
    cell_data = [
        [
            f"R${rev:,.0f}K",
            f"{orders/1_000:.1f}K",
            f"R${aov:.0f}",
            f"{ret:.1f}%",
        ]
        for rev, orders, aov, ret in zip(
            D.CATEGORY_REVENUE_K,
            D.CATEGORY_ORDERS,
            D.CATEGORY_AOV,
            D.CATEGORY_RETURN_RATE,
        )
    ]
    # alert rows: return rate > 5%
    alert_rows = {i for i, r in enumerate(D.CATEGORY_RETURN_RATE) if r > 5.0}

    fig = make_table_fig(
        col_labels=["Revenue", "Orders", "Avg Order Value", "Return Rate"],
        row_labels=row_labels,
        cell_data=cell_data,
        title="Product Category Performance — Olist Platform (Jan 2017 – Aug 2018)",
        alert_rows=alert_rows,
        figsize=(11, 6.5),
    )
    savefig(fig, "T02_category_performance_table")
    metadata["T02_category_performance_table.png"] = {
        "title": "Product Category Performance — Olist Platform (Jan 2017 – Aug 2018)",
        "table_type": "category performance matrix",
        "key_insight": (
            "Health & Beauty leads with R$890K revenue across 9,200 orders. "
            "Watches & Gifts has the highest return rate at 6.8% (highlighted in red). "
            "Computers & Accessories (5.5%) and Cool Stuff (5.2%) also exceed the "
            "5% return rate threshold. Housewares has the highest average order value "
            "per unit relative to its category rank."
        ),
        "data_source": "synthetic — modelled on Olist Brazilian E-Commerce dataset",
        "alert_condition": "Red rows: return rate exceeds 5%",
        "time_period": "January 2017 – August 2018 cumulative",
    }


# ── T03  Regional performance table ──────────────────────────────────────────
def table_03():
    row_labels = list(D.REGIONS)
    cell_data = [
        [
            f"R${rev:,.0f}K",
            f"{orders/1_000:.1f}K",
            f"{ret:.1f}%",
            f"{days:.1f}d",
            f"{rev / D.REGION_REVENUE_K.sum() * 100:.1f}%",
        ]
        for rev, orders, ret, days in zip(
            D.REGION_REVENUE_K,
            D.REGION_ORDERS,
            D.REGION_RETURN_RATE,
            D.REGION_AVG_DELIVERY_DAYS,
        )
    ]
    alert_rows = {i for i, d in enumerate(D.REGION_AVG_DELIVERY_DAYS) if d > 14}

    fig = make_table_fig(
        col_labels=["Revenue", "Orders", "Return Rate", "Avg Delivery", "Revenue Share"],
        row_labels=row_labels,
        cell_data=cell_data,
        title="Regional Performance Summary — Olist Platform (Jan 2017 – Aug 2018)",
        alert_rows=alert_rows,
        figsize=(11, 4.5),
    )
    savefig(fig, "T03_regional_performance_table")
    metadata["T03_regional_performance_table.png"] = {
        "title": "Regional Performance Summary — Olist Platform (Jan 2017 – Aug 2018)",
        "table_type": "regional performance matrix",
        "key_insight": (
            "Southeast dominates with R$4.2M revenue (51.9% share) and 49K orders. "
            "North and Northeast are highlighted in red due to delivery times exceeding "
            "14 days — North at 18.4 days and Northeast at 14.6 days. Lower delivery "
            "performance in these regions correlates with their smaller revenue share. "
            "Central-West has the lowest return rate at 3.1%."
        ),
        "data_source": "synthetic — modelled on Olist Brazilian E-Commerce dataset",
        "alert_condition": "Red rows: average delivery time exceeds 14 days",
        "time_period": "January 2017 – August 2018 cumulative",
    }


# ── T04  Top sellers table ────────────────────────────────────────────────────
def table_04():
    row_labels = D.SELLER_IDS
    cell_data = [
        [
            f"R${rev:,.0f}K",
            f"{orders:,}",
            f"R${rev*1000/orders:.0f}",
            f"{ret:.1f}%",
        ]
        for rev, orders, ret in zip(
            D.SELLER_REVENUE_K,
            D.SELLER_ORDERS,
            D.SELLER_RETURN_RATE,
        )
    ]
    alert_rows = {i for i, r in enumerate(D.SELLER_RETURN_RATE) if r > 5.0}

    fig = make_table_fig(
        col_labels=["Revenue", "Orders", "Avg Order Value", "Return Rate"],
        row_labels=row_labels,
        cell_data=cell_data,
        title="Top 10 Seller Performance — Olist Platform (Jan 2017 – Aug 2018)",
        alert_rows=alert_rows,
        figsize=(10, 6),
    )
    savefig(fig, "T04_top_sellers_table")
    metadata["T04_top_sellers_table.png"] = {
        "title": "Top 10 Seller Performance — Olist Platform (Jan 2017 – Aug 2018)",
        "table_type": "seller leaderboard table",
        "key_insight": (
            "Seller_A leads with R$420K revenue across 4,800 orders. "
            "Seller_G has the highest return rate at 6.3% (highlighted in red) despite "
            "being 7th in revenue rank, suggesting quality or fulfilment issues. "
            "Seller_D (5.1%) also exceeds the 5% threshold. Seller_F at 2.4% return "
            "rate is the most reliable seller by this metric."
        ),
        "data_source": "synthetic — anonymised seller IDs",
        "alert_condition": "Red rows: return rate exceeds 5%",
        "time_period": "January 2017 – August 2018 cumulative",
    }


# ── T05  Quarterly comparison table ──────────────────────────────────────────
def table_05():
    row_labels = D.QUARTERS
    aovs = D.QUARTERLY_REVENUE_K * 1_000 / D.QUARTERLY_ORDERS
    cell_data = [
        [
            f"R${rev:,.0f}K",
            f"{orders/1_000:.1f}K",
            f"R${aov:.0f}",
            f"{ret:.1f}%",
        ]
        for rev, orders, aov, ret in zip(
            D.QUARTERLY_REVENUE_K,
            D.QUARTERLY_ORDERS,
            aovs,
            D.QUARTERLY_RETURN_RATE,
        )
    ]
    # highlight Q2 2018 — the dip quarter
    alert_rows = {5}

    fig = make_table_fig(
        col_labels=["Revenue", "Orders", "Avg Order Value", "Return Rate"],
        row_labels=row_labels,
        cell_data=cell_data,
        title="Quarterly Performance Comparison — Olist Platform (2017–2018)",
        alert_rows=alert_rows,
        figsize=(10, 5),
    )
    savefig(fig, "T05_quarterly_comparison_table")
    metadata["T05_quarterly_comparison_table.png"] = {
        "title": "Quarterly Performance Comparison — Olist Platform (2017–2018)",
        "table_type": "quarterly trend table",
        "key_insight": (
            "Revenue grew from R$1.11M in Q1 2017 to R$2.36M in Q1 2018 — a 113% "
            "increase over four quarters. Q2 2018 (highlighted in red) marks the "
            "first quarter-over-quarter revenue decline: −4.7% to R$2.25M. "
            "Return rate also rose to 5.2% in Q2 2018, the highest in the dataset, "
            "while average order value remained stable at R$85–86 throughout."
        ),
        "data_source": "synthetic — modelled on Olist Brazilian E-Commerce dataset",
        "alert_condition": "Red row: Q2 2018 — first QoQ revenue decline",
        "time_period": "Q1 2017 – Q2 2018",
    }


# ── Run all ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating tables...")
    table_01()
    table_02()
    table_03()
    table_04()
    table_05()

    meta_path = Path("documents/multimodal/metadata/tables_metadata.json")
    import json
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nMetadata written → {meta_path}")
    print(f"Total tables: {len(metadata)}")
