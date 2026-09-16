"""
generate_charts.py
──────────────────
Produces 10 business charts (mix of PNG and JPG) representing the kinds
of visuals a BI team would embed in reports or store in a knowledge base.

Each chart is saved with a companion metadata dict (written to JSON by
the caller) so the ingestion pipeline knows what the chart depicts
without having to re-read the image.

Chart inventory
───────────────
01  monthly_revenue_trend.png       — line chart, 20-month revenue + order overlay
02  quarterly_revenue_bar.png       — grouped bar, revenue + orders by quarter
03  category_revenue_bar.jpg        — horizontal bar, top-10 categories by revenue
04  category_return_rate.png        — bar chart, return rate % by category
05  regional_revenue_pie.png        — pie chart, revenue share by region
06  regional_delivery_heatmap.png   — bar chart styled as heatmap, avg delivery days
07  payment_method_donut.jpg        — donut chart, payment method share
08  review_score_distribution.png   — bar chart, review score histogram
09  customer_segment_revenue.png    — stacked bar, customer segment revenue contribution
10  monthly_aov_delivery.png        — dual-axis line, AOV vs avg delivery days
"""

import numpy as np
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
import matplotlib
import data_layer as D
import sys
import json
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

matplotlib.use("Agg")

OUT = Path("documents/multimodal/charts")
OUT.mkdir(parents=True, exist_ok=True)

# ── Shared style ──────────────────────────────────────────────────────────────
BRAND_BLUE = "#1A56DB"
BRAND_ORANGE = "#E3770A"
BRAND_GREEN = "#057A55"
BRAND_RED = "#E02424"
BRAND_GREY = "#6B7280"
BG = "#F9FAFB"

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor":   BG,
    "axes.spines.top":  False,
    "axes.spines.right": False,
    "font.family":      "DejaVu Sans",
    "font.size":        10,
    "axes.titlesize":   13,
    "axes.titleweight": "bold",
    "axes.labelsize":   10,
})

MONTH_LABELS = [m.strftime("%b %Y") for m in D.MONTHS]
metadata = {}   # filled as we go; written to JSON at end


print(os.getcwd())


# ── Helper ────────────────────────────────────────────────────────────────────
def savefig(fig, name: str, fmt: str = "png", dpi: int = 150):
    path = OUT / f"{name}.{fmt}"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {path.name}")
    return path


# ── 01  Monthly revenue trend ─────────────────────────────────────────────────
def chart_01():
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax2 = ax1.twinx()

    ax1.plot(MONTH_LABELS, D.MONTHLY_REVENUE_K, color=BRAND_BLUE,
             linewidth=2.5, marker="o", markersize=4, label="Revenue (K BRL)")
    ax1.fill_between(MONTH_LABELS, D.MONTHLY_REVENUE_K, alpha=0.12, color=BRAND_BLUE)
    ax2.bar(MONTH_LABELS, D.MONTHLY_ORDERS / 1_000, alpha=0.3,
            color=BRAND_ORANGE, label="Orders (K)")

    ax1.set_ylabel("Revenue (thousands BRL)", color=BRAND_BLUE)
    ax2.set_ylabel("Orders (thousands)", color=BRAND_ORANGE)
    ax1.set_title("Monthly Revenue & Order Volume — Olist Platform (Jan 2017 – Aug 2018)")
    ax1.tick_params(axis="x", rotation=45)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"R${x:,.0f}K"))

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", framealpha=0.8)

    fig.tight_layout()
    savefig(fig, "01_monthly_revenue_trend", "png")
    metadata["01_monthly_revenue_trend.png"] = {
        "title": "Monthly Revenue & Order Volume — Olist Platform (Jan 2017 – Aug 2018)",
        "chart_type": "dual-axis line chart",
        "x_axis": "Month (Jan 2017 – Aug 2018)",
        "y_axis_left": "Revenue in thousands BRL",
        "y_axis_right": "Order count in thousands",
        "key_insight": (
            "Revenue grew consistently from R$320K in Jan 2017 to a peak of R$810K "
            "in Mar 2018, followed by a gradual decline to R$660K by Aug 2018. "
            "Order volume mirrors the revenue trend, peaking at ~9,500 orders in Mar 2018."
        ),
        "data_source": "synthetic — modelled on Olist Brazilian E-Commerce dataset",
        "time_period": "January 2017 – August 2018",
    }


# ── 02  Quarterly revenue bar ─────────────────────────────────────────────────
def chart_02():
    x = np.arange(len(D.QUARTERS))
    w = 0.38
    fig, ax = plt.subplots(figsize=(10, 5))

    bars = ax.bar(x - w/2, D.QUARTERLY_REVENUE_K, w, color=BRAND_BLUE,
                  label="Revenue (K BRL)", zorder=3)
    ax.bar(x + w/2, D.QUARTERLY_ORDERS / 1_000, w, color=BRAND_ORANGE,
           label="Orders (K)", zorder=3)

    # annotate revenue bars
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 15,
                f"R${h:,.0f}K", ha="center", va="bottom", fontsize=8.5)

    ax.set_xticks(x)
    ax.set_xticklabels(D.QUARTERS)
    ax.set_ylabel("Value (thousands)")
    ax.set_title("Quarterly Revenue & Orders — Olist Platform (2017–2018)")
    ax.legend(framealpha=0.8)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}K"))
    ax.grid(axis="y", alpha=0.3, zorder=0)

    # highlight Q2 2018 dip with annotation
    ax.annotate("Q2 2018 dip\n−4.7% QoQ",
                xy=(4.5, D.QUARTERLY_REVENUE_K[4] - 20),
                xytext=(4.1, D.QUARTERLY_REVENUE_K[4] + 180),
                arrowprops=dict(arrowstyle="->", color=BRAND_RED),
                color=BRAND_RED, fontsize=9, fontweight="bold")

    fig.tight_layout()
    savefig(fig, "02_quarterly_revenue_bar", "png")
    metadata["02_quarterly_revenue_bar.png"] = {
        "title": "Quarterly Revenue & Orders — Olist Platform (2017–2018)",
        "chart_type": "grouped bar chart",
        "x_axis": "Quarter",
        "y_axis": "Revenue (K BRL) and Orders (K) — dual bars",
        "key_insight": (
            "Revenue grew from R$1.1M in Q1 2017 to R$2.36M in Q1 2018 (peak), "
            "then declined 4.7% quarter-over-quarter to R$2.25M in Q2 2018. "
            "Order volume followed the same pattern, falling 5.4% QoQ in Q2 2018."
        ),
        "data_source": "synthetic — modelled on Olist Brazilian E-Commerce dataset",
        "time_period": "Q1 2017 – Q2 2018",
        "anomaly": "Q2 2018 revenue and order volume both declined quarter-over-quarter",
    }


# ── 03  Category revenue (horizontal bar) ────────────────────────────────────
def chart_03():
    idx = np.argsort(D.CATEGORY_REVENUE_K)
    cats = [D.CATEGORIES[i] for i in idx]
    revs = D.CATEGORY_REVENUE_K[idx]
    colors = [BRAND_RED if c in ("Computers & Accessories", "Cool Stuff") else BRAND_BLUE
              for c in cats]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(cats, revs, color=colors, zorder=3)
    ax.set_xlabel("Revenue (thousands BRL)")
    ax.set_title("Revenue by Product Category — Top 10 (Cumulative Jan 2017 – Aug 2018)")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"R${x:,.0f}K"))
    ax.grid(axis="x", alpha=0.3, zorder=0)

    for bar, v in zip(bars, revs):
        ax.text(v + 8, bar.get_y() + bar.get_height() / 2,
                f"R${v:,.0f}K", va="center", fontsize=9)

    fig.tight_layout()
    savefig(fig, "03_category_revenue_bar", "jpg")
    metadata["03_category_revenue_bar.jpg"] = {
        "title": "Revenue by Product Category — Top 10 (Jan 2017 – Aug 2018)",
        "chart_type": "horizontal bar chart",
        "x_axis": "Revenue in thousands BRL",
        "y_axis": "Product category",
        "key_insight": (
            "Health & Beauty is the top revenue category at R$890K, followed by "
            "Watches & Gifts (R$760K) and Bed & Bath (R$680K). "
            "Computers & Accessories and Cool Stuff (highlighted in red) have "
            "disproportionately high return rates relative to their revenue rank."
        ),
        "data_source": "synthetic — modelled on Olist Brazilian E-Commerce dataset",
        "time_period": "January 2017 – August 2018 cumulative",
    }


# ── 04  Category return rate bar ──────────────────────────────────────────────
def chart_04():
    idx = np.argsort(D.CATEGORY_RETURN_RATE)[::-1]
    cats = [D.CATEGORIES[i] for i in idx]
    rates = D.CATEGORY_RETURN_RATE[idx]
    colors = [BRAND_RED if r > 5.0 else BRAND_ORANGE if r > 4.0 else BRAND_GREEN
              for r in rates]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(cats, rates, color=colors, zorder=3)
    ax.axhline(np.mean(D.CATEGORY_RETURN_RATE), color=BRAND_GREY, linestyle="--",
               linewidth=1.5, label=f"Platform avg ({np.mean(D.CATEGORY_RETURN_RATE):.1f}%)")
    ax.set_ylabel("Return Rate (%)")
    ax.set_title("Return Rate by Product Category (Jan 2017 – Aug 2018)")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(framealpha=0.8)
    ax.grid(axis="y", alpha=0.3, zorder=0)

    for bar, v in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.08,
                f"{v:.1f}%", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    savefig(fig, "04_category_return_rate", "png")
    metadata["04_category_return_rate.png"] = {
        "title": "Return Rate by Product Category (Jan 2017 – Aug 2018)",
        "chart_type": "bar chart with threshold lines",
        "x_axis": "Product category",
        "y_axis": "Return rate percentage",
        "key_insight": (
            "Watches & Gifts has the highest return rate at 6.8%, well above the "
            "platform average of 4.2%. Computers & Accessories (5.5%) and Cool Stuff "
            "(5.2%) also exceed average. Housewares (2.7%) and Bed & Bath (2.9%) "
            "have the lowest return rates. Red bars exceed 5%, orange exceed 4%."
        ),
        "data_source": "synthetic — modelled on Olist Brazilian E-Commerce dataset",
        "time_period": "January 2017 – August 2018 cumulative",
        "anomaly": "Watches & Gifts return rate 6.8% is 62% above platform average",
    }


# ── 05  Regional revenue pie ──────────────────────────────────────────────────
def chart_05():
    colors = [BRAND_BLUE, BRAND_ORANGE, BRAND_GREEN, BRAND_GREY, "#9061F9"]
    explode = [0.05, 0, 0, 0, 0]

    fig, ax = plt.subplots(figsize=(8, 6))
    wedges, texts, autotexts = ax.pie(
        D.REGION_REVENUE_K, labels=D.REGIONS,
        autopct="%1.1f%%", colors=colors, explode=explode,
        startangle=140, pctdistance=0.82,
        wedgeprops=dict(edgecolor="white", linewidth=1.5)
    )
    for t in autotexts:
        t.set_fontsize(9)
        t.set_fontweight("bold")

    ax.set_title("Revenue Share by Region — Olist Platform\n(Jan 2017 – Aug 2018)")
    total = D.REGION_REVENUE_K.sum()
    ax.text(0, -1.3, f"Total GMV: R${total:,.0f}K  |  {len(D.REGIONS)} regions",
            ha="center", fontsize=9, color=BRAND_GREY)
    fig.tight_layout()
    savefig(fig, "05_regional_revenue_pie", "png")
    metadata["05_regional_revenue_pie.png"] = {
        "title": "Revenue Share by Region — Olist Platform (Jan 2017 – Aug 2018)",
        "chart_type": "pie chart",
        "key_insight": (
            "The Southeast region dominates with 51.9% of total revenue (R$4.2M), "
            "reflecting Brazil's economic concentration in São Paulo and Rio de Janeiro. "
            "South (22.2%) is the second largest. North (4.4%) and Central-West (7.9%) "
            "are the smallest markets. Total platform GMV across all regions is R$8.2M."
        ),
        "data_source": "synthetic — modelled on Olist Brazilian E-Commerce dataset",
        "regions": list(D.REGIONS),
        "time_period": "January 2017 – August 2018 cumulative",
    }


# ── 06  Regional delivery days bar ───────────────────────────────────────────
def chart_06():
    colors = [BRAND_RED if d > 15 else BRAND_ORANGE if d > 10 else BRAND_GREEN
              for d in D.REGION_AVG_DELIVERY_DAYS]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(D.REGIONS, D.REGION_AVG_DELIVERY_DAYS, color=colors, zorder=3)
    ax.axhline(D.REGION_AVG_DELIVERY_DAYS.mean(), color=BRAND_GREY, linestyle="--",
               linewidth=1.5,
               label=f"Platform avg ({D.REGION_AVG_DELIVERY_DAYS.mean():.1f} days)")
    ax.set_ylabel("Average Delivery Days")
    ax.set_title("Average Delivery Time by Region — Olist Platform")
    ax.legend(framealpha=0.8)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.tick_params(axis="x", rotation=15)

    for bar, v in zip(bars, D.REGION_AVG_DELIVERY_DAYS):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.2,
                f"{v:.1f}d", ha="center", va="bottom", fontsize=9.5)

    fig.tight_layout()
    savefig(fig, "06_regional_delivery_days", "png")
    metadata["06_regional_delivery_days.png"] = {
        "title": "Average Delivery Time by Region — Olist Platform",
        "chart_type": "bar chart",
        "x_axis": "Brazilian region",
        "y_axis": "Average delivery days",
        "key_insight": (
            "The North region has the longest average delivery time at 18.4 days, "
            "more than double the Southeast's 7.2 days. Northeast at 14.6 days also "
            "exceeds the platform average of 11.9 days. Delivery time correlates "
            "negatively with revenue — longer delivery regions have lower customer "
            "conversion and repeat purchase rates."
        ),
        "data_source": "synthetic — modelled on Olist Brazilian E-Commerce dataset",
    }


# ── 07  Payment method donut ──────────────────────────────────────────────────
def chart_07():
    colors = [BRAND_BLUE, BRAND_ORANGE, BRAND_GREEN, BRAND_GREY]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for ax, shares, title, suffix in [
        (ax1, D.PAYMENT_SHARE,         "Orders by Payment Method",  "orders"),
        (ax2, D.PAYMENT_REVENUE_SHARE, "Revenue by Payment Method", "revenue"),
    ]:
        wedges, texts, autotexts = ax.pie(
            shares, labels=D.PAYMENT_METHODS,
            autopct="%1.1f%%", colors=colors, startangle=90,
            pctdistance=0.78,
            wedgeprops=dict(width=0.55, edgecolor="white", linewidth=1.5),
        )
        for t in autotexts:
            t.set_fontsize(9)
            t.set_fontweight("bold")
        ax.set_title(title)

    fig.suptitle("Payment Method Mix — Olist Platform (Jan 2017 – Aug 2018)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    savefig(fig, "07_payment_method_donut", "jpg")
    metadata["07_payment_method_donut.jpg"] = {
        "title": "Payment Method Mix — Olist Platform (Jan 2017 – Aug 2018)",
        "chart_type": "dual donut chart",
        "key_insight": (
            "Credit card dominates both order share (73.7%) and revenue share (78.2%), "
            "suggesting credit card customers place higher-value orders on average. "
            "Boleto (Brazilian bank slip) accounts for 19.1% of orders but only 16.3% "
            "of revenue, consistent with its use for smaller-ticket purchases. "
            "Vouchers and debit cards together account for under 7% of both metrics."
        ),
        "data_source": "synthetic — modelled on Olist Brazilian E-Commerce dataset",
        "time_period": "January 2017 – August 2018 cumulative",
    }


# ── 08  Review score distribution ────────────────────────────────────────────
def chart_08():
    colors = [BRAND_RED, BRAND_ORANGE, BRAND_ORANGE, BRAND_GREEN, BRAND_BLUE]
    total = D.REVIEW_COUNTS.sum()
    pcts = D.REVIEW_COUNTS / total * 100

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar([str(s) for s in D.REVIEW_SCORES], D.REVIEW_COUNTS,
                  color=colors, zorder=3)
    ax2 = ax.twinx()
    ax2.plot([str(s) for s in D.REVIEW_SCORES], pcts,
             color=BRAND_GREY, marker="o", linewidth=2, linestyle="--", label="% share")
    ax2.set_ylabel("Percentage of total reviews")
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))

    ax.set_xlabel("Review Score")
    ax.set_ylabel("Number of Reviews")
    ax.set_title("Customer Review Score Distribution — Olist Platform")
    ax.grid(axis="y", alpha=0.3, zorder=0)

    avg = sum(s * c for s, c in zip(D.REVIEW_SCORES, D.REVIEW_COUNTS)) / total
    ax.axvline(ax.get_xlim()[0], alpha=0)  # dummy for spacing
    fig.text(0.15, 0.82, f"Mean score: {avg:.2f} / 5.00",
             fontsize=10, color=BRAND_BLUE, fontweight="bold")

    for bar, v in zip(bars, D.REVIEW_COUNTS):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 200,
                f"{v:,}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    savefig(fig, "08_review_score_distribution", "png")
    metadata["08_review_score_distribution.png"] = {
        "title": "Customer Review Score Distribution — Olist Platform",
        "chart_type": "bar chart with line overlay",
        "x_axis": "Review score (1–5)",
        "y_axis": "Number of reviews",
        "key_insight": (
            "67% of reviews are 5-star and 20% are 4-star, giving a mean score of 4.12. "
            "Only 3.3% of reviews are 1-star and 2.9% are 2-star, indicating generally "
            "high customer satisfaction. The distribution is strongly left-skewed, "
            "typical for e-commerce platforms where dissatisfied customers may not "
            "leave reviews at all."
        ),
        "data_source": "synthetic — modelled on Olist Brazilian E-Commerce dataset",
        "total_reviews": int(total),
        "mean_score": round(avg, 2),
    }


# ── 09  Customer segment revenue ─────────────────────────────────────────────
def chart_09():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    colors = [BRAND_BLUE, BRAND_ORANGE, BRAND_GREEN]

    # left: customer count
    ax1.bar(D.SEGMENTS, D.SEGMENT_CUSTOMER_COUNT / 1_000, color=colors, zorder=3)
    ax1.set_ylabel("Customers (thousands)")
    ax1.set_title("Customer Count by Segment")
    ax1.grid(axis="y", alpha=0.3, zorder=0)
    for i, v in enumerate(D.SEGMENT_CUSTOMER_COUNT):
        ax1.text(i, v / 1_000 + 0.3, f"{v/1_000:.0f}K", ha="center", fontsize=9)

    # right: revenue
    ax2.bar(D.SEGMENTS, D.SEGMENT_REVENUE_K, color=colors, zorder=3)
    ax2.set_ylabel("Revenue (thousands BRL)")
    ax2.set_title("Revenue Contribution by Segment")
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"R${x:,.0f}K"))
    ax2.grid(axis="y", alpha=0.3, zorder=0)
    for i, v in enumerate(D.SEGMENT_REVENUE_K):
        ax2.text(i, v + 20, f"R${v:,.0f}K", ha="center", fontsize=9)

    fig.suptitle("Customer Segment Analysis — Olist Platform (Jan 2017 – Aug 2018)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    savefig(fig, "09_customer_segment_revenue", "png")
    metadata["09_customer_segment_revenue.png"] = {
        "title": "Customer Segment Analysis — Olist Platform (Jan 2017 – Aug 2018)",
        "chart_type": "side-by-side bar charts",
        "key_insight": (
            "While one-time buyers make up 74% of the customer base (72K customers), "
            "repeat buyers (2–3x) and loyal customers (4+x) together contribute 61% "
            "of total revenue despite being only 26% of customers. Loyal customers "
            "(6,800 customers) generate R$2.3M — nearly equal to the R$2.7M from "
            "18,000 repeat buyers — demonstrating strong revenue concentration in "
            "the top customer tier."
        ),
        "data_source": "synthetic — modelled on Olist Brazilian E-Commerce dataset",
        "time_period": "January 2017 – August 2018 cumulative",
    }


# ── 10  AOV vs delivery days dual-axis ───────────────────────────────────────
def chart_10():
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax2 = ax1.twinx()

    ax1.plot(MONTH_LABELS, D.MONTHLY_AOV, color=BRAND_BLUE, linewidth=2.5,
             marker="o", markersize=4, label="Avg Order Value (BRL)")
    ax2.plot(MONTH_LABELS, D.MONTHLY_AVG_DELIVERY_DAYS, color=BRAND_RED,
             linewidth=2.5, marker="s", markersize=4, linestyle="--",
             label="Avg Delivery Days")

    ax1.set_ylabel("Average Order Value (BRL)", color=BRAND_BLUE)
    ax2.set_ylabel("Average Delivery Days", color=BRAND_RED)
    ax1.set_title("Average Order Value vs Delivery Performance\n(Jan 2017 – Aug 2018)")
    ax1.tick_params(axis="x", rotation=45)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"R${x:.0f}"))

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", framealpha=0.8)

    # annotate the AOV dip
    min_idx = int(np.argmin(D.MONTHLY_AOV))
    ax1.annotate(f"AOV low\nR${D.MONTHLY_AOV[min_idx]:.0f}",
                 xy=(MONTH_LABELS[min_idx], D.MONTHLY_AOV[min_idx]),
                 xytext=(MONTH_LABELS[min_idx], D.MONTHLY_AOV[min_idx] + 8),
                 arrowprops=dict(arrowstyle="->", color=BRAND_BLUE),
                 color=BRAND_BLUE, fontsize=8.5)

    fig.tight_layout()
    savefig(fig, "10_aov_vs_delivery", "png")
    metadata["10_aov_vs_delivery.png"] = {
        "title": "Average Order Value vs Delivery Performance (Jan 2017 – Aug 2018)",
        "chart_type": "dual-axis line chart",
        "x_axis": "Month",
        "y_axis_left": "Average order value in BRL",
        "y_axis_right": "Average delivery days",
        "key_insight": (
            "Average order value grew from R$84 in Jan 2017 to ~R$85 by mid-2018 "
            "with low volatility. Delivery performance improved steadily from 13.2 days "
            "to 8.8 days (Mar 2018) before deteriorating to 12.8 days by Aug 2018, "
            "coinciding with the Q2 2018 revenue decline. The negative correlation "
            "between delivery days and order volume suggests delivery speed is a "
            "meaningful demand driver on the Olist platform."
        ),
        "data_source": "synthetic — modelled on Olist Brazilian E-Commerce dataset",
        "time_period": "January 2017 – August 2018",
    }


# ── Run all ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating charts...")
    chart_01()
    chart_02()
    chart_03()
    chart_04()
    chart_05()
    chart_06()
    chart_07()
    chart_08()
    chart_09()
    chart_10()

    meta_path = Path("documents/multimodal/metadata/charts_metadata.json")
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    import json
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nMetadata written → {meta_path}")
    print(f"Total charts: {len(metadata)}")
