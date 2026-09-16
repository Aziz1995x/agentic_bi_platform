"""
generate_pdfs.py
────────────────
Produces 3 mixed-content business report PDFs using reportlab.
Each PDF contains: cover page, text sections, embedded chart images,
embedded table images, and a summary section — exactly what a real BI
team would produce and store in a knowledge base.

PDF inventory
─────────────
P01  q2_2018_business_report.pdf     — quarterly performance report (5 pages)
P02  revenue_decline_analysis.pdf    — root cause analysis of Q2 decline (4 pages)
P03  product_category_deep_dive.pdf  — category & returns analysis (4 pages)
"""

import datetime
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
    PageBreak, HRFlowable, KeepTogether,
)
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
import data_layer as D
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


OUT = Path("/documents/multimodal/pdfs")
CHARTS = Path("/documents/multimodal/charts")
TABLES_D = Path("/documents/multimodal/tables")
OUT.mkdir(parents=True, exist_ok=True)

BRAND_BLUE = colors.HexColor("#1A56DB")
BRAND_DARK = colors.HexColor("#1E3A5F")
BRAND_RED = colors.HexColor("#E02424")
BRAND_GREY = colors.HexColor("#6B7280")
LIGHT_GREY = colors.HexColor("#F3F4F6")
ALERT_RED = colors.HexColor("#FEE2E2")

PAGE_W, PAGE_H = A4
LEFT_MARGIN = RIGHT_MARGIN = 2.2 * cm
TOP_MARGIN = BOTTOM_MARGIN = 2.2 * cm

metadata = {}


def get_styles():
    styles = getSampleStyleSheet()
    custom = {
        "ReportTitle": ParagraphStyle(
            "ReportTitle", parent=styles["Title"],
            fontSize=24, textColor=BRAND_DARK, spaceAfter=8,
            alignment=TA_CENTER, leading=30,
        ),
        "ReportSubtitle": ParagraphStyle(
            "ReportSubtitle", parent=styles["Normal"],
            fontSize=13, textColor=BRAND_GREY, spaceAfter=6,
            alignment=TA_CENTER,
        ),
        "SectionHeader": ParagraphStyle(
            "SectionHeader", parent=styles["Heading1"],
            fontSize=14, textColor=BRAND_DARK, spaceBefore=18, spaceAfter=6,
            borderPad=4,
        ),
        "SubHeader": ParagraphStyle(
            "SubHeader", parent=styles["Heading2"],
            fontSize=11, textColor=BRAND_BLUE, spaceBefore=10, spaceAfter=4,
        ),
        "BodyText": ParagraphStyle(
            "BodyText", parent=styles["Normal"],
            fontSize=9.5, leading=14, spaceAfter=6, alignment=TA_JUSTIFY,
        ),
        "BulletText": ParagraphStyle(
            "BulletText", parent=styles["Normal"],
            fontSize=9.5, leading=14, spaceAfter=3,
            leftIndent=14, bulletIndent=4,
        ),
        "Caption": ParagraphStyle(
            "Caption", parent=styles["Normal"],
            fontSize=8.5, textColor=BRAND_GREY, alignment=TA_CENTER,
            spaceAfter=10, spaceBefore=4, fontName="Helvetica-Oblique",
        ),
        "KPILabel": ParagraphStyle(
            "KPILabel", parent=styles["Normal"],
            fontSize=8, textColor=BRAND_GREY, alignment=TA_CENTER,
        ),
        "KPIValue": ParagraphStyle(
            "KPIValue", parent=styles["Normal"],
            fontSize=16, textColor=BRAND_DARK, alignment=TA_CENTER,
            fontName="Helvetica-Bold",
        ),
        "AlertText": ParagraphStyle(
            "AlertText", parent=styles["Normal"],
            fontSize=9.5, leading=14, textColor=colors.HexColor("#991B1B"),
            leftIndent=10, spaceAfter=4,
        ),
        "FooterText": ParagraphStyle(
            "FooterText", parent=styles["Normal"],
            fontSize=7.5, textColor=BRAND_GREY, alignment=TA_CENTER,
        ),
    }
    return custom


def chart_img(filename, width_cm=14):
    path = CHARTS / filename
    if not path.exists():
        return Spacer(1, 0.5 * cm)
    return Image(str(path), width=width_cm * cm,
                 height=width_cm * cm * 0.45, kind="proportional")


def table_img(filename, width_cm=14):
    path = TABLES_D / filename
    if not path.exists():
        return Spacer(1, 0.5 * cm)
    return Image(str(path), width=width_cm * cm,
                 height=width_cm * cm * 0.55, kind="proportional")


def kpi_row(labels, values, styles):
    """Render a row of KPI boxes as a reportlab Table."""
    n = len(labels)
    w = (PAGE_W - LEFT_MARGIN - RIGHT_MARGIN) / n
    data = [
        [Paragraph(v, styles["KPIValue"]) for v in values],
        [Paragraph(l, styles["KPILabel"]) for l in labels],
    ]
    tbl = Table(data, colWidths=[w] * n)
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), LIGHT_GREY),
        ("BOX",        (0, 0), (-1, -1), 0.5, colors.white),
        ("LINEAFTER",  (0, 0), (-2, -1), 0.5, colors.white),
        ("TOPPADDING",    (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    return tbl


def hr(styles):
    return HRFlowable(width="100%", thickness=0.5, color=BRAND_GREY, spaceAfter=6)


# ── P01  Q2 2018 Business Report ─────────────────────────────────────────────
def pdf_01():
    path = OUT / "P01_q2_2018_business_report.pdf"
    doc = SimpleDocTemplate(str(path), pagesize=A4,
                            leftMargin=LEFT_MARGIN, rightMargin=RIGHT_MARGIN,
                            topMargin=TOP_MARGIN, bottomMargin=BOTTOM_MARGIN)
    s = get_styles()
    kpi = D.KPI_Q2_2018
    story = []

    # ── Cover ──────────────────────────────────────────────────────────────
    story += [
        Spacer(1, 2 * cm),
        Paragraph("Olist E-Commerce Platform", s["ReportSubtitle"]),
        Paragraph("Q2 2018 Business Performance Report", s["ReportTitle"]),
        Spacer(1, 0.4 * cm),
        Paragraph("April – June 2018  |  Confidential  |  Prepared by: BI Analytics Team",
                  s["ReportSubtitle"]),
        Spacer(1, 1.5 * cm),
        hr(s),
        Spacer(1, 0.8 * cm),
    ]

    # KPI band
    story += [
        kpi_row(
            ["Total Revenue", "Total Orders", "Avg Order Value", "Return Rate"],
            ["R$2,250,000", "26,300", "R$85.55", "5.2%"],
            s,
        ),
        Spacer(1, 0.5 * cm),
        kpi_row(
            ["Unique Customers", "Repeat Rate", "Avg Review Score", "Avg Delivery"],
            ["24,800", "22.4%", "4.12 / 5.00", "10.6 days"],
            s,
        ),
        Spacer(1, 1 * cm),
        PageBreak(),
    ]

    # ── Section 1: Executive Summary ──────────────────────────────────────
    story += [
        Paragraph("1. Executive Summary", s["SectionHeader"]),
        hr(s),
        Paragraph(
            "Q2 2018 marks the first quarter-over-quarter revenue decline since "
            "platform launch, with total GMV falling 4.7% to R$2,250,000 compared "
            "to R$2,360,000 in Q1 2018. Order volume contracted 5.4% to 26,300 orders. "
            "Despite these top-line pressures, year-over-year performance remains "
            "strongly positive — H1 2018 revenue of R$4,610,000 represents 92.5% "
            "growth versus H1 2017's R$2,395,000.",
            s["BodyText"],
        ),
        Paragraph(
            "Three factors are identified as primary contributors to the QoQ decline: "
            "(1) deterioration in average delivery time from 9.1 days (Q1 2018) to "
            "10.6 days, concentrated in the Southeast and Northeast regions; "
            "(2) a return rate increase of 1.1 percentage points to 5.2%, driven "
            "primarily by the Watches & Gifts and Computers & Accessories categories; "
            "and (3) a modest softening in new customer acquisition relative to Q1.",
            s["BodyText"],
        ),
        Spacer(1, 0.3 * cm),
        chart_img("02_quarterly_revenue_bar.png", width_cm=15),
        Paragraph(
            "Figure 1: Quarterly revenue and order volume, Q1 2017 – Q2 2018. "
            "Q2 2018 decline is annotated.",
            s["Caption"],
        ),
    ]

    # ── Section 2: Revenue Trend ──────────────────────────────────────────
    story += [
        Paragraph("2. Revenue Trend Analysis", s["SectionHeader"]),
        hr(s),
        Paragraph(
            "Monthly revenue peaked at R$810,000 in March 2018 before declining "
            "for five consecutive months. The decline accelerated through June 2018 "
            "(R$720,000), which ended Q2. The pattern is consistent with a demand "
            "response to delivery quality — the months with the steepest revenue "
            "falls coincide precisely with the months where average delivery time "
            "began its upward trend.",
            s["BodyText"],
        ),
        chart_img("01_monthly_revenue_trend.png", width_cm=15),
        Paragraph(
            "Figure 2: Monthly revenue (BRL, left axis) and order volume (right axis), "
            "January 2017 – August 2018.",
            s["Caption"],
        ),
        chart_img("10_aov_vs_delivery.png", width_cm=15),
        Paragraph(
            "Figure 3: Average order value vs. average delivery days. "
            "Delivery time deterioration from March 2018 onward correlates with "
            "the order volume decline.",
            s["Caption"],
        ),
        PageBreak(),
    ]

    # ── Section 3: Category & Returns ─────────────────────────────────────
    story += [
        Paragraph("3. Category Performance & Returns", s["SectionHeader"]),
        hr(s),
        Paragraph(
            "Return rate increased to 5.2% in Q2 2018 — the highest quarterly figure "
            "in the dataset. Category-level analysis identifies Watches & Gifts (6.8% "
            "cumulative return rate) and Computers & Accessories (5.5%) as the primary "
            "contributors. These two categories account for an estimated 38% of all "
            "return events despite representing only 16% of total revenue.",
            s["BodyText"],
        ),
        chart_img("04_category_return_rate.png", width_cm=15),
        Paragraph(
            "Figure 4: Return rate by product category. Red bars exceed 5% "
            "(platform Q2 2018 rate). Dashed line shows platform average (4.2%).",
            s["Caption"],
        ),
        Spacer(1, 0.3 * cm),
        table_img("T02_category_performance_table.png", width_cm=15),
        Paragraph(
            "Table 1: Full category performance matrix. Red rows indicate "
            "return rates exceeding 5%.",
            s["Caption"],
        ),
        PageBreak(),
    ]

    # ── Section 4: Regional Analysis ──────────────────────────────────────
    story += [
        Paragraph("4. Regional Analysis", s["SectionHeader"]),
        hr(s),
        Paragraph(
            "The Southeast region continues to account for 51.9% of platform revenue "
            "(R$4.2M cumulative). However, delivery performance degradation was most "
            "pronounced in this region during Q2 2018, which likely amplified the "
            "revenue impact. The North region's 18.4-day average delivery time remains "
            "a persistent structural constraint on market penetration.",
            s["BodyText"],
        ),
        chart_img("05_regional_revenue_pie.png", width_cm=12),
        Paragraph("Figure 5: Revenue share by region (cumulative Jan 2017 – Aug 2018).",
                  s["Caption"]),
        chart_img("06_regional_delivery_days.png", width_cm=15),
        Paragraph(
            "Figure 6: Average delivery days by region. Red bars exceed 15 days; "
            "orange bars exceed the 11.9-day platform average.",
            s["Caption"],
        ),
    ]

    # ── Section 5: Recommendations ────────────────────────────────────────
    story += [
        Paragraph("5. Recommendations", s["SectionHeader"]),
        hr(s),
        Paragraph("Based on the Q2 2018 analysis, the following actions are recommended:",
                  s["BodyText"]),
        Paragraph("1. Delivery infrastructure: Prioritise last-mile logistics "
                  "improvements in the Southeast and Northeast to reverse the delivery "
                  "time increase observed from March 2018 onward.", s["BulletText"]),
        Paragraph("2. Returns programme: Implement enhanced quality checks for "
                  "Watches & Gifts and Computers & Accessories sellers. Consider "
                  "requiring seller-level return rate SLAs (target: below 4%).",
                  s["BulletText"]),
        Paragraph("3. Customer retention: The 22.4% repeat customer rate, while "
                  "stable, has room to improve. A targeted loyalty programme for "
                  "the Repeat segment could accelerate migration to the Loyal tier.",
                  s["BulletText"]),
        Paragraph("4. Monitoring cadence: Establish a weekly delivery quality "
                  "dashboard to enable earlier detection of logistics deterioration.",
                  s["BulletText"]),
        Spacer(1, 1 * cm),
        Paragraph(
            "Olist BI Analytics Team  |  Report generated: July 2018  |  "
            "Data: synthetic, modelled on Olist Brazilian E-Commerce dataset (CC BY-NC-SA)",
            s["FooterText"],
        ),
    ]

    doc.build(story)
    print(f"  saved → {path.name}")
    metadata["P01_q2_2018_business_report.pdf"] = {
        "title": "Olist E-Commerce Platform — Q2 2018 Business Performance Report",
        "document_type": "quarterly business report",
        "pages": 5,
        "content_types": ["text", "charts", "tables", "KPI summary"],
        "key_topics": [
            "Q2 2018 revenue decline of 4.7% QoQ",
            "delivery time deterioration as primary cause",
            "return rate increase to 5.2%",
            "regional performance",
            "category return rates",
            "recommendations",
        ],
        "embedded_charts": [
            "02_quarterly_revenue_bar.png",
            "01_monthly_revenue_trend.png",
            "10_aov_vs_delivery.png",
            "04_category_return_rate.png",
            "05_regional_revenue_pie.png",
            "06_regional_delivery_days.png",
        ],
        "embedded_tables": ["T02_category_performance_table.png"],
        "data_source": "synthetic — modelled on Olist Brazilian E-Commerce dataset",
        "time_period": "Q2 2018 (April – June 2018)",
    }


# ── P02  Revenue Decline Root Cause Analysis ──────────────────────────────────
def pdf_02():
    path = OUT / "P02_revenue_decline_analysis.pdf"
    doc = SimpleDocTemplate(str(path), pagesize=A4,
                            leftMargin=LEFT_MARGIN, rightMargin=RIGHT_MARGIN,
                            topMargin=TOP_MARGIN, bottomMargin=BOTTOM_MARGIN)
    s = get_styles()
    story = []

    story += [
        Spacer(1, 1.5 * cm),
        Paragraph("Olist E-Commerce Platform", s["ReportSubtitle"]),
        Paragraph("Revenue Decline Root Cause Analysis", s["ReportTitle"]),
        Paragraph("Q2 2018  |  Internal Investigation Report", s["ReportSubtitle"]),
        Spacer(1, 1 * cm),
        hr(s),
        Spacer(1, 0.5 * cm),
        Paragraph("1. Investigation Scope", s["SectionHeader"]),
        Paragraph(
            "This document presents the findings of a structured root cause analysis "
            "into the Q2 2018 quarter-over-quarter revenue decline. Revenue fell from "
            "R$2,360,000 (Q1 2018) to R$2,250,000 (Q2 2018), a reduction of R$110,000 "
            "or 4.7%. This is the first QoQ decline since platform launch in 2016.",
            s["BodyText"],
        ),
        Paragraph(
            "The investigation examined three hypotheses: (H1) demand-side factors "
            "(macro seasonality, customer acquisition slowdown); (H2) supply-side "
            "factors (seller quality, return rates, stockouts); and (H3) operational "
            "factors (delivery performance, logistics capacity).",
            s["BodyText"],
        ),
        PageBreak(),

        Paragraph("2. Hypothesis H3: Delivery Performance (Primary)", s["SectionHeader"]),
        hr(s),
        Paragraph(
            "Delivery performance analysis provides the strongest explanatory signal. "
            "Average delivery time increased from 9.1 days (Q1 2018) to 10.6 days "
            "(Q2 2018) — a 16.5% deterioration. Historical analysis shows a negative "
            "correlation between delivery days and order volume across all 20 months "
            "in the dataset (Pearson r = -0.71).",
            s["BodyText"],
        ),
        Paragraph(
            "The deterioration was concentrated in two regions:",
            s["BodyText"],
        ),
        Paragraph("Southeast: from 6.8 days (Q1) to 8.1 days (Q2) — "
                  "a 19% increase in the highest-revenue region.", s["BulletText"]),
        Paragraph("Northeast: from 13.2 days (Q1) to 15.9 days (Q2) — "
                  "a 20% increase, compounding an already poor baseline.", s["BulletText"]),
        Spacer(1, 0.3 * cm),
        chart_img("10_aov_vs_delivery.png", width_cm=15),
        Paragraph(
            "Figure 1: AOV vs. delivery days (monthly, Jan 2017 – Aug 2018). "
            "Delivery time increase from March 2018 precedes and tracks "
            "the order volume decline.",
            s["Caption"],
        ),

        Paragraph("3. Hypothesis H2: Supply-Side / Returns (Contributing)", s["SectionHeader"]),
        hr(s),
        Paragraph(
            "Return rate increased 1.1 percentage points to 5.2% in Q2 2018. "
            "Each return event represents both a lost GMV unit and a negative "
            "customer experience signal. At 26,300 orders and 5.2% return rate, "
            "approximately 1,368 orders were returned in Q2 2018, compared to "
            "approximately 1,141 in Q1 2018 (+20% increase in return events).",
            s["BodyText"],
        ),

        # Return reasons table (manual data table using reportlab Table)
        Spacer(1, 0.2 * cm),
        Table(
            [["Return Reason", "Share (%)"]] +
            [[r, f"{c}%"] for r, c in zip(D.RETURN_REASONS, D.RETURN_REASON_COUNTS)],
            colWidths=[10 * cm, 4 * cm],
            style=TableStyle([
                ("BACKGROUND",  (0, 0), (-1, 0),   BRAND_DARK),
                ("TEXTCOLOR",   (0, 0), (-1, 0),   colors.white),
                ("FONTNAME",    (0, 0), (-1, 0),   "Helvetica-Bold"),
                ("FONTSIZE",    (0, 0), (-1, -1),  9),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [LIGHT_GREY, colors.white]),
                ("GRID",        (0, 0), (-1, -1),  0.3, colors.HexColor("#D1D5DB")),
                ("ALIGN",       (1, 0), (1, -1),   "CENTER"),
                ("TOPPADDING",  (0, 0), (-1, -1),  5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]),
        ),
        Paragraph(
            "Table 1: Q2 2018 return reason breakdown. Product defect (38%) and "
            "wrong item delivered (27%) account for 65% of returns — both addressable "
            "through seller quality controls.",
            s["Caption"],
        ),
        PageBreak(),

        Paragraph("4. Hypothesis H1: Demand-Side (Weak, Not Primary)", s["SectionHeader"]),
        hr(s),
        Paragraph(
            "Macro seasonality analysis does not support H1 as a primary cause. "
            "Q2 is not historically a weak quarter on the Olist platform — Q2 2017 "
            "showed quarter-over-quarter growth of +15.7%. New customer acquisition "
            "did slow modestly (estimated -8% vs Q1) but is insufficient to explain "
            "the magnitude of the revenue decline on its own.",
            s["BodyText"],
        ),
        chart_img("09_customer_segment_revenue.png", width_cm=15),
        Paragraph(
            "Figure 2: Customer segment analysis. Loyal customer revenue is "
            "disproportionately valuable — a decline in repeat rates amplifies "
            "revenue sensitivity.",
            s["Caption"],
        ),

        Paragraph("5. Findings Summary", s["SectionHeader"]),
        hr(s),
        Table(
            [
                ["Factor", "Contribution", "Evidence Strength"],
                ["Delivery deterioration (H3)", "Primary", "Strong (r = -0.71)"],
                ["Return rate increase (H2)",   "Contributing", "Moderate"],
                ["Demand/seasonality (H1)",     "Minor",   "Weak"],
            ],
            colWidths=[7 * cm, 4 * cm, 4 * cm],
            style=TableStyle([
                ("BACKGROUND",  (0, 0), (-1, 0),   BRAND_DARK),
                ("TEXTCOLOR",   (0, 0), (-1, 0),   colors.white),
                ("FONTNAME",    (0, 0), (-1, 0),   "Helvetica-Bold"),
                ("FONTSIZE",    (0, 0), (-1, -1),  9),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [LIGHT_GREY, colors.white]),
                ("GRID",        (0, 0), (-1, -1),  0.3, colors.HexColor("#D1D5DB")),
                ("TOPPADDING",  (0, 0), (-1, -1),  5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]),
        ),
        Spacer(1, 1 * cm),
        Paragraph(
            "Olist BI Analytics Team  |  Internal use only  |  "
            "Data: synthetic, modelled on Olist Brazilian E-Commerce dataset",
            s["FooterText"],
        ),
    ]

    doc.build(story)
    print(f"  saved → {path.name}")
    metadata["P02_revenue_decline_analysis.pdf"] = {
        "title": "Revenue Decline Root Cause Analysis — Q2 2018",
        "document_type": "root cause analysis report",
        "pages": 4,
        "content_types": ["text", "charts", "data tables"],
        "key_topics": [
            "Q2 2018 revenue decline investigation",
            "delivery performance as primary cause",
            "Pearson correlation between delivery days and orders",
            "return rate increase contributing factor",
            "demand-side seasonality ruled out as primary cause",
            "return reason breakdown",
        ],
        "embedded_charts": [
            "10_aov_vs_delivery.png",
            "09_customer_segment_revenue.png",
        ],
        "data_source": "synthetic — modelled on Olist Brazilian E-Commerce dataset",
        "time_period": "Q2 2018 investigation; data Jan 2017 – Aug 2018",
    }


# ── P03  Product Category Deep Dive ──────────────────────────────────────────
def pdf_03():
    path = OUT / "P03_product_category_deep_dive.pdf"
    doc = SimpleDocTemplate(str(path), pagesize=A4,
                            leftMargin=LEFT_MARGIN, rightMargin=RIGHT_MARGIN,
                            topMargin=TOP_MARGIN, bottomMargin=BOTTOM_MARGIN)
    s = get_styles()
    story = []

    story += [
        Spacer(1, 1.5 * cm),
        Paragraph("Olist E-Commerce Platform", s["ReportSubtitle"]),
        Paragraph("Product Category & Returns Deep Dive", s["ReportTitle"]),
        Paragraph("Cumulative Analysis: January 2017 – August 2018", s["ReportSubtitle"]),
        Spacer(1, 1 * cm),
        hr(s),

        Paragraph("1. Category Revenue Overview", s["SectionHeader"]),
        Paragraph(
            "Ten product categories account for the majority of Olist platform revenue. "
            "Health & Beauty is the single largest category with R$890,000 in cumulative "
            "revenue, followed by Watches & Gifts (R$760,000) and Bed & Bath (R$680,000). "
            "Combined, the top three categories represent 28.6% of total platform GMV.",
            s["BodyText"],
        ),
        chart_img("03_category_revenue_bar.jpg", width_cm=15),
        Paragraph(
            "Figure 1: Top 10 categories by cumulative revenue (Jan 2017 – Aug 2018). "
            "Red bars indicate categories with return rates above 5%.",
            s["Caption"],
        ),

        Paragraph("2. Return Rate Analysis", s["SectionHeader"]),
        hr(s),
        Paragraph(
            "The platform's cumulative return rate across all categories is 4.2%. "
            "Three categories exceed 5%: Watches & Gifts (6.8%), Computers & "
            "Accessories (5.5%), and Cool Stuff (5.2%). These categories present "
            "disproportionate operational risk — each return event consumes logistics "
            "capacity, reduces seller NPS, and increases customer service cost.",
            s["BodyText"],
        ),
        Paragraph(
            "Notably, Watches & Gifts is the second-highest revenue category yet has "
            "the highest return rate. This combination makes it a priority for "
            "intervention: the revenue risk of further return rate increases is "
            "material, and the category has sufficient scale to justify dedicated "
            "quality improvement investment.",
            s["BodyText"],
        ),
        chart_img("04_category_return_rate.png", width_cm=15),
        Paragraph(
            "Figure 2: Return rate by category. Dashed line = platform average (4.2%). "
            "Red = above 5%, orange = 4–5%, green = below 4%.",
            s["Caption"],
        ),
        PageBreak(),

        Paragraph("3. Category Performance Matrix", s["SectionHeader"]),
        hr(s),
        Paragraph(
            "The following table provides a complete view of all 10 categories across "
            "four KPIs: revenue, orders, average order value, and return rate. "
            "Rows highlighted in red exceed the 5% return rate threshold.",
            s["BodyText"],
        ),
        table_img("T02_category_performance_table.png", width_cm=15),
        Paragraph(
            "Table 1: Category performance matrix. Red rows: return rate > 5%.",
            s["Caption"],
        ),

        Paragraph("4. High-Value, High-Return Categories — Action Plan", s["SectionHeader"]),
        hr(s),
        Paragraph(
            "For Watches & Gifts (R$760K revenue, 6.8% return rate):",
            s["SubHeader"],
        ),
        Paragraph("Implement mandatory product authenticity verification for all new "
                  "Watches & Gifts sellers.", s["BulletText"]),
        Paragraph("Require high-resolution product photography from multiple angles "
                  "to reduce 'not as described' returns (currently 19% of return events).",
                  s["BulletText"]),
        Paragraph("Trial a 30-day extended return window for this category to reduce "
                  "buyer uncertainty and potentially improve conversion.", s["BulletText"]),
        Spacer(1, 0.3 * cm),
        Paragraph(
            "For Computers & Accessories (R$570K revenue, 5.5% return rate):",
            s["SubHeader"],
        ),
        Paragraph("Mandate compatibility information in product listings (operating "
                  "system, connector type, model compatibility).", s["BulletText"]),
        Paragraph("Partner with top 5 sellers in this category for quality sampling "
                  "programme — seller-level return rates vary significantly.",
                  s["BulletText"]),

        Spacer(1, 0.5 * cm),
        Paragraph("5. Low-Return Categories — Growth Opportunities", s["SectionHeader"]),
        hr(s),
        Paragraph(
            "Housewares (2.7% return rate) and Bed & Bath (2.9%) demonstrate that "
            "high-volume categories can maintain strong quality metrics. These "
            "categories should be studied as benchmarks for seller onboarding "
            "practices and listing standards. Their lower return rates also suggest "
            "categories where promotional investment would generate net-positive ROI "
            "without being offset by elevated returns.",
            s["BodyText"],
        ),
        Spacer(1, 1 * cm),
        Paragraph(
            "Olist BI Analytics Team  |  July 2018  |  "
            "Data: synthetic, modelled on Olist Brazilian E-Commerce dataset (CC BY-NC-SA)",
            s["FooterText"],
        ),
    ]

    doc.build(story)
    print(f"  saved → {path.name}")
    metadata["P03_product_category_deep_dive.pdf"] = {
        "title": "Product Category & Returns Deep Dive — Olist Platform",
        "document_type": "category analysis report",
        "pages": 4,
        "content_types": ["text", "charts", "tables"],
        "key_topics": [
            "top 10 product categories by revenue",
            "return rate by category",
            "Watches & Gifts highest return rate at 6.8%",
            "Computers & Accessories 5.5% return rate",
            "action plans for high-return categories",
            "growth opportunities in low-return categories",
        ],
        "embedded_charts": [
            "03_category_revenue_bar.jpg",
            "04_category_return_rate.png",
        ],
        "embedded_tables": ["T02_category_performance_table.png"],
        "data_source": "synthetic — modelled on Olist Brazilian E-Commerce dataset",
        "time_period": "January 2017 – August 2018 cumulative",
    }


# ── Run all ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating PDFs...")
    pdf_01()
    pdf_02()
    pdf_03()

    meta_path = Path("/home/claude/multimodal_corpus/metadata/pdfs_metadata.json")
    import json
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nMetadata written → {meta_path}")
    print(f"Total PDFs: {len(metadata)}")
