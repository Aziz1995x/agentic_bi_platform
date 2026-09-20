"""
generate_retention_pdf.py
─────────────────────────
Generates the customer_retention_framework.pdf document that will be
added to the text RAG corpus via update_vectorstore().

Run once from the project root:
    python scripts/generate_retention_pdf.py
"""

from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak,
)
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT

OUT = Path("documents")
OUT.mkdir(exist_ok=True)

PAGE_W, PAGE_H = A4
BRAND_DARK  = colors.HexColor("#1E3A5F")
BRAND_BLUE  = colors.HexColor("#1A56DB")
BRAND_GREY  = colors.HexColor("#6B7280")
LIGHT_GREY  = colors.HexColor("#F3F4F6")


def get_styles():
    base = getSampleStyleSheet()
    return {
        "Title": ParagraphStyle(
            "DocTitle", parent=base["Title"],
            fontSize=22, textColor=BRAND_DARK,
            spaceAfter=6, alignment=TA_CENTER,
        ),
        "Subtitle": ParagraphStyle(
            "Subtitle", parent=base["Normal"],
            fontSize=11, textColor=BRAND_GREY,
            spaceAfter=4, alignment=TA_CENTER,
        ),
        "H1": ParagraphStyle(
            "H1", parent=base["Heading1"],
            fontSize=14, textColor=BRAND_DARK,
            spaceBefore=16, spaceAfter=6,
        ),
        "H2": ParagraphStyle(
            "H2", parent=base["Heading2"],
            fontSize=11, textColor=BRAND_BLUE,
            spaceBefore=10, spaceAfter=4,
        ),
        "Body": ParagraphStyle(
            "Body", parent=base["Normal"],
            fontSize=9.5, leading=14,
            spaceAfter=6, alignment=TA_JUSTIFY,
        ),
        "Bullet": ParagraphStyle(
            "Bullet", parent=base["Normal"],
            fontSize=9.5, leading=14,
            spaceAfter=3, leftIndent=14, bulletIndent=4,
        ),
        "Footer": ParagraphStyle(
            "Footer", parent=base["Normal"],
            fontSize=7.5, textColor=BRAND_GREY, alignment=TA_CENTER,
        ),
    }


def build_pdf():
    path = OUT / "customer_retention_framework.pdf"
    doc  = SimpleDocTemplate(
        str(path), pagesize=A4,
        leftMargin=2.2*cm, rightMargin=2.2*cm,
        topMargin=2.2*cm, bottomMargin=2.2*cm,
    )
    s = get_styles()
    story = []

    # ── Cover ──────────────────────────────────────────────────────────
    story += [
        Spacer(1, 1.5*cm),
        Paragraph("Olist E-Commerce Platform", s["Subtitle"]),
        Paragraph("Customer Retention Framework", s["Title"]),
        Paragraph(
            "Strategic Guidelines for Customer Lifecycle Management  "
            "|  Version 1.3  |  March 2018",
            s["Subtitle"],
        ),
        Spacer(1, 0.8*cm),
        HRFlowable(width="100%", thickness=0.5, color=BRAND_GREY),
        Spacer(1, 0.8*cm),
    ]

    # ── Section 1: Purpose ─────────────────────────────────────────────
    story += [
        Paragraph("1. Purpose and Scope", s["H1"]),
        Paragraph(
            "This framework defines Olist's approach to customer retention "
            "across the full customer lifecycle, from first purchase through "
            "long-term loyalty. It establishes the segmentation model used "
            "to classify customers, the retention interventions applied to "
            "each segment, and the KPIs used to measure retention programme "
            "effectiveness. All retention initiatives must align with this "
            "framework before implementation.",
            s["Body"],
        ),
        Paragraph(
            "Retention is a platform-level priority because the revenue "
            "contribution of repeat customers is disproportionate to their "
            "share of the customer base. Platform data shows that repeat "
            "buyers (2 or more orders) represent approximately 26% of "
            "registered customers but contribute 61% of total GMV. The "
            "cost of acquiring a new customer is estimated at 5–7x the cost "
            "of retaining an existing one at equivalent revenue contribution.",
            s["Body"],
        ),
    ]

    # ── Section 2: Customer Segments ──────────────────────────────────
    story += [
        Paragraph("2. Customer Segmentation Model", s["H1"]),
        Paragraph(
            "All registered customers are classified into one of four "
            "segments based on recency, frequency, and order value. "
            "Segment assignments are recalculated monthly.",
            s["Body"],
        ),
        Spacer(1, 0.3*cm),
        Table(
            [
                ["Segment", "Definition", "Retention Priority", "% of Base"],
                ["New",
                 "First order placed within past 90 days",
                 "High — convert to repeat",
                 "~18%"],
                ["Active Repeat",
                 "2–3 orders, most recent within 180 days",
                 "High — increase frequency",
                 "~18%"],
                ["Loyal",
                 "4+ orders, most recent within 180 days",
                 "Medium — protect and reward",
                 "~7%"],
                ["Lapsed",
                 "No order in past 180 days",
                 "Medium — win-back campaign",
                 "~57%"],
            ],
            colWidths=[3.2*cm, 6.5*cm, 4.0*cm, 2.5*cm],
            style=TableStyle([
                ("BACKGROUND",    (0,0), (-1,0),  BRAND_DARK),
                ("TEXTCOLOR",     (0,0), (-1,0),  colors.white),
                ("FONTNAME",      (0,0), (-1,0),  "Helvetica-Bold"),
                ("FONTSIZE",      (0,0), (-1,-1), 8.5),
                ("ROWBACKGROUNDS",(0,1), (-1,-1), [LIGHT_GREY, colors.white]),
                ("GRID",          (0,0), (-1,-1), 0.3, colors.HexColor("#D1D5DB")),
                ("TOPPADDING",    (0,0), (-1,-1), 5),
                ("BOTTOMPADDING", (0,0), (-1,-1), 5),
                ("VALIGN",        (0,0), (-1,-1), "TOP"),
            ]),
        ),
        Spacer(1, 0.6*cm),
    ]

    # ── Section 3: Retention Interventions ────────────────────────────
    story += [
        Paragraph("3. Retention Interventions by Segment", s["H1"]),

        Paragraph("3.1 New Customer Onboarding (Days 1–90)", s["H2"]),
        Paragraph(
            "The 90-day post-first-purchase window is the highest-leverage "
            "retention opportunity. Data shows that customers who make a "
            "second purchase within 90 days of their first have a 3.2x "
            "higher 12-month lifetime value than those who do not.",
            s["Body"],
        ),
        Paragraph("Day 3: Delivery confirmation email with product care tips and "
                  "category-relevant recommendations.", s["Bullet"]),
        Paragraph("Day 14: Review request email if no review submitted. "
                  "Incentivise with 5% discount voucher on next order.", s["Bullet"]),
        Paragraph("Day 45: Re-engagement email with personalised product "
                  "recommendations based on first purchase category.", s["Bullet"]),
        Paragraph("Day 75: Final new-customer offer — 10% discount valid for "
                  "15 days, expiring before the 90-day window closes.", s["Bullet"]),
        Spacer(1, 0.3*cm),

        Paragraph("3.2 Active Repeat Customer Development", s["H2"]),
        Paragraph(
            "The objective for Active Repeat customers is frequency increase "
            "— moving from 2–3 orders per year toward 4+ orders, which "
            "defines the Loyal tier. Interventions should be personalised "
            "to purchase history and avoid generic mass-email fatigue.",
            s["Body"],
        ),
        Paragraph("Monthly: Category-specific recommendations based on "
                  "last 3 purchases.", s["Bullet"]),
        Paragraph("Quarterly: Loyalty progress notification showing how many "
                  "orders until Loyal tier status is reached.", s["Bullet"]),
        Paragraph("Seasonal: Priority access to promotional events 24 hours "
                  "before general customer communications.", s["Bullet"]),
        Spacer(1, 0.3*cm),

        Paragraph("3.3 Loyal Customer Retention", s["H2"]),
        Paragraph(
            "Loyal customers require protection, not conversion. "
            "The primary risk is involuntary churn driven by a poor "
            "experience — delayed delivery, a high return rate seller, "
            "or a failed customer service interaction. Retention for this "
            "segment is therefore operationally focused rather than "
            "promotional.",
            s["Body"],
        ),
        Paragraph("Real-time: Proactive delivery delay notification if carrier "
                  "SLA is breached, with apology voucher issued automatically.",
                  s["Bullet"]),
        Paragraph("Bi-annual: Loyalty programme status review with summary "
                  "of benefits unlocked.", s["Bullet"]),
        Paragraph("Post-return: Personal outreach from customer success team "
                  "within 48 hours of any return event for Loyal customers.",
                  s["Bullet"]),
        Spacer(1, 0.3*cm),

        Paragraph("3.4 Lapsed Customer Win-Back", s["H2"]),
        Paragraph(
            "Win-back campaigns target customers with no order in 180+ days. "
            "This is the largest segment by headcount (approximately 57% "
            "of the registered base) but has the lowest response rate. "
            "Win-back spend should be capped at the estimated residual "
            "lifetime value of each customer to avoid unprofitable reactivation.",
            s["Body"],
        ),
        Paragraph("Month 6: Win-back email with 15% discount, "
                  "valid 30 days.", s["Bullet"]),
        Paragraph("Month 9: Second win-back attempt with curated "
                  "'what's new' product selection.", s["Bullet"]),
        Paragraph("Month 12: Final reactivation attempt. "
                  "If no response, customer is moved to dormant status "
                  "and removed from active campaign lists.", s["Bullet"]),
        PageBreak(),
    ]

    # ── Section 4: KPIs ───────────────────────────────────────────────
    story += [
        Paragraph("4. Retention KPIs and Measurement", s["H1"]),
        Paragraph(
            "Retention programme performance is measured monthly against "
            "the following KPIs. Targets are set annually and reviewed "
            "quarterly by the Customer Experience leadership team.",
            s["Body"],
        ),
        Spacer(1, 0.3*cm),
        Table(
            [
                ["KPI", "Definition", "Target", "Owner"],
                ["Repeat Purchase Rate",
                 "% of customers placing a second order within 90 days of first",
                 "≥ 28%",
                 "CRM Team"],
                ["12-Month Retention Rate",
                 "% of customers active in month M still active in month M+12",
                 "≥ 35%",
                 "Analytics"],
                ["Loyal Tier Conversion",
                 "% of Active Repeat customers reaching Loyal tier within 12 months",
                 "≥ 22%",
                 "CRM Team"],
                ["Win-Back Rate",
                 "% of Lapsed customers placing an order within 90 days of "
                 "win-back campaign",
                 "≥ 8%",
                 "CRM Team"],
                ["Net Promoter Score",
                 "Standard NPS survey, quarterly sample",
                 "≥ 45",
                 "CX Team"],
            ],
            colWidths=[3.8*cm, 6.2*cm, 2.0*cm, 2.8*cm],
            style=TableStyle([
                ("BACKGROUND",    (0,0), (-1,0),  BRAND_DARK),
                ("TEXTCOLOR",     (0,0), (-1,0),  colors.white),
                ("FONTNAME",      (0,0), (-1,0),  "Helvetica-Bold"),
                ("FONTSIZE",      (0,0), (-1,-1), 8.5),
                ("ROWBACKGROUNDS",(0,1), (-1,-1), [LIGHT_GREY, colors.white]),
                ("GRID",          (0,0), (-1,-1), 0.3, colors.HexColor("#D1D5DB")),
                ("TOPPADDING",    (0,0), (-1,-1), 5),
                ("BOTTOMPADDING", (0,0), (-1,-1), 5),
                ("VALIGN",        (0,0), (-1,-1), "TOP"),
            ]),
        ),
        Spacer(1, 0.8*cm),
    ]

    # ── Section 5: Definitions ────────────────────────────────────────
    story += [
        Paragraph("5. Definitions", s["H1"]),
        Paragraph(
            "Active customer: A customer who has placed at least one order "
            "in the past 180 days with delivered status confirmed. "
            "Customers with only cancelled or returned orders do not qualify "
            "as active regardless of order recency.",
            s["Body"],
        ),
        Paragraph(
            "Churn: The transition of a customer from any active segment "
            "(New, Active Repeat, or Loyal) to the Lapsed segment, "
            "triggered by 180 consecutive days without a delivered order.",
            s["Body"],
        ),
        Paragraph(
            "Lifetime value (LTV): The sum of gross margin contribution "
            "from all orders attributed to a customer from first purchase "
            "to the present date. Projected LTV extends this using a "
            "12-month forward model based on segment purchase frequency "
            "and average order value.",
            s["Body"],
        ),
        Paragraph(
            "Repeat purchase rate: Measured at the cohort level — "
            "of all customers who made their first purchase in month M, "
            "what percentage placed a second purchase by month M+3. "
            "This is not the same as the platform-level repeat customer "
            "rate reported in business dashboards, which counts all "
            "customers with 2+ lifetime orders regardless of timing.",
            s["Body"],
        ),
        Spacer(1, 1*cm),
        Paragraph(
            "Olist Customer Experience Team  |  Version 1.3  |  March 2018  |  "
            "Internal use only  |  Synthetic document for the Agentic BI Platform project",
            s["Footer"],
        ),
    ]

    doc.build(story)
    print(f"Generated: {path}")
    return path


if __name__ == "__main__":
    build_pdf()
