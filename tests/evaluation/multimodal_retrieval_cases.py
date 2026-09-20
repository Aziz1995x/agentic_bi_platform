"""
multimodal_retrieval_cases.py
─────────────────────────────
Fixed evaluation set for the multimodal RAG retrieval pipeline.

Design decisions — same principles as Phase 5 text eval set
────────────────────────────────────────────────────────────
1. Fixed at definition time.
   No case is added or removed based on results.  If a case fails,
   we understand why and either fix the pipeline or record the verdict.

2. Ground truth is file names, not chunk IDs.
   expected_sources lists the file_name values of artefacts that MUST
   appear in the retrieved results for the case to pass.  This is the
   multimodal equivalent of the text eval's "which document should be
   retrieved" — the same source-level granularity used in Phase 5.

3. Cases cover all three artefact types.
   Charts, table images, and PDF text pages each have dedicated cases,
   plus multi-source cases that require artefacts of different types.

4. Cases vary in specificity.
   Some are precise ("which file shows the Q2 2018 KPI numbers") and
   some are intentionally broad ("delivery performance") to test whether
   the retriever ranks the right artefact above generic alternatives.

5. source_types field is informational only.
   The eval runner does not filter by source_type before scoring — it
   retrieves from the full index and checks whether ground-truth files
   appear in the top-k results.  This tests the index as a whole, not
   a pre-filtered subset.

Case inventory (10 cases)
─────────────────────────
  MM01  chart — monthly revenue trend line chart
  MM02  chart — quarterly revenue bar chart with Q2 dip
  MM03  chart — return rate by category bar chart
  MM04  chart — regional delivery time bar chart
  MM05  chart — payment method donut chart
  MM06  table — Q2 2018 KPI summary table
  MM07  table — category performance matrix
  MM08  table — quarterly comparison table
  MM09  PDF   — Q2 2018 root cause analysis report
  MM10  multi — Q2 revenue decline (should surface chart + PDF text)
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class MultimodalCase:
    id:               str
    query:            str
    expected_sources: list[str]   # file_name values that must appear in top-k
    source_types:     list[str]   # informational: which types are expected
    notes:            str = ""


MULTIMODAL_CASES: list[MultimodalCase] = [

    MultimodalCase(
        id="MM01",
        query="Show me the monthly revenue and order volume trend for the Olist platform",
        expected_sources=["01_monthly_revenue_trend.png"],
        source_types=["chart"],
        notes="Precise match — chart title directly mirrors the query.",
    ),

    MultimodalCase(
        id="MM02",
        query="Show me the quarterly revenue bar chart comparing performance across 2017 and 2018 quarters",
        expected_sources=["02_quarterly_revenue_bar.png"],
        source_types=["chart"],
        notes=(
            "Tests whether the Q2 2018 dip annotation in the chart caption "
            "is retrieved by a query framed around decline detection."
        ),
    ),

    MultimodalCase(
        id="MM03",
        query="What is the return rate for each product category and which categories exceed the platform average?",
        expected_sources=["04_category_return_rate.png"],
        source_types=["chart"],
        notes=(
            "Category return rates appear in both the chart (04) and the "
            "category performance table (T02). Both contain the answer; "
            "the chart is the primary expected source."
        ),
    ),

    MultimodalCase(
        id="MM04",
        query="How does average delivery time differ across Brazilian regions?",
        expected_sources=["06_regional_delivery_days.png"],
        source_types=["chart"],
        notes="Specific enough that only the delivery bar chart is the target.",
    ),

    MultimodalCase(
        id="MM05",
        query="What payment methods do Olist customers use and what share of revenue does each represent?",
        expected_sources=["07_payment_method_donut.jpg"],
        source_types=["chart"],
        notes="JPG file — confirms the retriever handles both PNG and JPG source files.",
    ),

    MultimodalCase(
        id="MM06",
        query="What were the key KPIs for Q2 2018 including revenue, orders, and return rate?",
        expected_sources=["T01_kpi_summary_table.png"],
        source_types=["table"],
        notes=(
            "KPI numbers appear in multiple places (table T01, quarterly "
            "table T05, PDF text). T01 is the primary KPI dashboard source."
        ),
    ),

    MultimodalCase(
        id="MM07",
        query="Compare revenue, order count, average order value, and return rate across product categories",
        expected_sources=["T02_category_performance_table.png"],
        source_types=["table"],
        notes=(
            "The category performance matrix (T02) is the only artefact "
            "that shows all four metrics side by side. Tests whether "
            "the caption's tabular structure survives embedding well enough "
            "to match a multi-metric query."
        ),
    ),

    MultimodalCase(
        id="MM08",
        query="Show me a quarter by quarter comparison of revenue growth from 2017 to 2018",
        expected_sources=["T05_quarterly_comparison_table.png"],
        source_types=["table"],
        notes="Temporal framing targets the quarterly comparison table specifically.",
    ),

    MultimodalCase(
        id="MM09",
        query="What are the root causes of the Q2 2018 revenue decline according to the investigation?",
        expected_sources=["P02_revenue_decline_analysis.pdf"],
        source_types=["pdf_text"],
        notes=(
            "The root cause analysis PDF is the only artefact with an "
            "investigation-framed conclusion. PDF text pages are indexed "
            "separately — this tests whether prose retrieval from PDFs works "
            "alongside image retrieval in the same index."
        ),
    ),

    MultimodalCase(
        id="MM10",
        query="Why did revenue decline in Q2 2018 and what data supports this finding?",
        expected_sources=[
            "02_quarterly_revenue_bar.png",
            "P02_revenue_decline_analysis.pdf",
        ],
        source_types=["chart", "pdf_text"],
        notes=(
            "Multi-source case. Both the bar chart (visual evidence of the "
            "decline) and the root cause PDF (analytical explanation) must "
            "appear in the top-k results. Tests whether a single query can "
            "simultaneously surface image and text artefacts. "
            "Requires k ≥ 2 for full coverage."
        ),
    ),
]
