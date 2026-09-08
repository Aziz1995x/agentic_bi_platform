# Customer Segmentation

**Document type:** Business Rules
**Owner:** Analytics Team
**Status:** Draft v1 (project-authored, synthetic)
**Last updated:** 2026-09-08

This document defines how customers are segmented for comparative reporting.
It is the authoritative source when a question asks to "compare enterprise
and retail customers" or similar segment-based analysis.

---

## Segment Definitions

The platform recognizes two customer segments, assigned based on trailing
365-day purchasing behavior (not self-declared by the customer):

### Retail

The default segment. Any customer not meeting the Enterprise criteria below
is classified as Retail.

### Enterprise

A customer (resolved to `customer_unique_id` via the crosswalk) qualifies as
**Enterprise** if, in the trailing 365 days, they meet **both** of the
following:

- 10 or more delivered orders, **and**
- Total delivered Revenue (per the KPI Definitions document) of R$5,000 or
  more

Both conditions must hold — high order count with low average value (e.g.
many small orders) does not qualify, nor does one large order without
repeat volume.

## Re-evaluation Cadence

Segment assignment is recalculated monthly on a rolling 365-day window. A
customer can move from Retail to Enterprise or back again month to month.
Historical reports should state which month's segmentation snapshot was
used, since a customer's segment at reporting time may differ from their
segment at the time an individual historical order was placed.

## Comparative Reporting Rules

When comparing Enterprise vs. Retail customers, the following must be
normalized for fairness, since Enterprise customers by definition have
higher volume:

- Compare **AOV**, not raw revenue, when comparing "value per customer"
- Compare **return rate** (per KPI Definitions) rather than raw return
  counts
- State sample sizes for both segments — Enterprise is a much smaller
  population than Retail, and small-sample statistics should be flagged per
  the Return Rate minimum-volume rule in KPI Definitions

## Relationship to Active Customer Status

Segmentation (Enterprise/Retail) and Active Customer status (see KPI
Definitions) are independent classifications. A customer can be Enterprise
and non-active simultaneously — for example, a customer who was a heavy
purchaser 8 months ago but has not ordered in the last 180 days retains
their Enterprise segment (recalculated monthly) even though they are
currently non-active. Reports comparing segments should specify whether the
comparison is restricted to active customers only or includes all customers
regardless of recency.
