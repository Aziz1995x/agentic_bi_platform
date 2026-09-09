# Refund and Returns Policy

**Document type:** Company Policy
**Owner:** Customer Operations
**Status:** Draft v1 (project-authored, synthetic)
**Last updated:** 2026-09-08

This policy governs when a delivered order is eligible for return and refund,
and how returns are recorded for analytical purposes. It applies to all
marketplace sellers on the platform.

---

## Eligibility Window

A customer may request a return within **7 calendar days** of the delivery
date recorded in `orders.order_delivered_customer_date`. Requests submitted
after this window are handled as **exceptions** and require manual approval;
they are still recorded in the `returns` table but flagged with
`is_exception = true` so they are excluded from standard return-rate
reporting unless explicitly requested.

## Eligible Reasons

Returns are accepted for the following reasons, recorded per return record:

- **Damaged in transit** — item arrived visibly damaged
- **Not as described** — item materially differs from its product listing
- **Wrong item shipped** — seller fulfillment error
- **Change of mind** — customer no longer wants the item (subject to the
  restocking condition below)
- **Defective / not working** — item fails to function as intended

"Change of mind" returns are the only category subject to a restocking fee
(see below); all other categories are fully refunded including original
shipping cost.

## Restocking Fee

For "change of mind" returns only, a restocking fee of **10% of the item
price** is deducted from the refund, provided the item is returned in
original condition. This fee does not apply to any other return reason.

## Refund Processing

- Refunds are issued to the original payment method used in
  `order_payments`.
- Processing time is **5–10 business days** from the day the returned item
  is received back at the seller's or platform's warehouse — not from the
  date the return was requested.
- Partial refunds (e.g. one item from a multi-item order) are calculated per
  `order_item_id`, not by dividing the total order value evenly.

## Non-Returnable Categories

Certain product categories are final-sale and not eligible for "change of
mind" returns, though they remain eligible for damaged/defective/wrong-item
claims:

- Perishable goods
- Personal care items once opened
- Made-to-order or customized products

## Seller Responsibility and Return Rate Impact

A return is attributed to the `seller_id` on the original `order_items`
record, regardless of which party (customer error, seller error, carrier
damage) ultimately caused it. This means a seller's raw return rate reflects
total return volume, not fault — analysts should not describe a high return
rate as "seller quality issues" without checking the reason breakdown first.
See **Return Rate** in the KPI Definitions document for the statistical
threshold used to flag a return rate as "unusually high."
