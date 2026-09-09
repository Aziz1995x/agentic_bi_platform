# Pricing and Freight Policy

**Document type:** Company Policy
**Owner:** Commercial Operations
**Status:** Draft v1 (project-authored, synthetic)
**Last updated:** 2026-09-08

This policy defines how product pricing and freight (shipping) charges are
set and reported across the marketplace.

---

## Pricing Model

Prices in `order_items.price` are seller-set at the individual product level.
The platform does not centrally set prices, but enforces the following
rules:

- **Minimum price floor:** no product may be listed below R$5.00.
- **Price changes** take effect only for orders placed after the change; a
  price change never retroactively affects `order_items.price` on existing
  orders. This means historical revenue analysis always reflects the price
  actually paid, not the current listed price.

## Freight Calculation

Freight (`order_items.freight_value`) is calculated per item based on:

1. Product weight and dimensions (from the `products` table)
2. Distance between the seller's region (derived via
   `sellers → employees → regions`) and the customer's delivery region
   (derived via `state_region_map`)

Freight is **not** a flat rate and is **not** a profit center — freight
revenue is tracked separately from product Revenue (see KPI Definitions) and
should never be added into headline revenue figures without an explicit
"incl. freight" label.

## Discounts and Promotions

- Discounts, when applied, are reflected directly in `order_items.price` as
  the post-discount price. There is no separate discount-amount column in
  the current schema — this is a known limitation. Any analysis of "discount
  depth" requires an external reference price, which is out of scope until
  a `list_price` field is added to the schema.
- Promotional campaigns are not currently tracked in structured data. Any
  revenue dip or spike correlated with a promotion must be investigated via
  qualitative means (e.g. checking historical business reports in the
  knowledge base) rather than assumed from sales data alone.

## Regional Price Variation

Sellers may price the same product differently across their own listings,
but the platform does not support region-specific pricing (i.e. a single
seller cannot charge Southeast customers a different price than Northeast
customers for the same listed item). Any observed regional price variation
in the data reflects different sellers serving different regions, not
deliberate regional pricing strategy — this distinction matters when
investigating regional revenue differences.

## Currency

All prices in the dataset are in Brazilian Real (R$ / BRL). No currency
conversion is applied anywhere in the pipeline; reports should always state
figures are in BRL unless a conversion has been explicitly requested and
performed.
