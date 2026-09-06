# Olist Raw Dataset Notes

This directory contains the original Olist Brazilian E-Commerce datasets used as the real-data source for the Agentic BI Platform.

The raw datasets are kept here as the source data, with a small number of project-specific changes documented below.

## Project-specific changes

### 1. Product column name corrections in olist_products_dataset.csv

The original Olist products dataset contains two column names with the spelling `lenght`.

For consistency and readability in the PostgreSQL schema, these were renamed:

| Original Olist column | Project column |
|---|---|
| `product_name_lenght` | `product_name_length` |
| `product_description_lenght` | `product_description_length` |

The corresponding PostgreSQL columns use the corrected names:

```text
product_name_length
product_description_length
```

The `load_products()` loader therefore expects the corrected column names to already be present in the raw CSV.

### 2. Missing product-category translations in product_category_name_translation.csv

The Olist product category translation dataset did not contain translations for two category values that are present in the products dataset:

- `pc_gamer`
- `portateis_cozinha_e_preparadores_de_alimentos`

To satisfy the PostgreSQL foreign-key relationship from `products.product_category_name` to `product_category_name_translation.product_category_name`, translations were added:

| Portuguese category | Project English translation |
|---|---|
| `pc_gamer` | `gaming_pc` |
| `portateis_cozinha_e_preparadores_de_alimentos` | `portable_kitchen_food_preparers` |

These additions are project-specific and should be preserved if the raw datasets are replaced or re-downloaded.

## Important

These modifications are intentional project decisions and are **not changes to the original Olist source dataset**.

If the raw Olist files are downloaded again, verify that these changes are reapplied before running the database loading pipeline.

The PostgreSQL schema and loaders are designed around these project-specific column names and category translations.
