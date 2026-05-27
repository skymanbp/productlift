# Data

This project uses the **Olist Brazilian E-Commerce** dataset — real, public order
data from a Brazilian marketplace (~100k orders, 2016–2018).

- Source: https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
- License: CC BY-NC-SA 4.0 (non-commercial; fine for learning/portfolio)
- Full column-by-column schema: see [docs/data-dictionary.md](../docs/data-dictionary.md)

## Get the data

**Option A — Kaggle API (automated)**
1. Create a token at https://www.kaggle.com/settings → "Create New API Token".
2. Put `KAGGLE_USERNAME` and `KAGGLE_KEY` in your `.env`.
3. `make data` (runs `python -m scripts.download_data --kaggle`).

**Option B — Manual**
1. Download the zip from the Kaggle link above.
2. Unzip all 9 CSVs into `data/raw/`.
3. Verify: `python -m scripts.download_data --check`.

## Folder layout (the standard raw → interim → processed flow)

| Folder            | Contents                                   | Committed? |
|-------------------|--------------------------------------------|------------|
| `data/raw/`       | Original Olist CSVs. **Treat as read-only.** | No (gitignored) |
| `data/interim/`   | The enriched base table (joined events).   | No |
| `data/processed/` | Final modeling matrices (features + label).| No |

Data is never committed — it's large and regenerable. Only `.gitkeep` files are
tracked so the folders exist. The *pipeline that produces* the data is the artifact.

## The prediction target

We adapt the Wayfair "underperforming products" case to what Olist supports. Olist
is order-level (no impression/traffic logs), so "conversion" isn't directly
observable. We define **underperformance relative to a product's category** —
bottom-quartile revenue among products with enough orders — measured in an
**outcome window** that comes *after* the features' as-of date (to prevent leakage).
The full reasoning, and the alternative framings, are in
[docs/problem-framing.md](../docs/problem-framing.md).
