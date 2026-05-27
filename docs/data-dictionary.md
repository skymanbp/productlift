# Data dictionary — Olist

The Olist dataset is 9 CSVs that join into a star-ish schema around `orders`. Below
are the columns we actually use; the full set is documented on the
[Kaggle page](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce).

## Join graph
```
order_items ──(order_id)──> orders ──(customer_id)──> customers
     │                          │
 (product_id)              (order_id)
     ▼                          ▼
  products                  reviews
     │
 (category)
     ▼
 category_translation
```

## olist_orders_dataset
| column | meaning |
|---|---|
| order_id | PK |
| customer_id | FK → customers |
| order_status | delivered / shipped / canceled / … (we keep `delivered`) |
| order_purchase_timestamp | **the prediction-time anchor** — when the order was placed |
| order_delivered_customer_date | actual delivery |
| order_estimated_delivery_date | promised delivery (→ `late_delivery`) |

## olist_order_items_dataset
| column | meaning |
|---|---|
| order_id, order_item_id | PK |
| product_id | FK → products |
| seller_id | FK → sellers |
| price | item price (→ revenue when summed) |
| freight_value | shipping cost |

## olist_products_dataset
| column | meaning |
|---|---|
| product_id | PK |
| product_category_name | category (Portuguese; translated via category_translation) |
| product_photos_qty | # listing photos (listing-quality signal) |
| product_description_lenght | description length [sic — Olist's typo] |
| product_name_lenght | title length |
| product_weight_g, product_*_cm | physical attributes (→ volume) |

## olist_order_reviews_dataset
| column | meaning |
|---|---|
| order_id | FK → orders |
| review_score | 1–5 (→ satisfaction / `bad_review_rate`) |
| review_creation_date | when the review was left |

## Derived fields (built in `app/data/load.py`)
| field | definition |
|---|---|
| delivery_days | (delivered − purchased) in days |
| late_delivery | delivered > estimated → 1 |

## ⚠️ Leakage notes
- `review_score`, `delivery_days`, `late_delivery` are only knowable **after** the
  order completes. They are valid **features** only when aggregated over the
  **feature window** (past orders), and they must never describe the same orders
  used to compute the **label**. See `.claude/skills/leakage-audit/`.
- `order_purchase_timestamp` is the clock for everything. Every split and rolling
  window is defined relative to it.
