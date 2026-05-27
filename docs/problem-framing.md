# Problem framing

> This is interview Steps 1–2 made concrete. A senior candidate spends real time
> here before naming a single model. So do we.

## The business problem

A marketplace has products that get listed but **underperform** — low sales, low
revenue, poor customer outcomes. The business wants to (a) **identify** them early
and (b) **improve** them (fix the listing, adjust price, promote, or de-list).
Improving the tail of a catalog lifts revenue, conversion, and customer trust.

## Why this is subtle (and a great interview problem)

"Underperforming" is not given — we have to **define** it, and the definition *is*
the modeling problem:

- **Relative to what?** Raw revenue just measures category (electronics ≫ pens). We
  define underperformance **within category** so we compare like with like.
- **Caused by what?** A product may underperform because it's genuinely bad, *or*
  because it's badly ranked / under-exposed. That's a **causal** question, not a
  predictive one — and it's why this repo has a whole `causal/` module. (The
  interview doc's sharpest push: *"How do you know it's the product and not its
  ranking position?"*)
- **Predict or rank or cause?** The same business goal supports several ML framings.

## ML framings (and which we build)

| Framing | Question | In this repo |
|---|---|---|
| **Classification** | Will this product underperform next period? | ✅ primary target |
| Regression | What revenue/sales will it do? | stretch (swap `target.task`) |
| Ranking | Which products to surface / fix first? | NDCG metric + uplift targeting |
| **Causal** | What *causes* underperformance; what intervention helps? | ✅ `causal/` module |

We start with **binary classification** (`is_underperforming`) because it's the
cleanest place to learn the full lifecycle, then use ranking metrics and causal
methods on top.

## How Olist constrains the framing (important honesty)

The Wayfair prompt talks about "traffic but poor conversion." **Olist has no
impression/traffic logs** — it's order-level. So we *cannot* literally model
conversion. We adapt:

- **Underperformance target** = bottom-quartile **revenue within category**, among
  products with ≥ `min_orders_for_label` orders, measured in an **outcome window
  after** the feature cut-off date.
- **Causal question we _can_ answer** with Olist: *does faster delivery cause better
  reviews?* (delivery time is a non-randomized "treatment" with measurable
  confounders) — perfect for propensity/IPW.
- **Exposure bias** is taught with a small **synthetic** click-position fixture
  (Olist has no position data), because the *concept* — position confounds quality —
  is the transferable interview skill.

The framework transfers even though the exact metric differs. Be ready to say
exactly this in an interview: *"With this dataset I'd define the target as … because
the impression data needed for true conversion isn't available; the approach is
identical."*

## Success metrics

- **Offline (proxy):** PR-AUC (imbalanced), calibration (Brier/ECE), recall@top-k%.
- **Business (truth):** revenue lift, conversion lift, reduced returns, retention —
  validated by an **A/B test** (`experimentation/`), because *offline metrics are
  only proxies for business impact*.

## Guardrails / tradeoffs to keep in mind

Maximizing one metric can hurt the business: over-promoting low-margin products,
hurting supplier diversity, or boosting conversion while returns rise. Every
modeling decision should name the business metric it serves and the guardrail it
could threaten.
