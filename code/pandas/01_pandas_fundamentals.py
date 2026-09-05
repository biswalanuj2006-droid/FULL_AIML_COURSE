# ============================================================
# PANDAS FUNDAMENTALS FOR ML
# The subset of Pandas you actually use to build ML datasets.
# Run: python 01_pandas_fundamentals.py
# Requires: pandas, numpy
# ============================================================
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)

# ------------------------------------------------------------
# 1. Build a realistic customer-transaction-like frame
# ------------------------------------------------------------
n = 500
df = pd.DataFrame({
    "customer_id": rng.integers(1, 50, n),
    "date": pd.date_range("2024-01-01", periods=n, freq="h"),
    "amount": np.round(rng.lognormal(mean=3, sigma=0.8, size=n), 2),
    "category": rng.choice(["electronics", "grocery", "fashion", "home"], n),
    "rating": rng.choice([1, 2, 3, 4, 5, np.nan], n, p=[.05, .1, .2, .3, .25, .1]),
})
print("Head:\n", df.head(3))

# ------------------------------------------------------------
# 2. Missing data — check and fill
# ------------------------------------------------------------
print("\nMissing per column:\n", df.isna().sum())
print("Rating fill rate:", df["rating"].notna().mean().round(2))

# ------------------------------------------------------------
# 3. GroupBy + aggregation — feature engineering style
# ------------------------------------------------------------
customer_stats = (
    df.groupby("customer_id")
      .agg(
          total_spend=("amount", "sum"),
          n_orders=("amount", "count"),
          avg_rating=("rating", "mean"),
          favorite_category=("category", lambda s: s.mode().iloc[0] if len(s) else None),
      )
      .reset_index()
)
customer_stats["avg_order"] = customer_stats["total_spend"] / customer_stats["n_orders"]
print("\nCustomer-level features:\n", customer_stats.head(3))
print("Rows:", len(customer_stats), "unique customers")

# ------------------------------------------------------------
# 4. Category encoding for ML
# ------------------------------------------------------------
category_encoded = pd.get_dummies(customer_stats["favorite_category"].fillna("unknown"),
                                  prefix="fav")
ml_ready = pd.concat([customer_stats.drop(columns="favorite_category"), category_encoded], axis=1)
print("\nML-ready shape:", ml_ready.shape)

# ------------------------------------------------------------
# 5. Datetime features
# ------------------------------------------------------------
df["hour"] = df["date"].dt.hour
df["dow"] = df["date"].dt.dayofweek            # Monday=0
df["is_weekend"] = df["dow"] >= 5
print("\nDatetime-derived features:\n", df[["date", "hour", "dow", "is_weekend"]].head(3))

# ------------------------------------------------------------
# 6. Merge: attach customer features back to transactions
# ------------------------------------------------------------
df_merged = df.merge(customer_stats[["customer_id", "total_spend"]],
                     on="customer_id", how="left")
print("\nMerged columns:", df_merged.columns.tolist())

# ------------------------------------------------------------
# 7. Pivot — wide format for time-series features
# ------------------------------------------------------------
pivot = (df.set_index("date")
           .groupby([pd.Grouper(freq="D"), "category"])["amount"]
           .sum().unstack(fill_value=0))
print("\nDaily spend by category (last 3 days):\n", pivot.tail(3))

# ------------------------------------------------------------
# 8. Vectorized ops — replace apply/loops when possible
# ------------------------------------------------------------
# Bad: df.apply(lambda r: r.amount * 1.1, axis=1)
df["amount_taxed"] = df["amount"] * 1.1        # vectorized
print("\nVectorized multiply confirmed:", np.allclose(
    df["amount_taxed"], df["amount"] * 1.1))

# ------------------------------------------------------------
# Cheat line: the single pipeline that wins Kaggle tabs
cleaned = (df.dropna(subset=["rating"])
             .query("amount > 0")
             .sort_values("date"))
print("\nCleaned pipeline shape:", cleaned.shape)

# ------------------------------------------------------------
# Golden rules:
#  1. Prefer vectorized ops and groupby over apply(axis=1) loops.
#  2. Handle missing values EXPLICITLY before training.
#  3. Never do groupby inside a for loop over unique values.
#  4. Use categorical dtype for low-cardinality strings (memory).
# ============================================================
