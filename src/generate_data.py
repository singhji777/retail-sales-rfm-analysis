"""Generates a realistic *synthetic* Indian retail orders dataset (with deliberate data-quality issues).

Why synthetic? Reproducible, no licence issues, and lets us plant realistic patterns
(seasonality, discount-driven losses, customer loyalty tiers) that the analysis then discovers.
Swap in any real dataset (e.g. Kaggle 'Superstore') by matching the column names.
"""
import numpy as np, pandas as pd
from pathlib import Path

rng = np.random.default_rng(42)
OUT = Path(__file__).resolve().parents[1] / "data" / "retail_orders_raw.csv"

cities = {
    "North": ["Delhi", "Meerut", "Lucknow", "Chandigarh", "Jaipur"],
    "South": ["Bengaluru", "Chennai", "Hyderabad", "Kochi"],
    "East": ["Kolkata", "Patna", "Bhubaneswar"],
    "West": ["Mumbai", "Pune", "Ahmedabad", "Surat"],
}
city_region = {c: r for r, cs in cities.items() for c in cs}
city_w = {"Delhi": 14, "Meerut": 3, "Lucknow": 4, "Chandigarh": 3, "Jaipur": 4, "Bengaluru": 13, "Chennai": 7,
          "Hyderabad": 8, "Kochi": 3, "Kolkata": 8, "Patna": 3, "Bhubaneswar": 2, "Mumbai": 14, "Pune": 6,
          "Ahmedabad": 5, "Surat": 3}
city_list = list(city_w); city_p = np.array(list(city_w.values()), float); city_p /= city_p.sum()

catalog = {  # category -> (sub-categories, (min price, max price), cost ratio)
    "Electronics": (["Mobiles", "Headphones", "Smartwatches", "Chargers", "Speakers"], (900, 45000), 0.78),
    "Furniture":   (["Chairs", "Tables", "Bookcases", "Storage"], (1500, 25000), 0.80),
    "Clothing":    (["Men's Wear", "Women's Wear", "Footwear", "Accessories"], (300, 5000), 0.55),
    "Grocery":     (["Staples", "Snacks", "Beverages", "Dairy"], (40, 900), 0.82),
    "Beauty":      (["Skincare", "Haircare", "Makeup", "Fragrance"], (150, 3500), 0.50),
}
products = []
for cat, (subs, (lo, hi), cr) in catalog.items():
    for sub in subs:
        for k in range(1, 5):
            price = float(np.round(np.exp(rng.uniform(np.log(lo), np.log(hi))), -1))
            products.append((cat, sub, f"{sub} Model {k}", price, cr))
prod_df = pd.DataFrame(products, columns=["category", "sub_category", "product_name", "unit_price", "cost_ratio"])
cat_w = {"Electronics": 0.24, "Furniture": 0.08, "Clothing": 0.28, "Grocery": 0.22, "Beauty": 0.18}
prod_df["w"] = prod_df["category"].map(cat_w) / prod_df.groupby("category")["product_name"].transform("count")
prod_df["w"] /= prod_df["w"].sum()
P_W = prod_df["w"].values

N_CUST = 3000
start, end = pd.Timestamp("2024-01-01"), pd.Timestamp("2025-12-31")
days = pd.date_range(start, end)

def season(d):
    w = 1.0
    if d.month in (10, 11): w *= 1.6   # festive season
    if d.month == 1: w *= 1.2          # new-year sale
    if d.month == 12: w *= 1.15
    if d.month in (6, 7): w *= 0.85
    if d.dayofweek >= 5: w *= 1.15
    return w
day_w = np.array([season(d) for d in days])

cust = pd.DataFrame({
    "customer_id": [f"C{str(i).zfill(5)}" for i in range(1, N_CUST + 1)],
    "customer_segment": rng.choice(["Consumer", "Corporate", "Home Office"], N_CUST, p=[0.55, 0.30, 0.15]),
    "city": rng.choice(city_list, N_CUST, p=city_p),
})
cust["join_idx"] = np.minimum(rng.exponential(250, N_CUST).astype(int), len(days) - 30)
cust["loyalty"] = rng.gamma(1.1, 1.2, N_CUST)   # latent propensity -> heavy-tailed order counts

rows, oid = [], 100000
for c in cust.itertuples():
    n_orders = 1 + rng.poisson(c.loyalty * (1.6 if c.customer_segment == "Corporate" else 1.0))
    avail = np.arange(c.join_idx, len(days))
    p = day_w[avail] / day_w[avail].sum()
    idxs = np.sort(rng.choice(avail, size=min(n_orders, len(avail)), replace=True, p=p))
    for ix in idxs:
        oid += 1
        odate = days[ix]
        ship_mode = rng.choice(["Standard", "Express", "Same Day"], p=[0.6, 0.3, 0.1])
        lag = {"Standard": int(rng.integers(3, 7)), "Express": int(rng.integers(1, 3)), "Same Day": 0}[ship_mode]
        sdate = odate + pd.Timedelta(days=lag)
        pay = rng.choice(["UPI", "Credit Card", "Debit Card", "COD", "Net Banking"], p=[0.45, 0.2, 0.12, 0.17, 0.06])
        for _ in range(rng.choice([1, 2, 3, 4], p=[0.5, 0.28, 0.15, 0.07])):
            pr = prod_df.iloc[rng.choice(len(prod_df), p=P_W)]
            qty = int(rng.choice([1, 2, 3, 4, 5], p=[0.55, 0.22, 0.12, 0.07, 0.04])) if pr.unit_price < 3000 else int(rng.choice([1, 2], p=[0.9, 0.1]))
            dprobs = {"Furniture": [0.20, 0.10, 0.15, 0.20, 0.20, 0.15],
                      "Electronics": [0.30, 0.15, 0.20, 0.20, 0.10, 0.05]}.get(pr.category, [0.50, 0.20, 0.15, 0.10, 0.04, 0.01])
            disc = float(rng.choice([0, 0.05, 0.10, 0.20, 0.30, 0.40], p=dprobs))
            if odate.month in (10, 11) and disc < 0.4 and rng.random() < 0.25: disc = round(min(disc + 0.10, 0.40), 2)
            sales = pr.unit_price * qty * (1 - disc)
            cost = pr.unit_price * pr.cost_ratio * qty * rng.uniform(0.97, 1.05)
            ship_cost = {"Standard": 0, "Express": 40, "Same Day": 90}[ship_mode] * (4 if pr.category == "Furniture" else 1)
            rows.append([f"ORD-{oid}", odate, sdate, c.customer_id, c.customer_segment, city_region[c.city], c.city,
                         pr.category, pr.sub_category, pr.product_name, pr.unit_price, qty, disc, round(sales, 2),
                         round(sales - cost - ship_cost, 2), pay, ship_mode])

df = pd.DataFrame(rows, columns=["order_id", "order_date", "ship_date", "customer_id", "customer_segment", "region", "city",
                                 "category", "sub_category", "product_name", "unit_price", "quantity", "discount", "sales",
                                 "profit", "payment_mode", "ship_mode"])

# ---- plant realistic data-quality issues (so the notebook has a real cleaning section) ----
n = len(df)
df = pd.concat([df, df.sample(int(0.006 * n), random_state=1)], ignore_index=True)                    # duplicates
df.loc[df.sample(frac=0.012, random_state=2).index, "city"] = np.nan                                  # missing city
idx = df.sample(frac=0.02, random_state=3).index; df.loc[idx, "category"] = df.loc[idx, "category"].str.upper()
df.loc[df.sample(frac=0.015, random_state=4).index, "payment_mode"] = " upi "                          # whitespace / case
df.loc[df.sample(frac=0.004, random_state=5).index, "quantity"] = -1                                   # invalid qty
df["order_date"] = df["order_date"].dt.strftime("%d-%m-%Y"); df["ship_date"] = df["ship_date"].dt.strftime("%d-%m-%Y")
df = df.sample(frac=1, random_state=7).reset_index(drop=True)
df.to_csv(OUT, index=False)
print(f"Saved {len(df):,} rows, {df.customer_id.nunique():,} customers -> {OUT}")
