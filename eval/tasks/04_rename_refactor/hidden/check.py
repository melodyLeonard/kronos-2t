import os, re, subprocess, sys
wd = sys.argv[1]
fails = []
sys.path.insert(0, wd)
try:
    import shop.pricing as p, shop.cart as c, shop.inventory as i, shop.main as mn
    if not hasattr(p, "compute_total"): fails.append("pricing.compute_total missing")
    if hasattr(p, "calc_total"): fails.append("pricing.calc_total still defined")
    if p.compute_total([(150, 4), (1200, 1)]) != 1944: fails.append("compute_total value wrong")
    if p.compute_total([(1000, 1)], 10, tax=False) != 900: fails.append("discount wrong")
    inv = i.Inventory(); inv.add_product("pen", 150, 10); inv.add_product("book", 1200, 2)
    cart = c.Cart(inv); cart.add("pen", 4); cart.add("book", 1)
    if cart.total() != 1944: fails.append(f"cart.total {cart.total()}")
    if inv.stock_value() != 3900: fails.append(f"stock_value {inv.stock_value()}")
    if cart.checkout() != 1944: fails.append("checkout total wrong")
    if not inv.has("pen", 6) or inv.has("pen", 7): fails.append("stock not taken")
except Exception as e:
    fails.append(f"import/run error: {type(e).__name__}: {e}")
for root, _, files in os.walk(os.path.join(wd, "shop")):
    for f in files:
        if f.endswith(".py"):
            src = open(os.path.join(root, f), encoding="utf-8").read()
            if re.search(r"\bcalc_total\b", src): fails.append(f"calc_total still in {f}")
r = subprocess.run([sys.executable, "-m", "shop.main"], cwd=wd, capture_output=True, text=True)
if r.returncode != 0 or "cart total: 1944" not in r.stdout: fails.append(f"shop.main: rc={r.returncode} {r.stderr[-300:]}")
print("\n".join(fails) if fails else "all checks passed")
sys.exit(1 if fails else 0)
