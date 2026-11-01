import hashlib, importlib.util, os, re, sys
wd = os.path.abspath(sys.argv[1]); here = os.path.dirname(os.path.abspath(__file__))
fails = []
spec = importlib.util.spec_from_file_location("analytics", os.path.join(wd, "analytics.py"))
m = importlib.util.module_from_spec(spec)
try: spec.loader.exec_module(m)
except Exception as e: print(f"import failed: {e}"); sys.exit(1)
cases = [(([1, 2, 3, 4, 5], 3), [1.0, 1.5, 2.0, 3.0, 4.0]), (([10, 20, 30], 1), [10.0, 20.0, 30.0]), (([4, 8], 5), [4.0, 6.0]), (([], 2), []), (([3, 1, 4, 1, 5, 9, 2, 6], 2), [3.0, 2.0, 2.5, 2.5, 3.0, 7.0, 5.5, 4.0])]
for (args, want) in cases:
    got = m.rolling_average(*args)
    if got != want: fails.append(f"rolling_average{args}: got {got} want {want}")
try: m.rolling_average([1], 0); fails.append("window=0 accepted")
except ValueError: pass
orig = open(os.path.join(here, "..", "workspace", "analytics.py"), encoding="utf-8").read()
cur = open(os.path.join(wd, "analytics.py"), encoding="utf-8").read()
def strip_ra(s): return re.sub(r"def rolling_average\(.*?\n(?=def |\Z)", "", s, flags=re.S)
if strip_ra(orig) != strip_ra(cur): fails.append("code outside rolling_average changed")
if len(m.summarize([1, 2, 3])) != 135: fails.append("summarize changed")
print("\n".join(fails) if fails else "all checks passed")
sys.exit(1 if fails else 0)
