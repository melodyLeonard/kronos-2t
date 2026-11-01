import os, subprocess, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from expected import report
wd = os.path.abspath(sys.argv[1]); here = os.path.dirname(os.path.abspath(__file__))
fails = []
for name, csvp in [("sales.csv", os.path.join(wd, "sales.csv")), ("sales_alt.csv", os.path.join(here, "sales_alt.csv"))]:
    r = subprocess.run([sys.executable, "-I", os.path.join(wd, "report.py"), csvp], cwd=wd, capture_output=True, text=True, timeout=60)
    want = report(csvp)
    if r.returncode != 0: fails.append(f"{name}: rc={r.returncode} {r.stderr[-400:]}")
    elif r.stdout != want:
        import difflib
        d = "".join(list(difflib.unified_diff(want.splitlines(True), r.stdout.splitlines(True), "expected", "got"))[:40])
        fails.append(f"{name}: output mismatch\n{d}")
print("\n".join(fails) if fails else "all checks passed")
sys.exit(1 if fails else 0)
