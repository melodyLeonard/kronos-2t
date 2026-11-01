import importlib.util, sys
from datetime import date
wd = sys.argv[1]
spec = importlib.util.spec_from_file_location("ledger", wd + "/ledger.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
fails = []
def eq(name, got, want):
    if got != want: fails.append(f"{name}: got {got!r} want {want!r}")
L = m.Ledger()
d1, d2, d3 = date(2026, 1, 1), date(2026, 1, 2), date(2026, 1, 3)
L.deposit("a", 2000, d1)
L.transfer("a", "b", 1000, d2)
eq("transfer_src", L.balance("a"), 975)
eq("transfer_dst", L.balance("b"), 1000)
eq("fees", L.total_fees(), 25)
eq("asof_inclusive", L.balance("a", as_of=d2), 975)
eq("asof_before", L.balance("a", as_of=d1), 2000)
eq("asof_after", L.balance("a", as_of=d3), 975)
L.deposit("a", 100, d3)
eq("asof_excl_future", L.balance("a", as_of=d2), 975)
try:
    L.withdraw("b", 5000, d3); fails.append("overdraft allowed")
except m.LedgerError:
    pass
st = L.statement("a")
eq("statement_len", len(st), 4)
eq("statement_running", st[-1][2], 1075)
eq("statement_order", [r[0] for r in st], ["2026-01-01", "2026-01-02", "2026-01-02", "2026-01-03"])
eq("fmt1", m.fmt_cents(-5), "-$0.05")
eq("fmt2", m.fmt_cents(123456), "$1234.56")
eq("fmt3", m.fmt_cents(0), "$0.00")
eq("fmt4", m.fmt_cents(-100), "-$1.00")
print("\n".join(fails) if fails else "all checks passed")
sys.exit(1 if fails else 0)
