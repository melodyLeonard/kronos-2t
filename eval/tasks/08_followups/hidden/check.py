import importlib.util, sys
wd = sys.argv[1]
spec = importlib.util.spec_from_file_location("settings", wd + "/settings.py")
m = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(m)
except Exception as e:
    print(f"settings.py failed to import: {type(e).__name__}: {e}"); sys.exit(1)
want = {
    "app_name": "exporter-service", "version": "1.2.0", "port": 9090, "host": "0.0.0.0", "workers": 4,
    "timeout_secs": 60, "max_retries": 5, "log_level": "DEBUG", "max_batch": 1000,
    "features": ["csv", "json", "parquet"], "rate_limit_per_min": 1200, "region": "eu-west-1",
    "metrics_enabled": True, "shutdown_grace_secs": 15,
}
got = dict(m.SETTINGS)
fails = []
for k, v in want.items():
    if k not in got: fails.append(f"missing {k}")
    elif got[k] != v or type(got[k]) is not type(v): fails.append(f"{k}: got {got[k]!r} want {v!r}")
for k in got:
    if k not in want: fails.append(f"unexpected key {k}")
keys = list(got)
if "port" in keys and "host" in keys and keys.index("host") != keys.index("port") + 1: fails.append("host not right after port")
if not callable(getattr(m, "get", None)) or m.get("port") != 9090: fails.append("get() broken")
print("\n".join(fails) if fails else f"all checks passed ({len(want)} keys)")
sys.exit(1 if fails else 0)
