import ast, os, sys
wd = sys.argv[1]
sys.path.insert(0, wd)
fails = []
try:
    import validators as v, app
    for fn in ["validate_email", "validate_age", "normalize_name", "validate_username"]:
        if not hasattr(v, fn): fails.append(f"validators.{fn} missing")
        elif getattr(app, fn, None) is not getattr(v, fn): fails.append(f"app.{fn} is not validators.{fn}")
    for c in ["EMAIL_RE", "MIN_AGE", "MAX_AGE"]:
        if not hasattr(v, c): fails.append(f"validators.{c} missing")
    if v.validate_email(" Foo@Bar.COM ") != "foo@bar.com": fails.append("validate_email")
    if v.validate_age("42") != 42: fails.append("validate_age")
    if v.normalize_name("  ann   LEE ") != "Ann Lee": fails.append("normalize_name")
    if v.validate_username("ann_1") != "ann_1": fails.append("validate_username")
    for bad in [lambda: v.validate_email("nope"), lambda: v.validate_age(5), lambda: v.validate_username("A")]:
        try: bad(); fails.append("bad input accepted")
        except ValueError: pass
    u = app.build_user({"username": "ann_1", "email": "A@B.io", "age": "30", "name": "ann  lee"})
    if u != {"username": "ann_1", "email": "a@b.io", "age": 30, "name": "Ann Lee"}: fails.append(f"build_user {u}")
    s = app.UserStore(); s.add(u)
    if "ann_1" not in s.dump(): fails.append("UserStore.dump")
    tree = ast.parse(open(os.path.join(wd, "app.py"), encoding="utf-8").read())
    defs = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    dup = defs & {"validate_email", "validate_age", "normalize_name", "validate_username"}
    if dup: fails.append(f"duplicate defs in app.py: {sorted(dup)}")
    if not any(isinstance(n, ast.ImportFrom) and n.module == "validators" for n in tree.body): fails.append("app.py lacks 'from validators import'")
except Exception as e:
    fails.append(f"error: {type(e).__name__}: {e}")
print("\n".join(fails) if fails else "all checks passed")
sys.exit(1 if fails else 0)
