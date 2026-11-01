import hashlib, os, subprocess, sys
wd = os.path.abspath(sys.argv[1]); here = os.path.dirname(os.path.abspath(__file__))
fails = []
orig = open(os.path.join(here, "..", "workspace", "test_queue.py"), "rb").read()
cur = open(os.path.join(wd, "test_queue.py"), "rb").read() if os.path.exists(os.path.join(wd, "test_queue.py")) else b""
if hashlib.sha256(orig).hexdigest() != hashlib.sha256(cur).hexdigest(): fails.append("test_queue.py was modified")
r = subprocess.run([sys.executable, "-m", "unittest", "test_queue"], cwd=wd, capture_output=True, text=True, timeout=120)
if r.returncode != 0: fails.append("visible tests fail:\n" + r.stderr[-600:])
sys.path.insert(0, wd)
try:
    from pyqueue import PriorityQueue, RateLimiter, parse_duration, Scheduler, merge_sorted
    q = PriorityQueue(); [q.push(i, 5) for i in range(50)]
    if [q.pop() for _ in range(50)] != list(range(50)): fails.append("hidden: fifo order over 50 ties")
    rl = RateLimiter(3, 1.0)
    if [rl.allow(t) for t in (0, 0.2, 0.4, 0.6, 0.99, 1.21)] != [True, True, True, False, False, True]: fails.append("hidden: sliding window")
    if parse_duration("1m 0s") != 60 or parse_duration("0ms") != 0: fails.append("hidden: durations")
    try: parse_duration(""); fails.append("hidden: empty duration accepted")
    except ValueError: pass
    s = Scheduler(); s.add("x", 1); s.add("y", 2)
    if s.run_due(1) != ["x"] or s.run_due(2) != ["y"] or s.pending() != []: fails.append("hidden: scheduler")
    if merge_sorted([], [1]) != [1] or merge_sorted([2], []) != [2] or merge_sorted([1, 1], [1]) != [1, 1, 1]: fails.append("hidden: merge edges")
except Exception as e:
    fails.append(f"hidden import/run error: {type(e).__name__}: {e}")
print("\n".join(fails) if fails else "all checks passed")
sys.exit(1 if fails else 0)
