import unittest
from pyqueue import PriorityQueue, RateLimiter, parse_duration, Scheduler, merge_sorted


class TestPriorityQueue(unittest.TestCase):
    def test_fifo_on_ties(self):
        q = PriorityQueue()
        q.push("a", 1); q.push("b", 1); q.push("c", 0)
        self.assertEqual([q.pop(), q.pop(), q.pop()], ["c", "a", "b"])

    def test_unorderable_items_with_equal_priority(self):
        q = PriorityQueue()
        q.push({"x": 1}, 2); q.push({"y": 2}, 2)
        self.assertEqual(q.pop(), {"x": 1})
        self.assertEqual(len(q), 1)


class TestRateLimiter(unittest.TestCase):
    def test_limit_is_inclusive(self):
        rl = RateLimiter(limit=2, window=10)
        self.assertTrue(rl.allow(0)); self.assertTrue(rl.allow(1))
        self.assertFalse(rl.allow(2))
        self.assertTrue(rl.allow(11))


class TestParseDuration(unittest.TestCase):
    def test_compound(self):
        self.assertEqual(parse_duration("1h30m"), 5400)
        self.assertEqual(parse_duration("250ms"), 0.25)
        self.assertEqual(parse_duration("2d 4h"), 2 * 86400 + 4 * 3600)

    def test_zero_is_valid_and_garbage_rejected(self):
        self.assertEqual(parse_duration("0s"), 0)
        with self.assertRaises(ValueError):
            parse_duration("soon")
        with self.assertRaises(ValueError):
            parse_duration("5 parsecs")


class TestScheduler(unittest.TestCase):
    def test_due_now_runs_and_runs_once(self):
        s = Scheduler()
        s.add("b", 5); s.add("a", 5); s.add("c", 9)
        self.assertEqual(s.run_due(5), ["a", "b"])
        self.assertEqual(s.run_due(5), [])
        self.assertEqual(s.pending(), ["c"])


class Rec:
    """Ordered by key only, so equal keys are ties."""

    def __init__(self, k, tag):
        self.k, self.tag = k, tag

    def __lt__(self, other):
        return self.k < other.k

    def __le__(self, other):
        return self.k <= other.k

    def __repr__(self):
        return f"Rec({self.k}, {self.tag!r})"


class TestMerge(unittest.TestCase):
    def test_stable_ties(self):
        a = [Rec(1, "a"), Rec(2, "a")]
        b = [Rec(1, "b"), Rec(3, "b")]
        self.assertEqual([r.tag for r in merge_sorted(a, b)], ["a", "b", "a", "b"])


if __name__ == "__main__":
    unittest.main()
