"""Small scheduling utilities."""

import heapq
import re


class PriorityQueue:
    """Min-priority queue; equal priorities pop in insertion order."""

    def __init__(self):
        self._heap = []
        self._n = 0

    def push(self, item, priority):
        heapq.heappush(self._heap, (priority, item, self._n))
        self._n += 1

    def pop(self):
        if not self._heap:
            raise IndexError("pop from empty queue")
        return heapq.heappop(self._heap)[1]

    def peek(self):
        return self._heap[0][1] if self._heap else None

    def __len__(self):
        return len(self._heap)


class RateLimiter:
    """Allow at most `limit` events per `window` seconds (sliding window)."""

    def __init__(self, limit, window):
        self.limit = limit
        self.window = window
        self._times = []

    def allow(self, now):
        cutoff = now - self.window
        self._times = [t for t in self._times if t > cutoff]
        if len(self._times) > self.limit:
            return False
        self._times.append(now)
        return True


_DUR_RE = re.compile(r"(\d+)\s*(ms|s|m|h|d)")
_UNITS = {"ms": 0.001, "s": 1, "m": 60, "h": 3600, "d": 86400}


def parse_duration(text):
    """Parse strings like '1h30m', '250ms', '2d 4h' into seconds (float)."""
    total = 0.0
    pos = 0
    for m in _DUR_RE.finditer(text):
        total += int(m.group(1)) * _UNITS[m.group(2)]
        pos = m.end()
    if total == 0.0 or text.strip() == "":
        raise ValueError(f"bad duration: {text!r}")
    return total


class Scheduler:
    """Runs jobs whose `due` time has passed, in due order, each at most once."""

    def __init__(self):
        self._jobs = []

    def add(self, name, due):
        self._jobs.append((due, name))

    def run_due(self, now):
        ran = []
        for due, name in sorted(self._jobs):
            if due < now:
                ran.append(name)
        self._jobs = [(d, n) for d, n in self._jobs if n not in ran]
        return ran

    def pending(self):
        return [n for _, n in sorted(self._jobs)]


def merge_sorted(a, b):
    """Merge two ascending lists into one ascending list (stable: a before b on ties)."""
    out = []
    i = j = 0
    while i < len(a) and j < len(b):
        if a[i] < b[j]:
            out.append(a[i])
            i += 1
        else:
            out.append(b[j])
            j += 1
    out.extend(a[i:])
    out.extend(b[j:])
    return out
