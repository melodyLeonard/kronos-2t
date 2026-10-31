"""Small text helpers used by the docs pipeline."""

import re


def slugify(title):
    """Lowercase, replace runs of non-alphanumerics with '-', strip edge dashes."""
    s = title.lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return s.lstrip("-")


def truncate(text, limit):
    """Return at most `limit` characters; add '...' when cut (the dots count toward limit)."""
    if len(text) <= limit:
        return text
    if limit <= 3:
        return "." * limit
    return text[: limit - 2] + "..."


def word_freq(text):
    """Case-insensitive word counts, most frequent first, ties alphabetical."""
    counts = {}
    for w in re.findall(r"[A-Za-z']+", text):
        counts[w] = counts.get(w, 0) + 1
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))


def wrap(text, width):
    """Greedy word wrap into lines no longer than `width`."""
    lines, cur = [], []
    cur_len = 0
    for w in text.split():
        extra = len(w) if not cur else len(w) + 1
        if cur and cur_len + extra > width:
            lines.append(" ".join(cur))
            cur, cur_len = [w], len(w)
        else:
            cur.append(w)
            cur_len += extra
    if cur:
        lines.append(" ".join(cur))
    return lines


def title_case(text):
    """Capitalize each word except small joining words (not at the start)."""
    small = {"a", "an", "the", "of", "and", "or", "in", "on", "to"}
    out = []
    for i, w in enumerate(text.split()):
        lw = w.lower()
        if i > 0 and lw in small:
            out.append(lw)
        else:
            out.append(lw[:1].upper() + lw[1:])
    return " ".join(out)
