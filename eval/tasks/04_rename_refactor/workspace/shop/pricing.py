"""Price rules."""

TAX_RATE = 0.08


def line_price(unit_cents, qty):
    return unit_cents * qty


def apply_discount(cents, percent):
    if not 0 <= percent <= 100:
        raise ValueError("percent out of range")
    return round(cents * (100 - percent) / 100)


def calc_total(lines, discount_percent=0, tax=True):
    """lines: iterable of (unit_cents, qty). Returns total in cents after discount and tax."""
    subtotal = sum(line_price(u, q) for u, q in lines)
    discounted = apply_discount(subtotal, discount_percent)
    if tax:
        return round(discounted * (1 + TAX_RATE))
    return discounted
