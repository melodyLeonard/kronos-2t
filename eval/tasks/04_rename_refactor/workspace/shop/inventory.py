"""Stock levels and prices."""

from .pricing import calc_total


class Inventory:
    def __init__(self):
        self._stock = {}
        self._price = {}

    def add_product(self, sku, unit_cents, qty):
        self._price[sku] = unit_cents
        self._stock[sku] = self._stock.get(sku, 0) + qty

    def has(self, sku, qty):
        return self._stock.get(sku, 0) >= qty

    def price(self, sku):
        return self._price[sku]

    def take(self, sku, qty):
        if not self.has(sku, qty):
            raise ValueError("insufficient stock")
        self._stock[sku] -= qty

    def stock_value(self):
        """Untaxed, undiscounted value of all stock on hand."""
        lines = [(self._price[s], q) for s, q in self._stock.items()]
        return calc_total(lines, 0, tax=False)
