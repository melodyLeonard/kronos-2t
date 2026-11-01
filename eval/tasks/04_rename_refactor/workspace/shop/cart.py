"""Shopping cart."""

from .pricing import calc_total
from .inventory import Inventory


class Cart:
    def __init__(self, inventory: Inventory):
        self.inventory = inventory
        self.items = {}

    def add(self, sku, qty=1):
        if not self.inventory.has(sku, qty + self.items.get(sku, 0)):
            raise ValueError(f"not enough stock for {sku}")
        self.items[sku] = self.items.get(sku, 0) + qty

    def remove(self, sku):
        self.items.pop(sku, None)

    def lines(self):
        return [(self.inventory.price(sku), qty) for sku, qty in sorted(self.items.items())]

    def total(self, discount_percent=0, tax=True):
        return calc_total(self.lines(), discount_percent, tax)

    def checkout(self):
        for sku, qty in self.items.items():
            self.inventory.take(sku, qty)
        total = calc_total(self.lines())
        self.items.clear()
        return total
