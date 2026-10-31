"""Command-line demo."""

from .cart import Cart
from .inventory import Inventory
from .pricing import calc_total


def demo():
    inv = Inventory()
    inv.add_product("pen", 150, 100)
    inv.add_product("book", 1200, 10)
    cart = Cart(inv)
    cart.add("pen", 4)
    cart.add("book", 1)
    print("cart total:", cart.total())
    print("direct:", calc_total([(150, 4), (1200, 1)]))
    print("stock value:", inv.stock_value())
    return cart.checkout()


if __name__ == "__main__":
    demo()
