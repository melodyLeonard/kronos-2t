"""In-memory double-entry style ledger for a small shop."""

from dataclasses import dataclass, field
from datetime import date


class LedgerError(Exception):
    pass


@dataclass(order=True)
class Entry:
    when: date
    seq: int
    account: str = field(compare=False)
    amount_cents: int = field(compare=False)
    memo: str = field(compare=False, default="")


class Ledger:
    TRANSFER_FEE_CENTS = 25

    def __init__(self):
        self._entries = []
        self._seq = 0

    def _add(self, when, account, amount_cents, memo=""):
        self._seq += 1
        e = Entry(when, self._seq, account, amount_cents, memo)
        self._entries.append(e)
        return e

    def deposit(self, account, amount_cents, when, memo=""):
        if amount_cents <= 0:
            raise LedgerError("deposit must be positive")
        return self._add(when, account, amount_cents, memo)

    def withdraw(self, account, amount_cents, when, memo=""):
        if amount_cents <= 0:
            raise LedgerError("withdrawal must be positive")
        if self.balance(account) < amount_cents:
            raise LedgerError("insufficient funds")
        return self._add(when, account, -amount_cents, memo)

    def transfer(self, src, dst, amount_cents, when, memo=""):
        """Move money between accounts; the source pays a flat fee once."""
        fee = self.TRANSFER_FEE_CENTS
        self.withdraw(src, amount_cents + fee, when, memo or f"transfer to {dst}")
        self._add(when, src, -fee, "transfer fee")
        self.deposit(dst, amount_cents, when, memo or f"transfer from {src}")

    def balance(self, account, as_of=None):
        """Balance of `account` including entries dated on or before `as_of`."""
        total = 0
        for e in self._entries:
            if e.account != account:
                continue
            if as_of is not None and e.when >= as_of:
                continue
            total += e.amount_cents
        return total

    def statement(self, account):
        """Entries for `account`, oldest first, with a running balance."""
        rows = []
        running = 0
        for e in sorted(e for e in self._entries if e.account == account):
            running += e.amount_cents
            rows.append((e.when.isoformat(), e.amount_cents, running, e.memo))
        return rows

    def accounts(self):
        return sorted({e.account for e in self._entries})

    def total_fees(self):
        return -sum(e.amount_cents for e in self._entries if e.memo == "transfer fee")


def fmt_cents(cents):
    sign = "-" if cents < 0 else ""
    cents = abs(cents)
    return f"{sign}${cents // 100}.{cents % 100}"
