"""User sign-up handling for a CLI tool."""

import json
import re
import sys

EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")
MIN_AGE = 13
MAX_AGE = 120


def validate_email(email):
    """Return the lowercased email or raise ValueError."""
    e = email.strip().lower()
    if not EMAIL_RE.match(e):
        raise ValueError(f"invalid email: {email!r}")
    return e


def validate_age(age):
    """Accept int or numeric string within [MIN_AGE, MAX_AGE]."""
    try:
        a = int(age)
    except (TypeError, ValueError):
        raise ValueError(f"age must be an integer: {age!r}")
    if not MIN_AGE <= a <= MAX_AGE:
        raise ValueError(f"age out of range: {a}")
    return a


def normalize_name(name):
    """Collapse whitespace and title-case each part."""
    parts = [p for p in name.strip().split() if p]
    if not parts:
        raise ValueError("name is empty")
    return " ".join(p[:1].upper() + p[1:].lower() for p in parts)


def validate_username(username):
    u = username.strip()
    if not re.fullmatch(r"[a-z][a-z0-9_]{2,15}", u):
        raise ValueError(f"invalid username: {username!r}")
    return u


class UserStore:
    def __init__(self):
        self._users = {}

    def add(self, user):
        if user["username"] in self._users:
            raise ValueError("duplicate username")
        self._users[user["username"]] = user

    def get(self, username):
        return self._users.get(username)

    def dump(self):
        return json.dumps(sorted(self._users.values(), key=lambda u: u["username"]))


def build_user(raw):
    """Validate a raw dict and return a clean user record."""
    return {
        "username": validate_username(raw.get("username", "")),
        "email": validate_email(raw.get("email", "")),
        "age": validate_age(raw.get("age")),
        "name": normalize_name(raw.get("name", "")),
    }


def main(argv):
    store = UserStore()
    for line in argv:
        try:
            store.add(build_user(json.loads(line)))
        except ValueError as e:
            print(f"rejected: {e}", file=sys.stderr)
    print(store.dump())


if __name__ == "__main__":
    main(sys.argv[1:])
