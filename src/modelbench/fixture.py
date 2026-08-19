"""Fixture database and samples for internal development testing.

This module provides a small, self-contained SQLite database and a set
of predefined Text-to-SQL samples used by the ``evaluate-fixture`` CLI
command.

**This is NOT a benchmark like Spider or BIRD.**  It exists solely to
verify that the evaluation pipeline works end-to-end during development.

Database schema (e-commerce):

    customers(customer_id, name)
    products(product_id, name, price)
    orders(order_id, customer_id, order_date)
    order_items(order_id, product_id, quantity, unit_price)

Data is deterministic and small enough to verify by hand.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from modelbench.types import TextToSQLSample

# ── Schema ──────────────────────────────────────────────────────────

_SCHEMA = """\
CREATE TABLE customers (
    customer_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL
);

CREATE TABLE products (
    product_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    price REAL NOT NULL
);

CREATE TABLE orders (
    order_id INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL,
    order_date TEXT NOT NULL,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

CREATE TABLE order_items (
    order_id INTEGER NOT NULL,
    product_id INTEGER NOT NULL,
    quantity INTEGER NOT NULL,
    unit_price REAL NOT NULL,
    PRIMARY KEY (order_id, product_id),
    FOREIGN KEY (order_id) REFERENCES orders(order_id),
    FOREIGN KEY (product_id) REFERENCES products(product_id)
);
"""

# ── Deterministic test data ─────────────────────────────────────────
#
# 3 customers:  Alice, Bob, Charlie
# 3 products:   Widget ($9.99), Gadget ($24.99), Doohickey ($4.99)
# 3 orders:     Alice has 2 orders, Bob has 1 order, Charlie has 0 orders
# 5 order items:
#   Order 1 (Alice, 2024-01-15): Widget x2, Doohickey x1
#   Order 2 (Bob,   2024-01-20): Gadget x1
#   Order 3 (Alice, 2024-02-10): Widget x3, Gadget x1

_DATA = """\
INSERT INTO customers VALUES (1, 'Alice');
INSERT INTO customers VALUES (2, 'Bob');
INSERT INTO customers VALUES (3, 'Charlie');

INSERT INTO products VALUES (1, 'Widget', 9.99);
INSERT INTO products VALUES (2, 'Gadget', 24.99);
INSERT INTO products VALUES (3, 'Doohickey', 4.99);

INSERT INTO orders VALUES (1, 1, '2024-01-15');
INSERT INTO orders VALUES (2, 2, '2024-01-20');
INSERT INTO orders VALUES (3, 1, '2024-02-10');

INSERT INTO order_items VALUES (1, 1, 2, 9.99);
INSERT INTO order_items VALUES (1, 3, 1, 4.99);
INSERT INTO order_items VALUES (2, 2, 1, 24.99);
INSERT INTO order_items VALUES (3, 1, 3, 9.99);
INSERT INTO order_items VALUES (3, 2, 1, 24.99);
"""


def create_fixture_db(path: str | Path) -> Path:
    """Create the fixture e-commerce SQLite database.

    If a file already exists at ``path`` it is deleted first to ensure
    a clean state.  Parent directories are created as needed.

    Args:
        path: Where to write the database file.

    Returns:
        The resolved path to the created database.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()

    conn = sqlite3.connect(str(path))
    conn.executescript(_SCHEMA)
    conn.executescript(_DATA)
    conn.close()
    return path


# ── Fixture samples ─────────────────────────────────────────────────
#
# Each sample pairs a natural-language question with a gold SQL query.
# FIXTURE_PREDICTIONS contains the corresponding "predicted" SQL for
# the evaluate-fixture CLI command.
#
# The predictions are designed to demonstrate the difference between
# exact-match and execution-accuracy metrics:
#   - Samples 1 & 5:  predicted SQL matches gold exactly.
#   - Samples 2-4:    predicted SQL differs textually but produces
#                      the same result (different operand order,
#                      subquery vs JOIN, different COUNT expression).


def get_fixture_samples(db_path: str | Path) -> list[TextToSQLSample]:
    """Return the predefined fixture evaluation samples.

    These are used by ``modelbench evaluate-fixture`` and by tests.
    This is **not** a real benchmark — see module docstring.
    """
    db = str(db_path)
    return [
        TextToSQLSample(
            question="How many customers are there?",
            db_id="fixture_ecommerce",
            db_path=db,
            gold_sql="SELECT COUNT(*) FROM customers",
        ),
        TextToSQLSample(
            question="What is the total revenue from all orders?",
            db_id="fixture_ecommerce",
            db_path=db,
            gold_sql="SELECT SUM(quantity * unit_price) AS total FROM order_items",
        ),
        TextToSQLSample(
            question="Which customers have placed at least one order?",
            db_id="fixture_ecommerce",
            db_path=db,
            gold_sql=(
                "SELECT DISTINCT c.name "
                "FROM customers c "
                "JOIN orders o ON c.customer_id = o.customer_id"
            ),
        ),
        TextToSQLSample(
            question="How many orders has each customer placed?",
            db_id="fixture_ecommerce",
            db_path=db,
            gold_sql=(
                "SELECT c.name, COUNT(*) AS order_count "
                "FROM customers c "
                "JOIN orders o ON c.customer_id = o.customer_id "
                "GROUP BY c.customer_id"
            ),
        ),
        TextToSQLSample(
            question="List all product names and prices.",
            db_id="fixture_ecommerce",
            db_path=db,
            gold_sql="SELECT name, price FROM products",
        ),
    ]


FIXTURE_PREDICTIONS: list[str] = [
    # 1. Exact match — identical SQL
    "SELECT COUNT(*) FROM customers",
    # 2. Different operand order in SUM — same result
    "SELECT SUM(unit_price * quantity) AS total FROM order_items",
    # 3. Subquery instead of JOIN — same result
    ("SELECT DISTINCT name FROM customers WHERE customer_id IN (SELECT customer_id FROM orders)"),
    # 4. COUNT(column) + GROUP BY name instead of id — same result
    (
        "SELECT c.name, COUNT(o.order_id) AS order_count "
        "FROM customers c "
        "JOIN orders o ON c.customer_id = o.customer_id "
        "GROUP BY c.name"
    ),
    # 5. Exact match — identical SQL
    "SELECT name, price FROM products",
]
