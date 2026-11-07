from collections import deque
from contextlib import closing
from dataclasses import dataclass
from datetime import date
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DATABASE = DATA / "lisbon.sqlite3"
OUTPUT = ROOT / "output"
SNAPSHOT = "2026-09-19"
SOURCE = f"https://data.insideairbnb.com/portugal/lisbon/lisbon/{SNAPSHOT}/data"
MATCH_FIELDS = ["district", "room_type", "capacity"]
MIN_PEERS = 15
REPLAY_LIMIT = 40
WINDOW_DAYS = 90
MIN_TEXTS = 5
MIN_FLAGS = 2
FLAG_PERCENT = 20

CATEGORIES = (
    "Значительно ниже рынка", "Ниже рынка", "Типичный диапазон",
    "Выше рынка", "Значительно выше рынка",
)
ROOM_LABELS = {
    "Entire home/apt": "Жильё целиком", "Private room": "Отдельная комната",
    "Hotel room": "Номер в отеле", "Shared room": "Общая комната",
}


@dataclass(frozen=True)
class Benchmarks:
    p10: int
    p25: int
    median: int
    p75: int
    p90: int

    def category(self, price):
        if price < self.p10:
            return 0
        if price < self.p25:
            return 1
        if price <= self.p75:
            return 2
        if price <= self.p90:
            return 3
        return 4

    def nearest_typical_edge(self, price):
        edge = min((self.p25, self.p75), key=lambda value: abs(value - price))
        return ("P25" if edge == self.p25 else "P75"), edge - price


def get_benchmarks(prices):
    if len(prices) < MIN_PEERS:
        return None
    quantiles = np.percentile(np.asarray(prices, dtype=np.int64), [10, 25, 50, 75, 90])
    return Benchmarks(*(int(round(value)) for value in quantiles))


def select_peers(listings, district, room_type, capacity, exclude_id=None):
    mask = (
        listings["district"].eq(district) & listings["room_type"].eq(room_type)
        & listings["capacity"].eq(capacity) & listings["price_cents"].notna()
    )
    if exclude_id is not None:
        mask &= listings["listing_id"].ne(exclude_id)
    return listings.loc[mask, "price_cents"]


def open_database():
    return closing(sqlite3.connect(DATABASE.as_uri() + "?mode=ro", uri=True))


def load_listings(db):
    return pd.read_sql_query("SELECT * FROM listings", db, dtype={"price_cents": "Int64"})


@dataclass(frozen=True)
class Review:
    review_id: int
    day: date
    has_text: bool
    evidence: str
    body: str = ""


def load_reviews(db, listing_id):
    rows = db.execute(
        "SELECT review_id, day, has_text, evidence, body FROM reviews WHERE listing_id = ? "
        "ORDER BY day DESC, review_id DESC LIMIT ?", (int(listing_id), REPLAY_LIMIT),
    ).fetchall()
    return [Review(review_id, date.fromisoformat(day), bool(has_text), evidence, body)
            for review_id, day, has_text, evidence, body in reversed(rows)]


@dataclass(frozen=True)
class ReviewState:
    texts: int
    flags: int
    active: bool
    fired: bool
    decision_window: tuple[Review, ...] = ()

    @property
    def share(self):
        return self.flags / self.texts if self.texts else None


class Replay:
    def __init__(self, price_cents, benchmarks):
        self.above_typical = benchmarks is not None and price_cents > benchmarks.p75
        self.window = deque()
        self.active = False

    def push(self, event):
        while self.window and (event.day - self.window[0].day).days >= WINDOW_DAYS:
            self.window.popleft()
        if event.has_text:
            self.window.append(event)
        texts = len(self.window)
        flags = sum(bool(item.evidence) for item in self.window)
        active = (self.above_typical and texts >= MIN_TEXTS and flags >= MIN_FLAGS
                  and 100 * flags >= FLAG_PERCENT * texts)
        fired = active and not self.active
        result = ReviewState(texts, flags, active, fired, tuple(self.window) if fired else ())
        self.active = active
        return result
