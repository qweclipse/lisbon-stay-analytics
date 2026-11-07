import csv
from datetime import date
import gzip
from hashlib import file_digest
import html
import json
import re
import sqlite3
from contextlib import closing
from tempfile import TemporaryDirectory

import pandas as pd

from pricing import DATA, DATABASE, OUTPUT, SNAPSHOT, SOURCE


RAW = DATA / "raw"
PRICE = re.compile(r"\$(?:\d+|\d{1,3}(?:,\d{3})+)\.\d{2}")
WORD = re.compile(r"[^\W\d_]", re.UNICODE)
BREAK = re.compile(r"<br\s*/?>", re.IGNORECASE)
VALUE_COMPLAINT = re.compile(
    r"\b(?:too expensive|overpriced|poor value for money|not worth (?:the|its) price|"
    r"car[oa] demais|demasiado car[oa]|não vale o preço|trop ch(?:er|ère)|"
    r"ne vaut pas (?:le prix|son prix)|(?:no vale|no merece) (?:el|su) precio)\b",
    re.IGNORECASE,
)
NEGATION = re.compile(r"\b(?:not|never|isn't|wasn't|não|no|nunca|pas)\s+$", re.IGNORECASE)


def price_cents(text):
    if not text:
        return None
    if PRICE.fullmatch(text) is None:
        raise ValueError(f"Unexpected listing price format: {text!r}")
    return int(text[1:].replace(",", "").replace(".", "")) or None


def normalize_review(raw):
    return " ".join(BREAK.sub(" ", html.unescape(raw)).replace("’", "'").split())


def price_match(text):
    for match in VALUE_COMPLAINT.finditer(text):
        start = match.start()
        if not NEGATION.search(text[max(0, start - 16):start]):
            return match
    return None


def detect_review(text):
    has_text = WORD.search(text) is not None
    match = price_match(text)
    if match is None:
        return has_text, "", ""
    start, end = match.span()
    left, right = max(0, start - 55), min(len(text), end + 55)
    return has_text, match.group(), ("…" if left else "") + text[left:right] + ("…" if right < len(text) else "")


def review_signal(raw):
    return detect_review(normalize_review(raw))


def import_listings(db, path):
    db.execute("""CREATE TABLE listings (
        listing_id INTEGER PRIMARY KEY, host_id INTEGER NOT NULL, title TEXT NOT NULL,
        district TEXT NOT NULL, room_type TEXT NOT NULL, capacity INTEGER NOT NULL,
        price_cents INTEGER, scraped TEXT NOT NULL, CHECK (capacity > 0))""")
    with gzip.open(path, "rt", encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    db.executemany("INSERT INTO listings VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (
        (int(row["id"]), int(row["host_id"]), row["name"], row["neighbourhood_cleansed"],
         row["room_type"], int(row["accommodates"]), price_cents(row["price"]),
         date.fromisoformat(row["last_scraped"]).isoformat()) for row in rows
    ))
    usable, low, high, first, last = db.execute(
        "SELECT COUNT(price_cents), MIN(price_cents), MAX(price_cents), MIN(scraped), MAX(scraped) FROM listings"
    ).fetchone()
    if first != SNAPSHOT or usable == 0:
        raise ValueError("The listings file does not match the expected Lisbon snapshot")
    return {"rows": len(rows), "prices_filled": sum(bool(row["price"]) for row in rows),
            "prices_usable": usable, "min_price_eur": low / 100, "max_price_eur": high / 100,
            "scraped_min": first, "scraped_max": last}


def import_reviews(db, path, listing_ids, cutoff):
    db.execute("""CREATE TABLE reviews (
        review_id INTEGER PRIMARY KEY, listing_id INTEGER NOT NULL,
        day TEXT NOT NULL, has_text INTEGER NOT NULL, phrase TEXT NOT NULL, evidence TEXT NOT NULL,
        author_id INTEGER, body TEXT NOT NULL)""")
    audit = {"rows": 0, "unknown_listing": 0, "invalid_event": 0, "after_scrape": 0}

    def event_rows():
        with gzip.open(path, "rt", encoding="utf-8", newline="") as source:
            for row in csv.DictReader(source):
                audit["rows"] += 1
                try:
                    listing_id, review_id = int(row["listing_id"]), int(row["id"])
                    day = date.fromisoformat(row["date"]).isoformat()
                    if min(listing_id, review_id) <= 0:
                        raise ValueError("Nonpositive ID")
                except (TypeError, ValueError):
                    audit["invalid_event"] += 1
                    continue
                if listing_id not in listing_ids:
                    audit["unknown_listing"] += 1
                    continue
                if day > cutoff:
                    audit["after_scrape"] += 1
                    continue
                author = row.get("reviewer_id") or ""
                author_id = int(author) if author.isascii() and author.isdecimal() else None
                body = normalize_review(row["comments"] or "")
                yield review_id, listing_id, day, *detect_review(body), author_id or None, body

    # A repeated ID attached to different listings or dates has no unambiguous owner.
    db.executemany("""INSERT INTO reviews VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(review_id) DO UPDATE SET listing_id = 0
        WHERE reviews.listing_id <> excluded.listing_id OR reviews.day <> excluded.day""", event_rows())
    ambiguous_ids = db.execute("DELETE FROM reviews WHERE listing_id = 0").rowcount
    db.execute("CREATE INDEX reviews_by_listing_day ON reviews (listing_id, day, review_id)")
    stored, texts, flags, first, last, with_author = db.execute(
        "SELECT COUNT(*), SUM(has_text), SUM(phrase <> ''), MIN(day), MAX(day), COUNT(author_id) FROM reviews"
    ).fetchone()
    if not stored:
        raise ValueError("No valid review events for the snapshot")
    duplicate_rows = (audit["rows"] - audit["unknown_listing"] - audit["invalid_event"]
                      - audit["after_scrape"] - stored - ambiguous_ids)
    return audit | {"stored": stored, "duplicate_rows": duplicate_rows, "ambiguous_ids": ambiguous_ids,
                    "with_text": texts, "flagged": flags, "min_date": first, "max_date": last,
                    "with_author": with_author}


def audit_calendar(path, listing_ids):
    headers = pd.read_csv(path, nrows=0).columns.tolist()
    prices = [column for column in ("price", "adjusted_price") if column in headers]
    columns = ["listing_id", "date", "available", *prices]
    totals = {"rows": 0, "available": 0, "unavailable": 0, "unknown_availability": 0,
              "unknown_listing_rows": 0, "price_columns": prices, "filled_daily_prices": 0}
    dates, ids = set(), set()
    for chunk in pd.read_csv(path, usecols=columns, chunksize=350_000, dtype={"listing_id": "int64", "available": str}):
        date_values = chunk["date"].unique()
        dates.update(date.fromisoformat(day) for day in date_values)
        ids.update(chunk["listing_id"].unique())
        available = int(chunk["available"].eq("t").sum())
        unavailable = int(chunk["available"].eq("f").sum())
        totals["rows"] += len(chunk)
        totals["available"] += available
        totals["unavailable"] += unavailable
        totals["unknown_availability"] += len(chunk) - available - unavailable
        totals["unknown_listing_rows"] += int((~chunk["listing_id"].isin(listing_ids)).sum())
        totals["filled_daily_prices"] += sum(int(chunk[column].notna().sum()) for column in prices)
    return totals | {"columns": headers, "listing_count": len(ids), "min_date": min(dates).isoformat(),
                     "max_date": max(dates).isoformat()}


def main():
    csv.field_size_limit(10_000_000)
    paths = {name: RAW / f"{name}.csv.gz" for name in ("listings", "reviews", "calendar")}
    audit = {"schema": 2, "snapshot": SNAPSHOT, "files": {}}
    for name, path in paths.items():
        with path.open("rb") as source:
            audit["files"][name] = {"url": f"{SOURCE}/{path.name}", "bytes": path.stat().st_size,
                                    "sha256": file_digest(source, "sha256").hexdigest()}

    with TemporaryDirectory(dir=DATA) as directory:
        temp_db = DATA / directory / DATABASE.name
        with closing(sqlite3.connect(temp_db)) as db, db:
            audit["listings"] = import_listings(db, paths["listings"])
            print("Listings checked", flush=True)
            listing_ids = {row[0] for row in db.execute("SELECT listing_id FROM listings")}
            audit["reviews"] = import_reviews(db, paths["reviews"], listing_ids, audit["listings"]["scraped_max"])
            print("Reviews checked", flush=True)
            audit["calendar"] = audit_calendar(paths["calendar"], listing_ids)
            print("Calendar checked", flush=True)
            db.execute("CREATE TABLE audit (json TEXT NOT NULL)")
            db.execute("INSERT INTO audit VALUES (?)", (json.dumps(audit),))
        temp_db.replace(DATABASE)
    OUTPUT.mkdir(exist_ok=True)
    (OUTPUT / "data_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print("Prepared data/lisbon.sqlite3 and output/data_audit.json")


if __name__ == "__main__":
    main()
