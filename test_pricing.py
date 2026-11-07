from contextlib import closing
from datetime import date, timedelta
import gzip
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from dashboard_charts import market_domain
from experiments import activity_cohort, noise_variants
from prepare_data import import_reviews, price_cents, review_signal
from pricing import Benchmarks, Replay, Review, get_benchmarks, select_peers
from revision_analysis import activity_segment, binary_metrics
from download_data import verified


class PricingTest(unittest.TestCase):
    def test_descriptive_segments_partition_and_annotation_counts(self):
        self.assertEqual([activity_segment(c, d) for c, d in [(1, 90), (1, 91), (2, 90), (2, 365)]],
                         ["Разовая недавняя", "Разовая давняя", "Повторная недавняя", "Повторная давняя"])
        with self.assertRaises(ValueError):
            activity_segment(0, 20)
        self.assertEqual(binary_metrics([True, True, False, False], [True, False, True, False]),
                         {"tn": 1, "fp": 1, "fn": 1, "tp": 1, "precision": .5, "recall": .5})

    def test_download_checks_size_and_hash_before_replacement(self):
        import hashlib
        with TemporaryDirectory() as folder:
            target = Path(folder) / "sample.gz"
            expected = {"bytes": 3, "sha256": hashlib.sha256(b"abc").hexdigest()}
            self.assertFalse(verified(target, expected))
            target.write_bytes(b"xyz")
            self.assertFalse(verified(target, expected))
            target.write_bytes(b"abc")
            self.assertTrue(verified(target, expected))

    def test_comparables_exclude_self_and_unpriced_rows(self):
        rows = [(i, "A", "Private room", 2, 100 * i) for i in range(1, 16)]
        rows += [(16, "A", "Private room", 2, None), (17, "B", "Private room", 2, 10000),
                 (18, "A", "Private room", 3, 10000), (19, "A", "Entire home/apt", 2, 10000)]
        listings = pd.DataFrame(rows, columns=["listing_id", "district", "room_type", "capacity", "price_cents"])
        self.assertEqual(get_benchmarks(select_peers(listings, "A", "Private room", 2)),
                         Benchmarks(240, 450, 800, 1150, 1360))
        peers = select_peers(listings, "A", "Private room", 2, 1)
        self.assertEqual(len(peers), 14)
        self.assertNotIn(100, peers.tolist())
        self.assertIsNone(get_benchmarks(peers))

    def test_price_boundaries_and_distance(self):
        bounds = Benchmarks(4000, 5000, 6000, 7000, 8000)
        self.assertEqual([bounds.category(price) for price in (3999, 4000, 5000, 7000, 7001, 8000, 8001)],
                         [0, 1, 2, 2, 3, 3, 4])
        self.assertEqual(bounds.nearest_typical_edge(4900), ("P25", 100))
        self.assertEqual(bounds.nearest_typical_edge(7050), ("P75", -50))
        self.assertEqual(bounds.nearest_typical_edge(6000), ("P25", -1000))
        self.assertEqual(price_cents("$1,239.75"), 123975)
        self.assertIsNone(price_cents(""))

    def test_review_signal_and_rtap_transition(self):
        has_text, phrase, snippet = review_signal("Nice location. <br>Not worth the price.")
        self.assertTrue(has_text)
        self.assertEqual(phrase, "Not worth the price")
        self.assertIn(phrase, snippet)
        self.assertFalse(review_signal("It wasn't too expensive.")[1])
        self.assertFalse(review_signal("<br> 123")[0])

        bounds = Benchmarks(4000, 5000, 6000, 7000, 8000)
        expensive, typical, unavailable = Replay(7100, bounds), Replay(7000, bounds), Replay(7100, None)
        day = date(2026, 1, 1)
        events = [Review(i, day + timedelta(days=i), True, snippet if i < 2 else "") for i in range(6)]
        self.assertEqual([expensive.push(event).fired for event in events], [False, False, False, False, True, False])
        self.assertFalse(any(typical.push(event).fired for event in events))
        self.assertFalse(any(unavailable.push(event).fired for event in events))
        self.assertEqual(expensive.push(Review(6, day + timedelta(days=90), False, "")).flags, 1)
        self.assertFalse(expensive.active)
        self.assertTrue(expensive.push(Review(7, day + timedelta(days=90), True, snippet)).fired)

    def test_ambiguous_review_ids_are_removed_from_all_listings(self):
        with TemporaryDirectory() as directory, closing(sqlite3.connect(":memory:")) as db:
            path = Path(directory) / "reviews.csv.gz"
            with gzip.open(path, "wt", encoding="utf-8") as file:
                file.write("listing_id,id,date,comments\n"
                           "1,10,2026-09-19,Overpriced\n1,10,2026-09-19,Overpriced\n"
                           "1,11,2026-09-19,Too expensive\n2,11,2026-09-19,Too expensive\n"
                           "1,11,2026-09-19,Too expensive\n2,12,2026-09-19,Great room\n")
            audit = import_reviews(db, path, {1, 2}, "2026-09-19")
            self.assertEqual((audit["stored"], audit["ambiguous_ids"], audit["duplicate_rows"]), (2, 1, 3))
            self.assertEqual(db.execute("SELECT review_id, listing_id FROM reviews").fetchall(), [(10, 1), (12, 2)])

    def test_market_domain_contains_real_price_or_uses_callout(self):
        peers = list(range(8000, 19000, 500))
        low, high, on_axis = market_domain(peers, 27300)
        self.assertTrue(on_axis)
        self.assertLess(low, 27300)
        self.assertGreater(high, 27300)
        _, high, on_axis = market_domain(peers, 1000000)
        self.assertFalse(on_axis)
        self.assertLess(high, 1000000)

    def test_past_decision_retains_all_original_texts(self):
        day = date(2026, 1, 1)
        replay = Replay(7100, Benchmarks(4000, 5000, 6000, 7000, 8000))
        events = [Review(i, day, True, "Overpriced" if i < 2 else "", "Overpriced" if i < 2 else "Great stay")
                  for i in range(5)]
        states = [replay.push(event) for event in events]
        self.assertTrue(replay.above_typical)
        self.assertFalse(states[0].active)
        self.assertEqual(states[-1].decision_window, tuple(events))
        replay.push(Review(6, day + timedelta(days=90), True, "", "Nice place"))
        self.assertFalse(replay.active)
        self.assertEqual(states[-1].decision_window, tuple(events))
        formatted, typo = noise_variants(events[0])
        self.assertTrue(formatted.evidence)
        self.assertFalse(typo.evidence)

    def test_activity_features_and_label_respect_cutoff(self):
        with closing(sqlite3.connect(":memory:")) as db:
            db.execute("CREATE TABLE reviews (author_id INTEGER, day TEXT)")
            db.executemany("INSERT INTO reviews VALUES (?, ?)", [(1, "2026-05-30"), (1, "2026-06-01"),
                                                                 (2, "2026-05-01"), (2, "2026-08-30"),
                                                                 (3, "2026-06-01")])
            features = activity_cohort(db, date(2026, 6, 1)).set_index("author_id")
        self.assertEqual(features.index.tolist(), [1, 2])
        self.assertEqual(features.reviews.tolist(), [1, 1])
        self.assertEqual(features.recency.tolist(), [2, 31])
        self.assertEqual(features.repeat.tolist(), [1, 0])


if __name__ == "__main__":
    unittest.main()
