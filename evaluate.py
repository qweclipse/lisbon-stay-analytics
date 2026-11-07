import json
from time import perf_counter_ns

import numpy as np
import pandas as pd

from pricing import (MATCH_FIELDS, MIN_FLAGS, OUTPUT, REPLAY_LIMIT, Replay, get_benchmarks, load_listings,
                     load_reviews, open_database, select_peers)


SEED = 20


def host_split(listings):
    priced = listings.dropna(subset=["price_cents"])
    hosts = np.sort(priced["host_id"].unique())
    test_hosts = np.random.default_rng(SEED).choice(hosts, size=len(hosts) // 5, replace=False)
    is_test = priced["host_id"].isin(test_hosts)
    return priced.loc[~is_test], priced.loc[is_test]


def holdout_predictions(train, test):
    groups = train.groupby(MATCH_FIELDS)["price_cents"]
    sizes = groups.size().to_dict()
    benchmarks = {key: get_benchmarks(prices) for key, prices in groups}
    rows = []
    for row in test.itertuples(index=False):
        key = row.district, row.room_type, row.capacity
        bounds = benchmarks.get(key)
        rows.append({
            "listing_id": row.listing_id, "district": row.district, "room_type": row.room_type,
            "capacity": row.capacity, "peers": sizes.get(key, 0), "observed_eur": row.price_cents / 100,
            "median_eur": bounds.median / 100 if bounds else None,
            "p25_eur": bounds.p25 / 100 if bounds else None,
            "p75_eur": bounds.p75 / 100 if bounds else None,
            "within_typical": bounds.category(row.price_cents) == 2 if bounds else None,
        })
    return pd.DataFrame(rows)


def evaluate_holdout(listings):
    train, test = host_split(listings)
    results = holdout_predictions(train, test)
    assessed = results.dropna(subset=["median_eur"])
    error = assessed["median_eur"] - assessed["observed_eur"]
    group_metrics = []
    for room, group in results.groupby("room_type"):
        error_by_room = (group["median_eur"] - group["observed_eur"]).dropna()
        group_metrics.append({"room_type": room, "test": len(group), "assessed": len(error_by_room),
                              "mae_eur": float(error_by_room.abs().mean()) if len(error_by_room) else None,
                              "mbe_eur": float(error_by_room.mean()) if len(error_by_room) else None})
    metrics = {
        "seed": SEED, "train_hosts": int(train["host_id"].nunique()), "test_hosts": int(test["host_id"].nunique()),
        "train_listings": len(train), "test_listings": len(test), "assessed": len(assessed),
        "mae_eur": float(error.abs().mean()), "median_absolute_error_eur": float(error.abs().median()),
        "mbe_eur": float(error.mean()), "typical_coverage": float(assessed["within_typical"].mean()),
        "typical_hits": int(assessed["within_typical"].sum()),
        "assessment_coverage": len(assessed) / len(test),
        "global_median_eur": float(train["price_cents"].median() / 100),
        "global_median_mae_eur": float((assessed["observed_eur"] - train["price_cents"].median() / 100).abs().mean()),
        "room_types": group_metrics,
    }
    return results, metrics


def evaluate_demo(db, listings):
    # Select a real sequence where the price and review conditions both occur.
    flagged_ids = [row[0] for row in db.execute(
        "SELECT listing_id FROM reviews WHERE phrase <> '' GROUP BY listing_id HAVING COUNT(*) >= ?", (MIN_FLAGS,)
    )]
    candidates = listings.loc[listings["listing_id"].isin(flagged_ids)].dropna(subset=["price_cents"])
    for listing in candidates.itertuples(index=False):
        peers = select_peers(listings, listing.district, listing.room_type, listing.capacity, listing.listing_id)
        bounds = get_benchmarks(peers)
        if bounds is None or listing.price_cents <= bounds.p75:
            continue
        events = load_reviews(db, listing.listing_id)
        if len(events) < REPLAY_LIMIT:
            continue
        replay = Replay(listing.price_cents, bounds)
        elapsed, log = [], []
        for event in events:
            start = perf_counter_ns()
            state = replay.push(event)
            elapsed.append((perf_counter_ns() - start) / 1_000_000)
            log.append({"listing_id": listing.listing_id, "review_id": event.review_id, "date": event.day.isoformat(),
                        "texts": state.texts, "flags": state.flags, "share": state.share,
                        "price_eur": listing.price_cents / 100, "p75_eur": bounds.p75 / 100,
                        "price_only_review": replay.above_typical, "rtap_active": state.active,
                        "rtap_fired": state.fired, "evidence": event.evidence})
        firings = [i for i, row in enumerate(log, 1) if row["rtap_fired"]]
        if firings:
            return pd.DataFrame(log), {
                "listing_id": listing.listing_id, "price_eur": listing.price_cents / 100, "peers": len(peers),
                "p25_eur": bounds.p25 / 100, "median_eur": bounds.median / 100, "p75_eur": bounds.p75 / 100,
                "p10_eur": bounds.p10 / 100, "p90_eur": bounds.p90 / 100,
                "events": len(log), "start_date": log[0]["date"], "end_date": log[-1]["date"],
                "firings": firings, "active_events": sum(row["rtap_active"] for row in log),
                "p95_processing_ms": float(np.percentile(elapsed, 95)),
            }
    raise ValueError("No real sequence matched the demonstration criteria")


def main():
    with open_database() as db:
        listings = load_listings(db).sort_values("listing_id")
        audit = json.loads(db.execute("SELECT json FROM audit").fetchone()[0])
        holdout, metrics = evaluate_holdout(listings)
        log, demo = evaluate_demo(db, listings)
    OUTPUT.mkdir(exist_ok=True)
    holdout.to_csv(OUTPUT / "holdout.csv", index=False)
    log.to_csv(OUTPUT / "replay_demo.csv", index=False)
    result = {"schema": audit["schema"], "files": audit["files"], "holdout": metrics, "replay_demo": demo}
    temporary = OUTPUT / "evaluation.json.tmp"
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(OUTPUT / "evaluation.json")
    print(json.dumps({"holdout": metrics, "replay_demo": demo}, indent=2))


if __name__ == "__main__":
    main()
