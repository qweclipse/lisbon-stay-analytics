from collections import Counter
from dataclasses import replace
from datetime import date, timedelta
import json
import re

from imblearn.over_sampling import SMOTE
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.preprocessing import StandardScaler

from evaluate import SEED, holdout_predictions, host_split
from prepare_data import detect_review, price_match, review_signal
from pricing import (MATCH_FIELDS, MIN_PEERS, MIN_TEXTS, OUTPUT, Replay, get_benchmarks,
                     load_listings, load_reviews, open_database)


def bias_experiment(listings):
    train, test = host_split(listings)
    private = train[train.room_type.eq("Private room")].sample(frac=1, random_state=SEED)
    others = train[~train.room_type.eq("Private room")].sample(frac=1, random_state=SEED)
    size = int(len(train) * 0.7)
    normal_private = round(size * len(private) / len(train))
    reduced_private = normal_private // 3
    predictions = []
    for group_size in (normal_private, reduced_private):
        reference = pd.concat([private.iloc[:group_size], others.iloc[:size - group_size]])
        assert len(reference) == size
        predictions.append(holdout_predictions(reference, test))

    groups = []
    for label, mask in (("all", np.ones(len(test), dtype=bool)),
                        ("Private room", test.room_type.eq("Private room").to_numpy()),
                        ("other", test.room_type.ne("Private room").to_numpy())):
        control, reduced = [p.loc[mask] for p in predictions]
        errors = [(p.median_eur - p.observed_eur).abs() for p in (control, reduced)]
        common = errors[0].notna() & errors[1].notna()
        groups.append({"group": label, "test": len(control), "common": int(common.sum()),
                       "control_assessed": int(errors[0].count()), "reduced_assessed": int(errors[1].count()),
                       "control_mae": float(errors[0].mean()), "reduced_mae": float(errors[1].mean()),
                       "control_common_mae": float(errors[0][common].mean()),
                       "reduced_common_mae": float(errors[1][common].mean())})
    return {"seed": SEED, "train_per_arm": size, "control_private": normal_private,
            "reduced_private": reduced_private, "groups": groups}


def noise_variants(event):
    formatted = event.body.replace(" ", " <br> ").upper()
    match = price_match(event.body)
    typo = event.body
    if match:
        index = match.start() + 1
        typo = typo[:index] + typo[index + 1:]
    return [replace(event, evidence=review_signal(formatted)[2]),
            replace(event, evidence=detect_review(typo)[2])]


def policy_experiment(db, listings):
    priced = listings.dropna(subset=["price_cents"])
    control_ids = set(np.random.default_rng(SEED).choice(priced.listing_id, size=min(800, len(priced)), replace=False))
    policy_rows = []
    noise = Counter({"listings": 0, "events": 0, "clean_flags": 0, "clean_reviews": 0, "clean_firings": 0})
    for label in ("format", "typo"):
        noise.update({f"{label}_{metric}": 0 for metric in ("flags", "flag_flips", "reviews", "decision_flips", "firings")})

    for _, cohort in priced.groupby(MATCH_FIELDS):
        if len(cohort) <= MIN_PEERS:
            continue
        prices = cohort.price_cents.to_numpy(dtype=np.int64)
        for i, listing_id in enumerate(cohort.listing_id):
            bounds = get_benchmarks(np.concatenate((prices[:i], prices[i + 1:])))
            events = load_reviews(db, listing_id)
            replay = Replay(prices[i], bounds)
            clean = [replay.push(event) for event in events]
            price_only = replay.above_typical
            policy_rows.append({
                "listing_id": int(listing_id), "price_eur": prices[i] / 100, "p75_eur": bounds.p75 / 100,
                "above_p75": price_only, "events": len(events), "enough_text": sum(s.texts >= MIN_TEXTS for s in clean),
                "price_only": len(events) if price_only else 0, "combined": sum(s.active for s in clean),
                "text_withholds": sum(price_only and s.texts >= MIN_TEXTS and not s.active for s in clean),
                "text_waits": sum(price_only and s.texts < MIN_TEXTS for s in clean),
                "combined_firings": sum(s.fired for s in clean),
            })
            if not (price_only and events and (listing_id in control_ids or any(event.evidence for event in events))):
                continue
            noise["listings"] += 1
            noise["events"] += len(events)
            noise["clean_flags"] += sum(bool(event.evidence) for event in events)
            noise["clean_reviews"] += sum(s.active for s in clean)
            noise["clean_firings"] += sum(s.fired for s in clean)
            variants = {label: Replay(prices[i], bounds) for label in ("format", "typo")}
            for event, state in zip(events, clean):
                for (label, altered), modified in zip(variants.items(), noise_variants(event)):
                    variant = altered.push(modified)
                    noise[f"{label}_flags"] += bool(modified.evidence)
                    noise[f"{label}_flag_flips"] += bool(modified.evidence) != bool(event.evidence)
                    noise[f"{label}_reviews"] += variant.active
                    noise[f"{label}_decision_flips"] += variant.active != state.active
                    noise[f"{label}_firings"] += variant.fired

    rows = pd.DataFrame(policy_rows)
    summary = {"priced": len(priced), "eligible": len(rows), "with_events": int(rows.events.gt(0).sum()),
               "above_p75_listings": int((rows.above_p75 & rows.events.gt(0)).sum()),
               "changed_sufficient_listings": int(rows.text_withholds.gt(0).sum())}
    summary.update({col: int(rows[col].sum()) for col in (
        "events", "enough_text", "price_only", "combined", "text_withholds", "text_waits", "combined_firings",
    )})
    assert summary["price_only"] == summary["combined"] + summary["text_withholds"] + summary["text_waits"]
    return rows, summary, dict(noise)


TRAIN_CUTOFF = date(2025, 6, 1)
TEST_CUTOFF = date(2026, 6, 1)
LOOKBACK_DAYS = 365
FUTURE_DAYS = 90


def activity_cohort(db, cutoff):
    history_start = cutoff - timedelta(days=LOOKBACK_DAYS)
    future_end = cutoff + timedelta(days=FUTURE_DAYS)
    features = pd.read_sql_query("""SELECT author_id, COUNT(*) AS reviews, MAX(day) AS latest
        FROM reviews WHERE author_id IS NOT NULL AND day >= ? AND day < ? GROUP BY author_id""",
        db, params=(history_start.isoformat(), cutoff.isoformat()))
    future = {r[0] for r in db.execute("""SELECT DISTINCT author_id FROM reviews
        WHERE author_id IS NOT NULL AND day >= ? AND day < ?""", (cutoff.isoformat(), future_end.isoformat()))}
    features["recency"] = (pd.Timestamp(cutoff) - pd.to_datetime(features.pop("latest"))).dt.days
    features["repeat"] = features.author_id.isin(future).astype(int)
    features["test_author"] = pd.util.hash_pandas_object(features.author_id, index=False).mod(5).eq(0)
    return features


def author_experiment(db, audit):
    if TEST_CUTOFF + timedelta(days=FUTURE_DAYS) > date.fromisoformat(audit["listings"]["scraped_min"]):
        raise ValueError("Future test window extends beyond the earliest scrape")
    assert TRAIN_CUTOFF + timedelta(days=FUTURE_DAYS) <= TEST_CUTOFF
    train = activity_cohort(db, TRAIN_CUTOFF).loc[lambda frame: ~frame.test_author]
    test = activity_cohort(db, TEST_CUTOFF).loc[lambda frame: frame.test_author]
    assert set(train.author_id).isdisjoint(test.author_id)

    x_train = np.column_stack((np.log1p(train.reviews), train.recency))
    x_test = np.column_stack((np.log1p(test.reviews), test.recency))
    scaler = StandardScaler().fit(x_train)
    x_train, x_test = scaler.transform(x_train), scaler.transform(x_test)
    y_train, y_test = train.repeat, test.repeat
    balanced_x, balanced_y = SMOTE(random_state=SEED, k_neighbors=5).fit_resample(x_train, y_train)
    metrics, curves = [], []
    for name, x, y in (("baseline", x_train, y_train), ("smote", balanced_x, balanced_y)):
        model = LogisticRegression(max_iter=500).fit(x, y)
        scores = model.predict_proba(x_test)[:, 1]
        prediction = scores >= 0.5
        metrics.append({"model": name, "train_size": len(y), "train_positive": int(y.sum()),
                        "precision": float(precision_score(y_test, prediction, zero_division=0)),
                        "recall": float(recall_score(y_test, prediction)),
                        "macro_f1": float(f1_score(y_test, prediction, average="macro")),
                        "balanced_accuracy": float(balanced_accuracy_score(y_test, prediction)),
                        "average_precision": float(average_precision_score(y_test, scores)),
                        "roc_auc": float(roc_auc_score(y_test, scores)),
                        "confusion": confusion_matrix(y_test, prediction, labels=[0, 1]).tolist()})
        for count in np.linspace(1, 15, 141):
            weight_count, weight_days = model.coef_[0]
            log_count = (np.log1p(count) - scaler.mean_[0]) / scaler.scale_[0]
            boundary = (-model.intercept_[0] - weight_count * log_count) / weight_days
            days = float(boundary * scaler.scale_[1] + scaler.mean_[1])
            if 1 <= days <= LOOKBACK_DAYS:
                curves.append({"model": name, "reviews": float(count), "days": days})
    return {"train_cutoff": TRAIN_CUTOFF.isoformat(), "test_cutoff": TEST_CUTOFF.isoformat(),
            "lookback_days": LOOKBACK_DAYS, "future_days": FUTURE_DAYS, "threshold": 0.5,
            "test_size": len(test), "test_positive": int(y_test.sum()), "models": metrics, "curves": curves}


def annotation_sample(db):
    flagged = pd.read_sql_query("SELECT review_id, listing_id, body, phrase FROM reviews WHERE phrase <> ''", db)
    unflagged = pd.read_sql_query("""SELECT review_id, listing_id, body, phrase FROM reviews
        WHERE has_text = 1 AND phrase = '' AND review_id % 997 = 43""", db)
    price_like = unflagged.body.str.contains(
        r"\b(?:price|cost|car[oa]|preço|prix|precio|expensive|parking|tax|fee|frais|taxe|tarif|limpeza)\b",
        case=False, regex=True,
    )
    hard = unflagged[price_like].sample(12, random_state=SEED)
    selected = pd.concat([flagged.sample(24, random_state=SEED).assign(stratum="matched"),
                          hard.assign(stratum="unmatched_price_like"),
                          unflagged.drop(hard.index).sample(24, random_state=SEED).assign(stratum="unmatched_random")])
    selected = selected.sample(frac=1, random_state=SEED).reset_index(drop=True)
    selected.insert(0, "sample_id", [f"R{i:02d}" for i in range(1, len(selected) + 1)])
    review_text = selected.body.map(lambda value: "'" + value if re.match(r"^[=+@\-]", value) else value)
    template = pd.DataFrame({"sample_id": selected.sample_id, "text": review_text, "scope": "",
                             "evidence": "", "language": "", "notes": ""})
    template.to_csv(OUTPUT / "review_annotation_template.csv", sep=";", index=False, encoding="utf-8-sig")
    selected.drop(columns="body").assign(flag=selected.phrase.ne("")).to_csv(
        OUTPUT / "review_annotation_key.csv", index=False, encoding="utf-8-sig")
    return {"size": len(template), "strata": {"matched": 24, "unmatched_price_like": 12, "unmatched_random": 24}}


def main():
    OUTPUT.mkdir(exist_ok=True)
    with open_database() as db:
        audit = json.loads(db.execute("SELECT json FROM audit").fetchone()[0])
        listings = load_listings(db).sort_values("listing_id")
        bias = bias_experiment(listings)
        print("Bias complete", flush=True)
        rows, policy, noise = policy_experiment(db, listings)
        print("Policy comparison and noise complete", flush=True)
        activity = author_experiment(db, audit)
        print("Author activity complete", flush=True)
        annotation = annotation_sample(db)
    rows.to_csv(OUTPUT / "policy_comparison.csv", index=False)
    result = {"schema": audit["schema"], "files": audit["files"], "bias": bias, "policy": policy,
              "noise": noise, "author_activity": activity, "annotation": annotation}
    temporary = OUTPUT / "experiments.json.tmp"
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(OUTPUT / "experiments.json")
    print("Experiments saved to output/experiments.json", flush=True)


if __name__ == "__main__":
    main()
