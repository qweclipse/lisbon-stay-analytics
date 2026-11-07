"""Reproduce the additive Russian revision without changing baseline results."""
from collections import Counter
from datetime import date
import json

import numpy as np
import pandas as pd

from experiments import TEST_CUTOFF, activity_cohort
from pricing import OUTPUT, Benchmarks, Replay, load_listings, load_reviews, open_database

DEST = OUTPUT / "revision_ru"


def activity_segment(count, recency):
    if count < 1 or not 1 <= recency <= 365:
        raise ValueError("Expected an observed review within the preceding 365 days")
    return ("Разовая" if count == 1 else "Повторная") + (" недавняя" if recency <= 90 else " давняя")


def binary_metrics(truth, prediction):
    counts = Counter(zip(truth, prediction))
    tn, fp, fn, tp = (counts[(False, False)], counts[(False, True)], counts[(True, False)], counts[(True, True)])
    return {"tn": tn, "fp": fp, "fn": fn, "tp": tp,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None}


def annotations():
    sample = pd.read_csv(OUTPUT / "review_annotation_template.csv", sep=";", keep_default_na=False)
    key = pd.read_csv(OUTPUT / "review_annotation_key.csv", keep_default_na=False)
    seeds = json.loads((DEST / "machine_annotation_seed.json").read_text(encoding="utf-8"))
    rows = sample.merge(key, on="sample_id", validate="one_to_one")
    assert set(rows.sample_id) == set(seeds) and len(rows) == 60
    rows["machine_scope"] = rows.sample_id.map(lambda sid: seeds[sid][0])
    rows["machine_evidence"] = rows.sample_id.map(lambda sid: seeds[sid][1])
    for row in rows.itertuples():
        assert not row.machine_evidence or row.machine_evidence.casefold() in row.text.casefold(), row.sample_id
    rows["provenance"] = "Предварительная машинная разметка Codex; человек не проверял"
    reasons = {"lodging": "Негативная оценка стоимости самого проживания", "extras": "Стоимость дополнительных или сторонних услуг",
               "both": "Замечания и о проживании, и об отдельных доплатах", "none": "Нет негативной оценки цены",
               "unclear": "Неясен объект или знак оценки; требуется человек"}
    rows["machine_rationale"] = rows.machine_scope.map(reasons)
    for sid, explanation in {"R08": "Магазин рядом, не цена жилья", "R28": "Трансфер, не цена жилья",
        "R31": "Жалоба автора на изменение цены при бронировании; её причина не проверена",
        "R35": "Критика цен рынка сочетается с рекомендацией этой квартиры",
        "R51": "Есть слово overpriced, но далее okay value; спорная выраженность жалобы"}.items():
        rows.loc[rows.sample_id.eq(sid), "machine_rationale"] = explanation
    # Fixed split before any dictionary changes. The baseline dictionary stays unchanged.
    rows["partition"] = "control"
    for _, group in rows.groupby("stratum"):
        rows.loc[group.sort_values("sample_id").index[::2], "partition"] = "development"
    rows.to_csv(DEST / "reviews_machine_preliminary.csv", sep=";", encoding="utf-8-sig", index=False)
    results = {}
    for label, group in [("all", rows), *list(rows.groupby("partition"))]:
        known = group[group.machine_scope.ne("unclear")]
        results[label] = {"n": len(group), "unclear": len(group) - len(known),
                          "scope_counts": group.machine_scope.value_counts().to_dict(),
                          "lodging": binary_metrics(known.machine_scope.isin(["lodging", "both"]), known.flag),
                          "any_price": binary_metrics(known.machine_scope.ne("none"), known.flag)}
    return results


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    with open_database() as db:
        listings = load_listings(db)
        eda = []
        for room, group in listings.groupby("room_type"):
            prices = group.price_cents.dropna() / 100
            eda.append({"room_type": room, "listings": len(group), "missing": int(group.price_cents.isna().sum()),
                        **{f"p{q}": float(prices.quantile(q / 100)) for q in (10, 25, 50, 75, 90)},
                        "max": float(prices.max())})
        counts = pd.read_sql_query("""SELECT listing_id, COUNT(*) AS reviews,
            SUM(has_text) AS texts, SUM(phrase != '') AS flags FROM reviews
            WHERE day >= '2025-09-19' AND day < '2026-09-19' GROUP BY listing_id""", db)
        feedback = listings.merge(counts, on="listing_id", how="left").fillna({"reviews": 0, "texts": 0, "flags": 0})
        feedback["activity"] = pd.cut(feedback.reviews, [-1, 0, 9, 49, float("inf")], labels=["0", "1–9", "10–49", "50+"])
        availability = []
        for label, group in feedback.groupby("activity", observed=True):
            availability.append({"activity": str(label), "listings": len(group),
                                 "missing_price": int(group.price_cents.isna().sum()),
                                 "reviews": int(group.reviews.sum()), "texts": int(group.texts.sum()),
                                 "flags": int(group["flags"].sum()), "median_eur": float(group.price_cents.median() / 100)})
        cohort = activity_cohort(db, TEST_CUTOFF)
        cohort = cohort.loc[cohort.test_author].copy()
        cohort["segment"] = [activity_segment(c, d) for c, d in zip(cohort.reviews, cohort.recency)]
        segments = cohort.groupby("segment").agg(authors=("author_id", "size"),
                         past_reviews=("reviews", "sum"), future_repeat=("repeat", "sum")).reset_index()
        segments["repeat_rate"] = segments.future_repeat / segments.authors
        segments.to_csv(DEST / "descriptive_segments.csv", index=False, encoding="utf-8-sig")
        policy = pd.read_csv(OUTPUT / "policy_comparison.csv")
        funnel = Counter()
        for row in policy.loc[policy.above_p75].itertuples():
            replay = Replay(round(row.price_eur * 100), Benchmarks(0, 0, 0, round(row.p75_eur * 100), 0))
            for event in load_reviews(db, row.listing_id):
                state = replay.push(event)
                funnel["above_p75"] += 1
                if state.texts >= 5:
                    funnel["five_texts"] += 1
                    if state.flags >= 2:
                        funnel["two_flags"] += 1
                        if state.flags / state.texts >= .2:
                            funnel["share_20"] += 1
                funnel["firings"] += int(state.fired)
        assert funnel["above_p75"] == int(policy.price_only.sum())
        assert funnel["share_20"] == int(policy.combined.sum())
        assert len(cohort) == 62020 and int(cohort.repeat.sum()) == 735
    results = {"snapshot": "2026-09-19", "files": json.loads((OUTPUT / "data_audit.json").read_text(encoding="utf-8"))["files"],
               "feedback_window": ["2025-09-19", "2026-09-19"],
               "author_cutoff": TEST_CUTOFF.isoformat(), "eda": eda, "review_availability": availability,
               "segments": segments.to_dict(orient="records"), "policy_funnel": dict(funnel),
               "annotation": annotations(), "annotation_status": "preliminary_machine_not_human"}
    (DEST / "revision_summary.json").write_text(json.dumps(results, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
