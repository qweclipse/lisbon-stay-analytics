from html import escape
import json

import pandas as pd
import streamlit as st

from dashboard_charts import availability_chart, boundary_chart, error_chart, market_chart, timeline_chart
from prepare_data import price_match
from pricing import (CATEGORIES, DATABASE, FLAG_PERCENT, MIN_FLAGS, MIN_PEERS, MIN_TEXTS, OUTPUT,
                     REPLAY_LIMIT, ROOM_LABELS, WINDOW_DAYS, Replay, get_benchmarks, load_listings,
                     load_reviews, open_database, select_peers)


SNAPSHOT_NOTE = "Снимок 19.09.2026 · [Inside Airbnb](https://insideairbnb.com/get-the-data/) · цены предложений"
STYLES = """<style>
.stMainBlockContainer {max-width: 1080px; padding-top: 2.2rem; padding-bottom: 3rem;}
.stMain h1 {font-size: clamp(1.55rem, 4vw, 2.1rem); letter-spacing: -.025em; padding-top: 0;}
.stMain h2 {font-size: clamp(1.15rem, 3vw, 1.55rem); letter-spacing: -.018em;}
.stMain h3 {font-size: 1.12rem;}
.stMain [class*="st-key-panel_"] {background: #23282e; border-color: #3b454f; border-radius: 12px;}
.stMain [data-testid="stTabs"] [role="tablist"] {column-gap: .35rem;}
.stMain [data-testid="stTabs"] [role="tab"] {border-radius: .45rem .45rem 0 0; padding-inline: .65rem;}
.stMain [data-testid="stTabs"] [aria-selected="true"] {background: #273c37; color: #b6f4dd;}
.price-summary h3 {font-size: clamp(1.25rem, 4vw, 1.6rem); padding: 0; margin: 0 0 .15rem;}
.price-summary p {margin: 0; color: #eaf0f4;}
.price-metrics {display: grid; grid-template-columns: 1.2fr 1.25fr .65fr; gap: .65rem; margin-top: .8rem;}
.price-metrics div {padding: .55rem .75rem; background: #1b2025; border-radius: 8px; min-width: 0;}
.price-metrics span {display: block; font-size: .8rem; color: #b1bdc8;}
.price-metrics strong {font-size: clamp(.95rem, 2.5vw, 1.18rem);}
.review-body {white-space: pre-wrap; overflow-wrap: anywhere;}
.review-body mark {background: #ffd174; color: #1a1e23; border-radius: .2rem; padding: .03rem .12rem;}
@media(max-width: 540px) {
    .stMainBlockContainer {padding-top: 1.8rem;}
    .stMain [data-testid="stTabs"] [role="tab"] {padding-inline: .4rem;}
    .price-metrics {grid-template-columns: 1.1fr 1fr; gap: .4rem;}
    .price-metrics div {padding: .45rem .5rem;}
    .price-metrics div:first-child {grid-column: 1 / -1;}
}
</style>"""


def euros(value):
    return f"{value:,.2f}".replace(",", " ").replace(".", ",") + " €"


def money(cents):
    return euros(cents / 100)


def percent(value, digits=1):
    return f"{value:.{digits}%}".replace(".", ",")


def number(value):
    return f"{value:,}".replace(",", " ")


@st.cache_data(show_spinner=False)
def snapshot(db_revision, evaluation_revision, experiments_revision):
    with open_database() as db:
        listings = load_listings(db)
        audit = json.loads(db.execute("SELECT json FROM audit").fetchone()[0])
    results = []
    for name, revision in (("evaluation", evaluation_revision), ("experiments", experiments_revision)):
        result = json.loads((OUTPUT / f"{name}.json").read_text(encoding="utf-8")) if revision else {}
        results.append(result if result.get("files") == audit["files"] and result.get("schema") == audit.get("schema") else {})
    return listings, audit, *results


@st.cache_data(show_spinner=False)
def review_events(db_revision, listing_id):
    with open_database() as db:
        return load_reviews(db, listing_id)


def price_result(peers, bounds, price, district, room, capacity, listing_id):
    if bounds is None:
        with st.container(border=True, key="panel_summary"):
            st.warning(f"Недостаточно аналогов · найдено {len(peers)} из {MIN_PEERS} необходимых")
            st.write(f"В районе {district} мало предложений с ценой для этого типа размещения и вместимости ({capacity}).")
            st.write("В режиме «Вручную» попробуйте соседний район, сохранив тип размещения и вместимость.")
            st.caption(SNAPSHOT_NOTE)
        return

    category = bounds.category(price)
    color = ("#91bff5", "#a5cbf8", "#8cebc6", "#ffd174", "#ffbc7b")[category]
    if price < bounds.p25:
        conclusion = f"Цена на {money(bounds.p25 - price)} ниже типичного диапазона."
    elif price > bounds.p75:
        conclusion = f"Цена на {money(price - bounds.p75)} выше типичного диапазона."
    else:
        _, shift = bounds.nearest_typical_edge(price)
        conclusion = (f"Цена внутри диапазона. До ближайшей границы {money(abs(shift))}." if shift
                      else "Цена на границе типичного диапазона.")
    delta = price - bounds.median
    sign = "+" if delta > 0 else ""
    shift_percent = f"{delta / bounds.median:+.1%}".replace(".", ",") if delta else "0,0%"
    interval = f"{money(bounds.p25)} – {money(bounds.p75)}"

    with st.container(border=True, key="panel_summary"):
        st.html(f"""<div class="price-summary">
          <h3 style="color:{color}">{CATEGORIES[category]}</h3>
          <p>{conclusion}</p>
          <div class="price-metrics">
            <div><span>К медиане</span><strong>{sign}{money(delta)} · {shift_percent}</strong></div>
            <div><span>Типичный диапазон</span><strong>{interval}</strong></div>
            <div><span>Аналогов</span><strong>{len(peers)}</strong></div>
          </div>
        </div>""")
        st.caption(f"{SNAPSHOT_NOTE} · {district}, {ROOM_LABELS[room].lower()}, до {capacity} гостей")

    with st.container(border=True, key="panel_market"):
        st.subheader("Распределение цен аналогов", anchor=False)
        st.write(f"**Ваша цена {money(price)}** · Медиана {money(bounds.median)} · Типичный диапазон {interval}")
        excluded, show_price = market_chart(peers, bounds, price)
        st.caption("Высота столбца показывает число предложений, зелёная зона — средние 50% по цене, "
                   "пунктир — медиану" + (", жёлтая линия — вашу цену." if show_price else "."))
        if not show_price:
            side = "левее" if price < bounds.p25 else "правее"
            st.warning(f"Ваша цена {money(price)} — {side} графика. Она вынесена отдельно, чтобы сохранить масштаб аналогов.")
        if excluded:
            st.caption(f"За пределами графика {excluded} аналогов. Они учтены в расчёте диапазона и медианы.")

    with st.expander("Как это рассчитано"):
        st.write(f"Совпадают район, тип размещения и максимальная вместимость. Используются предложения с ценой. "
                 f"Минимум {MIN_PEERS} аналогов." + (" Текущее объявление исключено." if listing_id is not None else ""))
        st.write("Отклонение в евро равно вашей цене минус медиана. Для процента эту разницу делим на медиану.")
        st.write(f"P10 {money(bounds.p10)} · P25 {money(bounds.p25)} · P75 {money(bounds.p75)} · P90 {money(bounds.p90)}")
        for label, interval_name in zip(CATEGORIES, ("цена < P10", "P10 ≤ цена < P25", "P25 ≤ цена ≤ P75",
                                                    "P75 < цена ≤ P90", "цена > P90")):
            st.markdown(f"- **{label}** · {interval_name}")
        st.caption("Границы интерполируются линейно и округляются до цента. При равном расстоянии выбирается P25.")


def decision_evidence(state):
    st.caption(f"Тексты, учтённые в тот момент · {state.flags} из {state.texts} со словарным совпадением.")
    for item in state.decision_window:
        match = price_match(item.body)
        body = escape(item.body)
        if match:
            start, end = match.span()
            body = escape(item.body[:start]) + "<mark>" + escape(item.body[start:end]) + "</mark>" + escape(item.body[end:])
        label = "Совпадение по словарю" if match else "Без совпадения"
        st.caption(f"{item.day:%d.%m.%Y} · отзыв {item.review_id} · {label}")
        st.html(f'<div class="review-body">{body}</div>')


def replay_panel(events, listing_id, price, bounds, db_revision, study):
    st.header("Историческое воспроизведение", anchor=False)
    st.write("RTAP — аналитическая обработка событий по мере поступления. Каждый отзыв обновляет сводку и проверяет правило пересмотра.")
    st.caption(f"Цена сценария {money(price)}. Ценовой ориентир взят из снимка 19.09.2026, "
               "а отзывы могли быть опубликованы раньше. Изменение объявления или цены сбрасывает прогон.")
    if not events:
        st.info("Для этого объявления в выгрузке нет событий отзывов.")
        return
    if bounds is None:
        st.info("Можно воспроизвести отзывы. Для сравнения рекомендаций нужно больше аналогов.")

    context = listing_id, price, bounds, db_revision
    if st.session_state.get("replay_context") != context:
        st.session_state.replay_context = context
        st.session_state.playing = False
        st.session_state.history = []
        st.session_state.review_replay = Replay(price, bounds)

    def advance():
        history = st.session_state.history
        if len(history) < len(events):
            event = events[len(history)]
            history.append((event, st.session_state.review_replay.push(event)))

    @st.fragment(run_every=0.8 if st.session_state.playing else None)
    def controls():
        history = st.session_state.history
        replay = st.session_state.review_replay
        playing = st.session_state.playing
        finished = len(history) == len(events)
        with st.container(border=True, key="panel_replay"):
            play, step, reset = st.columns(3)
            if play.button("Пауза" if playing else "Воспроизвести", disabled=finished, type="primary", width="stretch"):
                st.session_state.playing = not playing
                st.rerun()
            step.button("Следующий отзыв", disabled=playing or finished, on_click=advance, width="stretch")
            if reset.button("Сначала", disabled=not history and not playing, width="stretch"):
                st.session_state.pop("replay_context")
                st.rerun()
            if playing:
                advance()
                if len(history) == len(events):
                    st.session_state.playing = False
                    st.rerun()
            st.progress(len(history) / len(events), text=f"Обработано {len(history)} из {len(events)} отзывов")

        if not history:
            st.info("Запустите воспроизведение или обработайте первый отзыв вручную.")
            return

        event, state = history[-1]
        decisions = [(i, item, status) for i, (item, status) in enumerate(history, 1) if status.fired]
        with st.container(border=True, key="panel_window"):
            st.subheader("Текущее окно", anchor=False)
            st.caption(f"{event.day:%d.%m.%Y} · событие №{len(history)}" + (" · прогон завершён" if finished else ""))
            texts, flags, total = st.columns(3)
            texts.metric("Текстов в окне", state.texts)
            flags.metric("Доля замечаний", percent(state.share) if state.share is not None else "—")
            total.metric("Срабатываний в журнале", len(decisions))
            if bounds is not None:
                price_only = "пересмотреть" if replay.above_typical else "наблюдать"
                combined = ("пересмотреть" if state.active else "подождать отзывов"
                            if replay.above_typical and state.texts < MIN_TEXTS else "наблюдать")
                st.write(f"**Только цена — {price_only}** · **Цена и отзывы — {combined}**")
                if state.active:
                    st.warning("Отзывы подтверждают пересмотр. Проверьте, о чём именно пишут гости.")
                elif replay.above_typical:
                    st.info("Текст меняет рекомендацию: цена выше порога, а отзывов пока недостаточно."
                            if state.texts < MIN_TEXTS else
                            "Текст меняет рекомендацию: цена выше порога, но замечания в окне не подтверждают пересмотр.")

        with st.container(border=True, key="panel_timeline"):
            st.subheader("Хронология отзывов", anchor=False)
            timeline_chart(history)
            st.caption("Сверху число текстов, снизу доля замечаний в окне на каждом шаге, жёлтый пунктир отмечает срабатывание.")

        st.subheader("Журнал прошлых решений", anchor=False)
        if not decisions:
            st.caption("Срабатываний пока нет.")
        for step_number, item, status in decisions:
            with st.expander(f"{item.day:%d.%m.%Y} · Событие №{step_number} · Пересмотреть цену"):
                decision_evidence(status)

    controls()

    with st.expander("Как это рассчитано"):
        st.write(f"Воспроизводятся до {REPLAY_LIMIT} последних уникальных отзывов по дате и ID. "
                 f"Окно начинается пустым и на каждом шаге включает последние {WINDOW_DAYS} календарных дней. "
                 "Для доли учитываются только события с текстом.")
        st.write(f"Оба правила используют цену строго выше P75. Для подтверждения отзывами нужны от {MIN_TEXTS} текстов, "
                 f"от {MIN_FLAGS} совпадений и доля от {FLAG_PERCENT}%. Журнал фиксирует переход в состояние «пересмотреть».")
        st.write("Флаг ищет отдельные выражения о цене и ценности. Замечание может относиться к проживанию, "
                 "парковке, сбору или другой услуге. Категория жалобы определяется при ручной проверке.")
        st.caption("Пример о проживании — «Overpriced for Lisbon» в отзыве о размере квартиры. "
                   "Пример о доплате — «cleaning fee overpriced» в отзыве о сборе за уборку.")

    if study:
        policy = study["policy"]
        with st.expander("Сравнение правил на всех доступных объявлениях"):
            st.write(f"Сопоставлены {number(policy['with_events'])} объявлений и {number(policy['events'])} событий. "
                     f"При общем пороге P75 правило цены рекомендует пересмотр на {number(policy['price_only'])} шагах, "
                     f"правило с отзывами — на {number(policy['combined'])}.")
            st.write(f"При достаточном числе текстов рекомендация меняется на {number(policy['text_withholds'])} шагах. "
                     f"На {number(policy['text_waits'])} шагах правило ждёт больше отзывов.")
            st.caption("Это сравнение действий правил. Качество распознавания оценивается после ручной разметки.")


def evaluation_panel(audit, result, study):
    st.header("Проверка ценового ориентира", anchor=False)
    if result:
        score = result["holdout"]
        st.write(f"Объявления одних хозяев использованы для расчёта, а {number(score['test_listings'])} объявлений других — для теста.")
        with st.container(border=True, key="panel_holdout"):
            error, coverage, assessed = st.columns(3)
            error.metric("Средняя ошибка (MAE)", euros(score["mae_eur"]))
            coverage.metric("Попадание в диапазон", percent(score["typical_coverage"], 2))
            assessed.metric("Охват оценки", percent(score["assessment_coverage"], 2))
            st.subheader("Ошибка двух методов на одних объявлениях", anchor=False)
            error_chart(score)
            st.caption(f"Чем короче полоса, тем меньше средняя ошибка на одних и тех же {number(score['assessed'])} объявлениях.")

        with st.container(border=True, key="panel_availability"):
            st.subheader("Где чаще не хватает аналогов", anchor=False)
            availability_chart(score["room_types"])
            st.caption("Длинная полоса означает, что в тесте у большей доли объявлений этого типа было меньше 15 аналогов.")
            st.caption("Редкие группы в тесте: 36 номеров в отеле и 29 мест в общих комнатах.")

        with st.expander("Как это рассчитано"):
            st.write("MAE — средняя абсолютная разница с наблюдаемой ценой предложения. "
                     "Охват — доля всех отложенных объявлений с достаточным числом аналогов. "
                     "Попадание — доля цен внутри P25–P75 среди оценённых объявлений.")
            st.write("Общая медиана рассчитана по опорной выборке, медиана аналогов — по совпавшему сегменту той же выборки.")
    else:
        st.info("Для метрик этой копии данных выполните `python evaluate.py`.")

    if study:
        activity = study["author_activity"]
        with st.expander("Повторный отзыв · отдельный эксперимент SMOTE"):
            st.write(f"По активности за {activity['lookback_days']} дней прогнозируется появление нового отзыва автора "
                     f"в следующие {activity['future_days']} дней. На тесте {number(activity['test_size'])} авторов.")
            labels = {"baseline": "Обычная модель", "smote": "Модель со SMOTE"}
            st.dataframe(pd.DataFrame([{"Модель": labels[row["model"]], "Точность положительного прогноза": percent(row["precision"]),
                                        "Полнота": percent(row["recall"]), "Среднее F1 классов": percent(row["macro_f1"])}
                                       for row in activity["models"]]), hide_index=True, width="stretch")
            for column, row in zip(st.columns(2), activity["models"]):
                with column:
                    st.write(labels[row["model"]])
                    st.dataframe(pd.DataFrame(row["confusion"], index=["Факт: нет", "Факт: да"],
                                              columns=["Прогноз: нет", "Прогноз: да"]), width="stretch")
            st.caption("SMOTE нашёл больше повторно активных авторов: 197 вместо 7. "
                       "Ложных положительных прогнозов тоже стало больше: 4 147 вместо 1. Порог обеих моделей — 0,5.")
            if activity["curves"]:
                boundary_chart(activity["curves"])
                st.caption("Линия показывает, где оценка модели равна 0,5 при заданном числе и давности отзывов.")
            st.caption("Метка описывает только повторную активность в корпусе отзывов. "
                       "Соответствие постановки требованиям задания нужно уточнить у преподавателя.")
        with st.expander("Словарный сигнал и ручная проверка"):
            reviews = audit["reviews"]
            st.write(f"Частота словарного флага — {percent(reviews['flagged'] / reviews['with_text'], 4)} "
                     f"({number(reviews['flagged'])} из {number(reviews['with_text'])} текстов).")
            st.caption(f"Подготовлено {study['annotation']['size']} отзывов для разметки жалоб на проживание и доплаты. "
                       "Есть предварительная машинная разметка; человек её пока не проверял. "
                       "Частота флага в корпусе и качество его распознавания — разные показатели.")

        revision_path = OUTPUT / "revision_ru/revision_summary.json"
        revision = json.loads(revision_path.read_text(encoding="utf-8")) if revision_path.is_file() else {}
        if revision.get("files") == audit.get("files"):
            with st.expander("Активность авторов и доступность обратной связи"):
                st.write("Сегменты описывают историю за 365 дней перед 01.06.2026: один или несколько отзывов, "
                         "последний отзыв не более 90 или более 90 дней назад.")
                segments = pd.DataFrame(revision["segments"])
                segments["repeat_rate"] = segments.repeat_rate.map(lambda value: percent(value, 2))
                st.dataframe(segments.rename(columns={"segment": "История", "authors": "Авторов", "past_reviews": "Отзывов до даты",
                                "future_repeat": "Новый отзыв за 90 дней", "repeat_rate": "Доля повторной активности"}),
                             hide_index=True, width="stretch")
                st.caption("Это описательная сегментация по правилам. Она не измеряет лояльность и не является проверкой прогноза.")
                st.dataframe(pd.DataFrame(revision["review_availability"]).rename(columns={"activity": "Отзывов за год",
                    "listings": "Объявлений", "missing_price": "Без цены", "reviews": "Отзывов", "texts": "Текстов",
                    "flags": "Словарных флагов", "median_eur": "Медиана цены, €"}), hide_index=True, width="stretch")
                st.caption("Отзывы за 19.09.2025–18.09.2026. Частота словарных флагов не измеряет качество распознавания. "
                           "Без числа проживаний без отзыва нельзя измерить смещение участия гостей.")
            with st.expander("Почему правило с отзывами срабатывает редко"):
                funnel = revision["policy_funnel"]
                st.dataframe(pd.DataFrame({"Последовательное условие": ["Цена выше P75", "И не менее 5 текстов", "И не менее 2 флагов", "И доля флагов ≥ 20%"],
                    "Шагов воспроизведения": [funnel[key] for key in ("above_p75", "five_texts", "two_flags", "share_20")]}),
                    hide_index=True, width="stretch")
                st.caption("Два активных шага относятся к одной последовательности и одному срабатыванию. "
                           "Редкость рекомендации не доказывает её качество. Ценовой ориентир фиксирован снимком 2026 года.")

    st.caption(SNAPSHOT_NOTE)
    st.write("В календаре отсутствуют столбцы `price` и `adjusted_price`. Недоступная ночь может означать бронь или блокировку хозяином.")


def main():
    st.set_page_config(page_title="Цена жилья в Лиссабоне", page_icon="🏠", layout="wide")
    st.html(STYLES)
    st.title("Цена жилья в Лиссабоне", anchor=False)
    if not DATABASE.is_file():
        st.error("Локальная база ещё не подготовлена. Выполните `python prepare_data.py` по инструкции запуска.")
        st.stop()
    db_revision = DATABASE.stat().st_mtime_ns
    paths = [OUTPUT / f"{name}.json" for name in ("evaluation", "experiments")]
    revisions = [path.stat().st_mtime_ns if path.is_file() else 0 for path in paths]
    listings, audit, result, study = snapshot(db_revision, *revisions)
    if audit.get("schema") != 2:
        st.error("Для версии 0.2 обновите локальную базу командой `python prepare_data.py` по инструкции запуска.")
        st.stop()
    pricing_tab, replay_tab, check_tab = st.tabs(["Оценка цены", "Отзывы", "Проверка сборки"])
    listing_id, bounds, price = None, None, None

    with pricing_tab:
        st.header("Как моя цена выглядит на фоне аналогов?", anchor=False)
        with st.container(border=True, key="panel_offer"):
            mode = st.radio("Способ оценки", ["По объявлению", "Вручную"], horizontal=True, label_visibility="collapsed")
            default_price = None
            if mode == "По объявлению":
                find, choose, offer = st.columns([1.05, 1.45, 0.78])
                query = find.text_input("Найти объявление по ID или названию", placeholder="ID или часть названия").strip()
                if query:
                    found = (listings.listing_id.astype(str).eq(query) if query.isascii() and query.isdigit()
                             else listings.title.str.contains(query, case=False, regex=False))
                    options = listings.loc[found].head(40)
                    if int(found.sum()) > 40:
                        st.caption("Показаны первые 40 совпадений. Уточните название или введите ID.")
                else:
                    example = result.get("replay_demo", {}).get("listing_id")
                    options = listings.loc[listings.listing_id.eq(example)] if example else listings.head(1)
                if options.empty:
                    st.info("Совпадений нет. Проверьте ID или попробуйте часть названия.")
                else:
                    names = dict(zip(options.listing_id, options.title))
                    listing_id = choose.selectbox("Объявление", list(names), format_func=lambda item: f"{names[item]} · #{item}")
                    row = options.loc[options.listing_id.eq(listing_id)].iloc[0]
                    district, room, capacity = row.district, row.room_type, int(row.capacity)
                    if pd.notna(row.price_cents):
                        default_price = int(row.price_cents)
            else:
                area, kind, size, offer = st.columns([1.15, 1.13, 0.74, 1])
                districts = sorted(listings.district.unique())
                district = area.selectbox("Район", districts, index=districts.index("Arroios") if "Arroios" in districts else 0)
                room = kind.selectbox("Тип размещения", list(ROOM_LABELS), format_func=ROOM_LABELS.get)
                capacity = size.selectbox("Вместимость", sorted(listings.capacity.unique()), index=1)

            if mode == "Вручную" or listing_id is not None:
                peers = select_peers(listings, district, room, capacity, listing_id)
                bounds = get_benchmarks(peers)
                price_context = mode, listing_id
                if st.session_state.get("price_context") != price_context or "night_price" not in st.session_state:
                    st.session_state.price_context = price_context
                    st.session_state.night_price = (default_price or (bounds.median if bounds else 10_000)) / 100
                value = offer.number_input("Ваша цена за ночь, €", min_value=0.01, max_value=1_000_000.0,
                                           step=1.0, format="%.2f", key="night_price")
                price = round(value * 100)
                if listing_id is not None:
                    st.caption(f"Цена в снимке {money(default_price)} · введите сумму и нажмите Enter."
                               if default_price else "Цена в снимке отсутствует · задайте свою сумму и нажмите Enter.")
                else:
                    st.caption("Введите сумму и нажмите Enter.")
        if price is not None:
            price_result(peers, bounds, price, district, room, capacity, listing_id)

    with replay_tab:
        if listing_id is None:
            st.session_state.pop("replay_context", None)
            st.session_state.playing = False
            st.info("Выберите объявление на вкладке «Оценка цены», чтобы воспроизвести его отзывы.")
        else:
            replay_panel(review_events(db_revision, listing_id), listing_id, price, bounds, db_revision, study)
    with check_tab:
        evaluation_panel(audit, result, study)


if __name__ == "__main__":
    main()
