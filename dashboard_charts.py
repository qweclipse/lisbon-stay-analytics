import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from pricing import ROOM_LABELS


MINT = "#8cebc6"
AMBER = "#ffd174"
TEXT = "#eaf0f4"
MUTED = "#b1bdc8"
GRID = "#39434d"


def draw(chart):
    styled = (chart.configure(background="transparent", font="Arial")
              .configure_view(stroke=None)
              .configure_axis(labelColor=MUTED, titleColor=TEXT, gridColor=GRID, domainColor=GRID,
                              tickColor=GRID, labelFontSize=12, titleFontSize=12, titlePadding=10)
              .configure_legend(labelColor=TEXT, titleColor=TEXT, labelFontSize=12))
    st.altair_chart(styled, theme=None, width="stretch")


def market_domain(peers, price):
    low, high = (int(round(value)) for value in np.percentile(peers, [2, 98]))
    span = max(2000, high - low)
    padding = max(500, span // 15)
    show_price = low - span <= price <= high + span
    if show_price:
        low, high = min(low, price), max(high, price)
    return max(0, low - padding), high + padding, show_price


def market_chart(peers, bounds, price):
    left, right, show_price = market_domain(peers, price)
    counts, edges = np.histogram(np.asarray(peers) / 100, bins=24, range=(left / 100, right / 100))
    x_scale = alt.Scale(domain=[left / 100, right / 100], nice=False, zero=False)
    y_scale = alt.Scale(domain=[0, max(1, int(counts.max())) * 1.09], nice=False)
    x = alt.X("start:Q", title="Цена за ночь, €", scale=x_scale, axis=alt.Axis(tickCount=6, format=".0f"))
    y = alt.Y("count:Q", title="Объявлений", scale=y_scale, axis=alt.Axis(tickCount=4, tickMinStep=1))
    bars = alt.Chart(pd.DataFrame({"start": edges[:-1], "end": edges[1:], "count": counts})).mark_rect(
        color=MINT, opacity=0.82, stroke="#23282e", strokeWidth=1,
    ).encode(x=x, x2="end:Q", y=y, y2=alt.datum(0), tooltip=[alt.Tooltip("start:Q", title="От, €", format=".2f"),
                                            alt.Tooltip("end:Q", title="До, €", format=".2f"),
                                            alt.Tooltip("count:Q", title="Аналогов")])
    band = alt.Chart(pd.DataFrame({"start": [bounds.p25 / 100], "end": [bounds.p75 / 100]})).mark_rect(
        color="#4ba78a", opacity=0.24,
    ).encode(x=x, x2="end:Q", y=alt.value(0), y2=alt.value("height"))
    median = alt.Chart(pd.DataFrame({"start": [bounds.median / 100]})).mark_rule(
        color=TEXT, strokeDash=[5, 4], strokeWidth=2,
    ).encode(x=x, tooltip=alt.Tooltip("start:Q", title="Медиана, €", format=".2f"))
    chart = band + bars + median
    if show_price:
        user_price = alt.Chart(pd.DataFrame({"start": [price / 100]})).mark_rule(
            color=AMBER, strokeWidth=3,
        ).encode(x=x, tooltip=alt.Tooltip("start:Q", title="Ваша цена, €", format=".2f"))
        chart += user_price
    draw(chart.properties(height=270, description="Распределение цен аналогов. Полоса выделяет типичный диапазон. "
                                                    "Пунктир отмечает медиану, жёлтая линия — цену пользователя."))
    return int(len(peers) - counts.sum()), show_price


def timeline_chart(history):
    data = pd.DataFrame([{"event": i, "day": item.day.isoformat(), "texts": state.texts,
                          "share": state.share, "fired": state.fired}
                         for i, (item, state) in enumerate(history, 1)])
    x = alt.X("event:Q", scale=alt.Scale(domain=[1, max(2, len(history))], nice=False, zero=False),
              title="Номер события по дате публикации", axis=alt.Axis(tickMinStep=1, tickCount=6))
    base = alt.Chart(data).encode(x=x)
    marker = base.transform_filter(alt.datum.fired).mark_rule(color=AMBER, strokeDash=[5, 3], strokeWidth=2)
    text_line = base.mark_line(interpolate="step-after", color=MINT, point=alt.OverlayMarkDef(size=28)).encode(
        y=alt.Y("texts:Q", title="Текстов", axis=alt.Axis(tickCount=3, tickMinStep=1)),
        tooltip=[alt.Tooltip("event:Q", title="Событие"), alt.Tooltip("day:N", title="Дата публикации"),
                 alt.Tooltip("texts:Q", title="Текстов в окне")])
    share_line = base.mark_line(interpolate="step-after", color=MINT, point=alt.OverlayMarkDef(size=28)).encode(
        y=alt.Y("share:Q", title="Доля замечаний", axis=alt.Axis(format=".0%", tickCount=3)),
        tooltip=[alt.Tooltip("event:Q", title="Событие"), alt.Tooltip("day:N", title="Дата публикации"),
                 alt.Tooltip("share:Q", title="Доля замечаний", format=".1%")])
    draw((text_line + marker).properties(height=155))
    draw((share_line + marker).properties(height=155))


def error_chart(score):
    data = pd.DataFrame({"method": ["Общая медиана", "Медиана аналогов"],
                         "mae": [score["global_median_mae_eur"], score["mae_eur"]]})
    data["value"] = data.mae.map(lambda value: f"{value:.2f}".replace(".", ",") + " €")
    base = alt.Chart(data).encode(
        x=alt.X("mae:Q", title="MAE, €", scale=alt.Scale(domain=[0, 1.25 * data.mae.max()]), axis=alt.Axis(tickCount=5)),
        y=alt.Y("method:N", title=None, sort=data.method.tolist(), axis=alt.Axis(labelLimit=150)),
    )
    bars = base.mark_bar(size=24, cornerRadiusEnd=4).encode(color=alt.Color(
        "method:N", scale=alt.Scale(domain=data.method.tolist(), range=["#8998a8", MINT]), legend=None))
    draw((bars + base.mark_text(align="left", dx=7, color=TEXT, fontSize=12).encode(text="value:N")).properties(height=130))


def availability_chart(groups):
    data = pd.DataFrame(groups)
    data["room"] = data.room_type.map(ROOM_LABELS)
    data["rate"] = 1 - data.assessed / data.test
    data["value"] = data.rate.map(lambda value: f"{value:.1%}".replace(".", ","))
    base = alt.Chart(data).encode(
        x=alt.X("rate:Q", scale=alt.Scale(domain=[0, 1], nice=False), title="Доля без оценки",
                axis=alt.Axis(tickCount=5, format=".0%")),
        y=alt.Y("room:N", title=None, sort="-x", axis=alt.Axis(labelLimit=145)),
    )
    bars = base.mark_bar(color="#a37b48", size=20, cornerRadiusEnd=4).encode(
        tooltip=[alt.Tooltip("room:N", title="Тип размещения"), alt.Tooltip("test:Q", title="В тесте"),
                 alt.Tooltip("assessed:Q", title="Оценено"), alt.Tooltip("value:N", title="Без оценки")],
    )
    labels = base.encode(text="value:N")
    inside = labels.transform_filter(alt.datum.rate >= 0.7).mark_text(align="right", dx=-6, color=TEXT, fontSize=11)
    outside = labels.transform_filter(alt.datum.rate < 0.7).mark_text(align="left", dx=6, color=TEXT, fontSize=11)
    draw((bars + inside + outside).properties(height=195))


def boundary_chart(curves):
    data = pd.DataFrame(curves)
    labels = {"baseline": "Обычная модель", "smote": "Модель со SMOTE"}
    data["method"] = data.model.map(labels)
    draw(alt.Chart(data).mark_line(strokeWidth=3).encode(
        x=alt.X("reviews:Q", title="Отзывы за предыдущие 365 дней", scale=alt.Scale(domain=[1, 15]),
                axis=alt.Axis(tickCount=6, tickMinStep=1)),
        y=alt.Y("days:Q", title="Дней с последнего отзыва", scale=alt.Scale(domain=[1, 365])),
        color=alt.Color("method:N", title=None, scale=alt.Scale(domain=list(labels.values()), range=["#a5b2c0", MINT]),
                        legend=alt.Legend(orient="bottom")),
        strokeDash=alt.StrokeDash("method:N", legend=None),
    ).properties(height=220))
