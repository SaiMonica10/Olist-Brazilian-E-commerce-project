"""Ops dashboard for late-delivery risk.

Usage (from the project root, with the database running):
    uv run streamlit run app/dashboard.py

Pages:
    * Risk queue: replay a past day and see its orders ranked by risk.
    * Score an order: type in an order's details and see its risk and reasons.

Decisions:
    * Replay: the data ends in Aug 2018, so ops picks a past day from the test
      months (May-Aug 2018), which the model never saw, and sees that day's
      orders as if they had just come in.
    * Ops can check only some orders, so the slider flags the riskiest X% of
      the day (default 10%, the share train.py evaluates).
    * "Top reason" = the feature with the biggest push toward late (SHAP).
    * The "actually late" column is hindsight, shown only to judge the demo;
      ops would not know it on the day.
    * Scores come from src/predict.py, the same code the API uses.
"""

import copy
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

# app/ is not a package of src/, so add src/ to the import path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from features import TARGET, TIME_COLUMN, load_features  # noqa: E402
from predict import EXAMPLE_ORDER, build_features, explain, known_values, score  # noqa: E402
from train import TEST_START, TOP_FRACTION  # noqa: E402

st.set_page_config(page_title="Late-delivery risk", page_icon="📦", layout="wide")

# Plain-language names for ops; the model's column names are kept in the code.
LABELS = {
    "promised_days": "Promised days",
    "distance_km": "Distance (km)",
    "n_items": "Items",
    "n_products": "Different products",
    "n_sellers": "Sellers in order",
    "total_price": "Price (R$)",
    "total_freight": "Freight (R$)",
    "freight_ratio": "Freight / price",
    "total_weight_g": "Weight (g)",
    "total_volume_cm3": "Volume (cm³)",
    "max_installments": "Installments",
    "purchase_month_of_year": "Month",
    "purchase_dow": "Weekday (1 = Mon)",
    "purchase_hour": "Hour of purchase",
    "seller_prior_orders": "Seller's past orders",
    "seller_prior_late_rate": "Seller's past late rate",
    "customer_state": "Customer state",
    "seller_state": "Seller state",
    "same_state": "Same state",
    "main_category": "Category",
    "main_payment_type": "Payment",
}

STATE_NAMES = {
    "AC": "Acre", "AL": "Alagoas", "AP": "Amapá", "AM": "Amazonas", "BA": "Bahia",
    "CE": "Ceará", "DF": "Distrito Federal", "ES": "Espírito Santo", "GO": "Goiás",
    "MA": "Maranhão", "MT": "Mato Grosso", "MS": "Mato Grosso do Sul", "MG": "Minas Gerais",
    "PA": "Pará", "PB": "Paraíba", "PR": "Paraná", "PE": "Pernambuco", "PI": "Piauí",
    "RJ": "Rio de Janeiro", "RN": "Rio Grande do Norte", "RS": "Rio Grande do Sul",
    "RO": "Rondônia", "RR": "Roraima", "SC": "Santa Catarina", "SP": "São Paulo",
    "SE": "Sergipe", "TO": "Tocantins",
}
STATE_FEATURES = {"customer_state", "seller_state"}
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


@st.cache_data(show_spinner="Loading orders from the database...")
def load_test_orders() -> pd.DataFrame:
    """Test-period orders (never seen by the model), with their risk scores."""
    df = load_features()
    df = df[df[TIME_COLUMN] >= TEST_START].reset_index(drop=True)
    df["risk"] = score(df)
    df["day"] = df[TIME_COLUMN].dt.date
    return df


def show_value(value) -> str:
    """A feature value formatted for people: 19.0 -> 19, 0.048 -> 0.048, 1968.3 -> 1,968."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "unknown"
    if isinstance(value, float):
        return f"{value:,.3g}" if abs(value) < 1 else f"{value:,.0f}"
    return str(value)


def show_feature(feature: str, value, bold: bool = False) -> str:
    """'Customer state: Maranhão (MA)', 'Promised days: 9', ..."""
    label = f"**{LABELS[feature]}**" if bold else LABELS[feature]
    if feature in STATE_FEATURES:
        return f"{label}: {STATE_NAMES.get(value, value)} ({value})"
    return f"{label}: {show_value(value)}"


def top_reason(row: pd.Series, pushes: pd.Series) -> str:
    """The feature pushing hardest toward late, shown with this order's value."""
    feature = pushes.idxmax()
    if pushes[feature] <= 0:
        return "nothing stands out"
    return show_feature(feature, row[feature])


def explain_order(order: pd.Series, shap_row) -> None:
    """The order's main reasons in words, then the waterfall chart."""
    pushes = pd.Series(shap_row.values, index=shap_row.feature_names).sort_values(ascending=False)
    up = [show_feature(f, order[f], bold=True) for f in pushes.index[:3] if pushes[f] > 0]
    down = [show_feature(f, order[f], bold=True) for f in pushes.index[::-1][:2] if pushes[f] < 0]
    st.markdown("🔺 **Raises risk:** " + (" · ".join(up) or "nothing"))
    st.markdown("🔻 **Lowers risk:** " + (" · ".join(down) or "nothing"))

    labelled = copy.copy(shap_row)  # same values, plain-language names for the chart
    labelled.feature_names = [LABELS[f] for f in shap_row.feature_names]
    shap.plots.waterfall(labelled, max_display=7, show=False)
    fig = plt.gcf()
    fig.set_size_inches(7, 3.6)
    st.pyplot(fig, width="stretch")
    plt.close("all")
    st.caption("Starts at the average order (bottom), then adds each feature's push. "
               "Red pushes toward late, blue toward on time (units: log-odds).")


def route_header(order: pd.Series, risk: float) -> None:
    a, b, c = st.columns(3)
    a.metric("Risk", f"{risk:.1%}")
    b.metric("Route", f"{order['seller_state']} → {order['customer_state']}")
    c.metric("Promised", f"{order['promised_days']:.0f} days")
    st.caption(f"From the seller in **{STATE_NAMES[order['seller_state']]}** "
               f"to the customer in **{STATE_NAMES[order['customer_state']]}**")


def load_orders_or_stop() -> pd.DataFrame:
    try:
        return load_test_orders()
    except Exception as error:  # most often: Docker / Postgres is not running
        st.error(f"Could not load orders from the database: {error}\n\n"
                 "Start it with `docker compose up -d`, then refresh.")
        st.stop()


# ---------------------------------------------------------------- Risk queue

def risk_queue() -> None:
    orders = load_orders_or_stop()
    days = sorted(orders["day"].unique())
    with st.sidebar:
        day = st.date_input("Day to replay", value=days[0], min_value=days[0], max_value=days[-1],
                            help="May–Aug 2018: months the model never saw.")
        share = st.slider("Share of orders ops can check", 1, 50, int(TOP_FRACTION * 100),
                          format="%d%%") / 100
        st.divider()
        st.markdown(f"🚩 = among the riskiest **{share:.0%}** of the day's orders: "
                    "**check these first** (e.g. contact the seller or carrier).")
        st.markdown("**Actually late?** is hindsight, shown only to judge the model. "
                    "Ops would not know it on the day.")

    st.title("📋 Risk queue")
    st.caption(f"Orders placed on **{day:%A %d %B %Y}**, riskiest first. "
               "Click a row to see why it got its score.")

    today = orders[orders["day"] == day].sort_values("risk", ascending=False).reset_index(drop=True)
    if today.empty:
        st.info("No delivered orders were placed on this day. Pick another day.")
        st.stop()

    n_flagged = max(1, round(share * len(today)))
    today["flag"] = np.arange(len(today)) < n_flagged

    shap_values = explain(today)
    pushes = pd.DataFrame(shap_values.values, columns=shap_values.feature_names)
    today["top_reason"] = [top_reason(today.loc[i], pushes.loc[i]) for i in today.index]

    late_flagged = int(today.loc[today["flag"], TARGET].sum())
    late_total = int(today[TARGET].sum())
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Orders placed", len(today), border=True)
    m2.metric("🚩 Flagged to check", n_flagged, border=True)
    m3.metric("Highest risk", f"{today['risk'].iloc[0]:.1%}", border=True)
    m4.metric("Late orders caught (hindsight)", f"{late_flagged} of {late_total}", border=True,
              help="How many of the day's late orders were in the flagged group. "
                   "Not known on the day.")

    table = pd.DataFrame({
        "": np.where(today["flag"], "🚩", ""),
        "Risk": today["risk"],
        "Order": today["order_id"].str[:8],
        "Time": today[TIME_COLUMN].dt.strftime("%H:%M"),
        "Route": today["seller_state"].astype(str).map(STATE_NAMES)
                 + " → " + today["customer_state"].astype(str).map(STATE_NAMES),
        "Promised": today["promised_days"].astype(int),
        "Top reason": today["top_reason"],
        "Actually late?": np.where(today[TARGET] == 1, "⚠️ late", "✅ on time"),
    })

    left, right = st.columns([3, 2], gap="medium")
    with left:
        selection = st.dataframe(
            table,
            hide_index=True,
            width="stretch",
            height=620,
            on_select="rerun",
            selection_mode="single-row",
            selection_default={"selection": {"rows": [0]}},
            key=f"queue-{day}",  # a new day starts with its riskiest order selected
            # Flag, risk and order stay pinned on the left when the table scrolls sideways.
            column_config={
                "": st.column_config.TextColumn(width=30, pinned=True),
                "Risk": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=0.5,
                                                        help="Late-delivery risk score", pinned=True),
                "Order": st.column_config.TextColumn(pinned=True),
                "Route": st.column_config.TextColumn(help="Seller state → customer state"),
                "Promised": st.column_config.NumberColumn(format="%d days"),
                "Actually late?": st.column_config.TextColumn(help="Hindsight, for the demo only"),
            },
        )
    with right:
        rows = selection.selection.rows
        i = rows[0] if rows else 0
        with st.container(border=True):
            st.markdown(f"#### Order `{today.loc[i, 'order_id'][:8]}`")
            route_header(today.loc[i], today.loc[i, "risk"])
            explain_order(today.loc[i], shap_values[i])


# ------------------------------------------------------------ Score an order

def score_an_order() -> None:
    st.title("🔍 Score an order")
    st.caption("Type in a new order's details, as known at checkout, to see its risk and why. "
               "Prefilled with the riskiest order of May–Aug 2018.")

    known = known_values()
    ex = EXAMPLE_ORDER
    states = sorted(STATE_NAMES)
    state_label = lambda code: f"{code} – {STATE_NAMES[code]}"  # noqa: E731

    form, result = st.columns([3, 2], gap="medium")
    with form, st.form("order"):
        st.markdown("**📍 Where**")
        a, b, c = st.columns(3)
        seller_state = a.selectbox("Seller state", states, states.index(ex["seller_state"]),
                                   format_func=state_label)
        customer_state = b.selectbox("Customer state", states, states.index(ex["customer_state"]),
                                     format_func=state_label)
        distance_km = c.number_input("Distance (km)", 0.0, 5000.0, ex["distance_km"])

        st.markdown("**🕒 When**")
        a, b, c, d = st.columns(4)
        promised_days = a.number_input("Promised days", 0, 200, ex["promised_days"],
                                       help="Delivery time shown to the customer at checkout.")
        month = b.selectbox("Month", range(1, 13), ex["purchase_month_of_year"] - 1,
                            format_func=lambda m: MONTHS[m - 1])
        weekday = c.selectbox("Weekday", range(1, 8), ex["purchase_dow"] - 1,
                              format_func=lambda d: WEEKDAYS[d - 1])
        hour = d.number_input("Hour", 0, 23, ex["purchase_hour"])

        st.markdown("**🛒 What**")
        a, b, c = st.columns(3)
        categories = [None] + known["main_category"]
        category = a.selectbox("Category", categories, categories.index(ex["main_category"]),
                               format_func=lambda c: "unknown" if c is None else c)
        price = b.number_input("Price (R$)", 0.0, 20000.0, ex["total_price"])
        freight = c.number_input("Freight (R$)", 0.0, 2000.0, ex["total_freight"])
        a, b, c = st.columns(3)
        n_items = a.number_input("Items", 1, 30, ex["n_items"])
        n_products = b.number_input("Different products", 1, 30, ex["n_products"])
        n_sellers = c.number_input("Sellers in order", 1, 10, ex["n_sellers"])
        a, b = st.columns(2)
        weight = a.number_input("Weight (g)", 0.0, 200000.0, float(ex["total_weight_g"]))
        volume = b.number_input("Volume (cm³)", 0.0, 2000000.0, float(ex["total_volume_cm3"]))

        st.markdown("**💳 Payment and 🏪 seller history**")
        a, b = st.columns(2)
        payment = a.selectbox("Payment", known["main_payment_type"],
                              known["main_payment_type"].index(ex["main_payment_type"]))
        installments = b.number_input("Installments", 0, 24, ex["max_installments"])
        a, b, c = st.columns(3)
        new_seller = a.checkbox("New seller", help="No delivered orders yet, so no track record.")
        prior_orders = b.number_input("Seller's past orders", 0, 5000, ex["seller_prior_orders"])
        prior_late = c.number_input("Seller's past late rate", 0.0, 1.0, ex["seller_prior_late_rate"],
                                    step=0.01, help="Share of past orders that were late, 0 to 1.")
        if new_seller:  # same as the data: no history means 0 orders and no late rate
            prior_orders, prior_late = 0, None

        submitted = st.form_submit_button("Score this order", type="primary", width="stretch")

    order = {
        "promised_days": promised_days, "distance_km": distance_km, "n_items": n_items,
        "n_products": n_products, "n_sellers": n_sellers, "total_price": price,
        "total_freight": freight, "total_weight_g": weight, "total_volume_cm3": volume,
        "max_installments": installments, "purchase_month_of_year": month,
        "purchase_dow": weekday, "purchase_hour": hour, "seller_prior_orders": prior_orders,
        "seller_prior_late_rate": prior_late, "customer_state": customer_state,
        "seller_state": seller_state, "main_category": category, "main_payment_type": payment,
    }
    X = build_features([order])
    risk = float(score(X)[0])

    with result, st.container(border=True):
        st.markdown("#### Result" + ("" if submitted else " (example order)"))
        route_header(X.iloc[0], risk)

        try:  # compare with real orders, if the database is available
            test_risks = load_test_orders()["risk"]
            riskier_than = (test_risks < risk).mean()
            flagged = riskier_than >= 1 - TOP_FRACTION
            st.markdown(f"Riskier than **{riskier_than:.0%}** of the orders placed in May–Aug 2018. "
                        + (f"🚩 It would be among the riskiest {TOP_FRACTION:.0%}: **check it.**"
                           if flagged else
                           f"Not in the riskiest {TOP_FRACTION:.0%}: no check needed."))
        except Exception:
            st.caption("Start the database to compare this order with real orders.")

        explain_order(X.iloc[0], explain(X)[0])


# ---------------------------------------------------------------- Navigation

if __name__ == "__main__":  # Streamlit runs the file as __main__; importing it (tests) does not
    with st.sidebar:
        st.header("📦 Late-delivery risk")

    page = st.navigation([
        st.Page(risk_queue, title="Risk queue", icon="📋", default=True),
        st.Page(score_an_order, title="Score an order", icon="🔍", url_path="score"),
    ])
    page.run()
