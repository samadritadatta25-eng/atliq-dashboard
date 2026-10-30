"""
AtliQ Grands - Revenue Leakage Dashboard (Streamlit web app)
Team: Samadrita Datta, Anika Sharma
Introduction to Python Programming (BBAT5E07), TAPMI Bengaluru

Run locally with:   streamlit run app.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# ------------------------------------------------------------------
# Page settings and chart colours
# ------------------------------------------------------------------
st.set_page_config(page_title="AtliQ Grands - Revenue Leakage Dashboard",
                   page_icon="🏨", layout="wide")

MAIN = "#2a78d6"        # blue   - normal values
HIGHLIGHT = "#eb6834"   # orange - the problem we want managers to notice
TEXT = "#52514e"

DATA_URL = "https://raw.githubusercontent.com/Harishankar97/atliq-hotels-analysis/main/Data/"
DATA_FILES = {                                   # dictionary: short name -> file name
    "bookings":   "fact_bookings.csv",
    "hotels":     "dim_hotels.csv",
    "rooms":      "dim_rooms.csv",
    "dates":      "dim_date.csv",
    "aggregated": "fact_aggregated_bookings.csv",
}
REALISED_SHARE = {"Checked Out": 1.0, "No Show": 1.0, "Cancelled": 0.4}
CURRENT_FEE = 0.40


# ------------------------------------------------------------------
# Step 1: load the data safely
# ------------------------------------------------------------------
@st.cache_data(show_spinner="Loading booking data...")
def load_tables():
    """Reads the 5 CSV files. Returns a dictionary of tables, or None if any file fails."""
    tables = {}
    for name, file_name in DATA_FILES.items():
        try:
            tables[name] = pd.read_csv(DATA_URL + file_name)
        except Exception as error:
            st.error(f"Could not load {file_name}: {error}")
            return None
    return tables


def standardise_date(text):
    """Converts '27-04-22' or '1/5/2022' into one format: '2022-04-27'."""
    text = str(text).strip().replace("/", "-")
    day, month, year = text.split("-")
    if len(year) == 2:
        year = "20" + year
    return f"{year}-{month.zfill(2)}-{day.zfill(2)}"


# ------------------------------------------------------------------
# Steps 2 and 3: clean and combine
# ------------------------------------------------------------------
@st.cache_data(show_spinner="Cleaning and combining the data...")
def prepare_data(tables):
    """Cleans the tables and returns (bookings table, occupancy table, hotel table)."""
    b = tables["bookings"].copy()

    # remove missing or impossible guest counts
    b = b[b["no_guests"].notnull() & (b["no_guests"] > 0)].copy()
    b["no_guests"] = b["no_guests"].astype(int)

    # two date formats -> one
    for column in ["booking_date", "check_in_date", "checkout_date"]:
        b[column] = pd.to_datetime(b[column].apply(standardise_date), format="%Y-%m-%d")

    # fix revenue typing errors using the business rule (cancelled guests pay 40%)
    share = b["booking_status"].map(REALISED_SHARE)
    expected = (b["revenue_realized"] / share).round()
    wrong = (b["revenue_generated"] - expected).abs() > 1
    b["revenue_generated"] = np.where(wrong, expected, b["revenue_generated"]).astype(int)

    b["booking_platform"] = b["booking_platform"].str.strip().str.title()

    hotels = tables["hotels"].copy()
    hotels["city"] = hotels["city"].replace({"Bangalore": "Bengaluru"})

    dates = tables["dates"].copy()
    dates.columns = [c.strip().lower().replace(" ", "_") for c in dates.columns]
    dates["day_type"] = dates["day_type"].str.replace("weekeday", "weekday")
    dates["date"] = pd.to_datetime(dates["date"], format="%d-%b-%y")

    # combine bookings with hotels, rooms and the calendar
    df = b.merge(hotels, on="property_id", how="left")
    df = df.merge(tables["rooms"], left_on="room_category", right_on="room_id", how="left")
    df = df.merge(dates[["date", "day_type"]], left_on="check_in_date", right_on="date", how="left")
    df["lead_time_days"] = (df["check_in_date"] - df["booking_date"]).dt.days
    df["is_cancelled"] = np.where(df["booking_status"] == "Cancelled", 1, 0)
    df["is_no_show"] = np.where(df["booking_status"] == "No Show", 1, 0)
    df["revenue_lost"] = df["revenue_generated"] - df["revenue_realized"]

    # daily occupancy table
    occ = tables["aggregated"].copy()
    usual = occ.groupby(["property_id", "room_category"])["capacity"].transform("median")
    occ["capacity"] = occ["capacity"].fillna(usual)
    occ["check_in_date"] = pd.to_datetime(occ["check_in_date"], format="%d-%b-%y")
    occ = occ.merge(hotels, on="property_id", how="left")
    occ = occ.merge(dates[["date", "day_type"]], left_on="check_in_date", right_on="date", how="left")
    occ = occ[occ["successful_bookings"] <= occ["capacity"]].copy()   # remove impossible days

    # one row per hotel: average rating and occupancy
    ratings = df.groupby(["property_id", "property_name", "city"])["ratings_given"].mean()
    rooms_used = occ.groupby(["property_id", "property_name", "city"])[["successful_bookings", "capacity"]].sum()
    hotel_table = pd.DataFrame({
        "avg_rating": ratings.round(2),
        "occupancy_pct": (rooms_used["successful_bookings"] / rooms_used["capacity"] * 100).round(1),
    }).reset_index()

    return df, occ, hotel_table


def occupancy_rate(data):
    """Occupancy % = rooms booked / rooms available x 100"""
    if data["capacity"].sum() == 0:
        return 0.0
    return round(data["successful_bookings"].sum() / data["capacity"].sum() * 100, 1)


def style_chart(ax, title, xlabel="", ylabel=""):
    ax.set_title(title, fontsize=12, fontweight="bold", loc="left", pad=10)
    ax.set_xlabel(xlabel, color=TEXT)
    ax.set_ylabel(ylabel, color=TEXT)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(colors=TEXT)


# ------------------------------------------------------------------
# Load everything
# ------------------------------------------------------------------
tables = load_tables()
if tables is None:
    st.stop()
df, occupancy, hotel_table = prepare_data(tables)

# ------------------------------------------------------------------
# Sidebar: the manager's choices
# ------------------------------------------------------------------
st.sidebar.header("Choose a view")
city_options = ["All cities"] + sorted(set(df["city"]))
city = st.sidebar.selectbox("City", city_options)

st.sidebar.header("Policy simulator")
new_fee = st.sidebar.slider("Cancellation fee (%)", min_value=40, max_value=80, value=50, step=5)
extra_points = st.sidebar.slider("Rise in weekday occupancy (points)", 0, 20, 5)

if city == "All cities":
    data, occ = df, occupancy
    hotels = hotel_table
else:
    data = df[df["city"] == city]
    occ = occupancy[occupancy["city"] == city]
    hotels = hotel_table[hotel_table["city"] == city]

# ------------------------------------------------------------------
# Header and headline numbers
# ------------------------------------------------------------------
st.title("AtliQ Grands: Revenue Leakage Dashboard")
st.caption(f"{city} · May to July 2022 · Samadrita Datta and Anika Sharma · "
           "Introduction to Python Programming, TAPMI Bengaluru")

lost_cr = data["revenue_lost"].sum() / 10_000_000
weekday_occ = occupancy_rate(occ[occ["day_type"] == "weekday"])
weekend_occ = occupancy_rate(occ[occ["day_type"] == "weekend"])
k1, k2, k3, k4 = st.columns(4)
k1.metric("Bookings", f"{len(data):,}")
k2.metric("Revenue lost", f"₹{lost_cr:.2f} cr")
k3.metric("Cancelled", f"{data['is_cancelled'].mean() * 100:.1f}%")
k4.metric("No-shows", f"{data['is_no_show'].mean() * 100:.1f}%")
k5, k6, k7, k8 = st.columns(4)
k5.metric("Weekday occupancy", f"{weekday_occ:.0f}%")
k6.metric("Weekend occupancy", f"{weekend_occ:.0f}%")
k7.metric("Weekend vs weekday gap", f"{weekend_occ - weekday_occ:.0f} pts")
k8.metric("Average rating", f"{data['ratings_given'].mean():.2f} / 5")
tab1, tab2, tab3, tab4 = st.tabs(["Cancellations", "Empty weekday rooms",
                                  "Hotel ratings", "Policy simulator"])
# ------------------------------------------------------------------
# Tab 1: cancellations
# ------------------------------------------------------------------
with tab1:
    left, right = st.columns(2)

    with left:
        fig, ax = plt.subplots(figsize=(7, 3.2))
        collected = data["revenue_realized"].sum() / 10_000_000
        ax.barh(["Booking value"], [collected], color=MAIN, height=0.5)
        ax.barh(["Booking value"], [lost_cr], left=[collected], color=HIGHLIGHT, height=0.5,
                edgecolor="white", linewidth=2)
        ax.text(collected / 2, 0, f"Collected\nRs {collected:.1f} cr", ha="center", va="center",
                color="white", fontweight="bold")
        ax.text(collected + lost_cr / 2, 0, f"Lost\nRs {lost_cr:.1f} cr", ha="center", va="center",
                color="white", fontweight="bold", fontsize=9)
        ax.set_yticks([])
        style_chart(ax, "Where the booking value went", xlabel="Rs crore")
        st.pyplot(fig)

    with right:
        fig, ax = plt.subplots(figsize=(7, 3.6))
        rates = (data.groupby("booking_platform")["is_cancelled"].mean() * 100).sort_values()
        bars = ax.barh(rates.index, rates.values, color=MAIN, height=0.6)
        ax.axvline(data["is_cancelled"].mean() * 100, color=HIGHLIGHT, linestyle="--", linewidth=2)
        for bar, value in zip(bars, rates.values):
            ax.text(value - 0.5, bar.get_y() + bar.get_height() / 2, f"{value:.1f}%",
                    ha="right", va="center", color="white", fontsize=9, fontweight="bold")
        ax.set_xlim(0, 30)
        style_chart(ax, "Cancellation rate by platform (dashed = average)", xlabel="%")
        st.pyplot(fig)

    st.info("Every platform cancels about 25%, so the problem is the chain-wide cancellation "
            "policy rather than any one partner. Platforms with the most bookings lose the most money.")

# ------------------------------------------------------------------
# Tab 2: empty weekday rooms
# ------------------------------------------------------------------
with tab2:
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    day_names = occ["check_in_date"].dt.day_name()
    values = [occupancy_rate(occ[day_names == day]) for day in days]
    colours = [MAIN if day in ("Saturday", "Sunday") else HIGHLIGHT for day in days]

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar([day[:3] for day in days], values, color=colours, width=0.6)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 1.5, f"{value:.0f}%",
                ha="center", color=TEXT, fontsize=10)
    ax.set_ylim(0, 100)
    style_chart(ax, "Occupancy by day of week (orange = weekday)", ylabel="% of rooms filled")
    st.pyplot(fig)

    st.info(f"Weekdays are only {weekday_occ:.0f}% full against {weekend_occ:.0f}% at weekends. "
            "AtliQ is missing business travellers: corporate contracts and weekday packages would help.")

# ------------------------------------------------------------------
# Tab 3: hotel ratings
# ------------------------------------------------------------------
with tab3:
    ranked = hotels.sort_values("occupancy_pct", ascending=False).tail(10)
    labels = (ranked["property_name"].str.replace("Atliq ", "") + ", " + ranked["city"]
              + " #" + ranked["property_id"].astype(str))
    colours = [HIGHLIGHT if r < 2.5 else MAIN for r in ranked["avg_rating"]]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    bars = ax.barh(labels, ranked["occupancy_pct"], color=colours, height=0.6)
    for bar, rating in zip(bars, ranked["avg_rating"]):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2, f"rating {rating:.1f}",
                va="center", fontsize=9, color=TEXT)
    ax.set_xlim(0, 85)
    style_chart(ax, "Emptiest hotels (orange = rated below 2.5)", xlabel="% of rooms filled")
    st.pyplot(fig)

    poor = hotels[hotels["avg_rating"] < 2.5]
    if len(poor) == 0:
        st.success("No poorly rated hotels in this view.")
    else:
        names = ", ".join(poor["property_name"] + " (" + poor["city"] + ")")
        st.info(f"{len(poor)} poorly rated hotel(s) here: {names}. They are only about "
                f"{poor['occupancy_pct'].mean():.0f}% full. Improving guest experience there is "
                "the clearest way to fill more rooms.")
    st.dataframe(hotels.sort_values("avg_rating").reset_index(drop=True), use_container_width=True)

# ------------------------------------------------------------------
# Tab 4: policy simulator
# ------------------------------------------------------------------
with tab4:
    cancelled_value = data.loc[data["is_cancelled"] == 1, "revenue_generated"].sum()
    fee_extra = cancelled_value * (new_fee / 100 - CURRENT_FEE)
    avg_booking_value = data.loc[data["booking_status"] == "Checked Out", "revenue_generated"].mean()
    weekday_capacity = occ.loc[occ["day_type"] == "weekday", "capacity"].sum()
    weekday_extra = weekday_capacity * extra_points / 100 * avg_booking_value

    c1, c2 = st.columns(2)
    c1.metric(f"Extra revenue from a {new_fee}% cancellation fee",
              f"₹{fee_extra / 10_000_000:.2f} cr per quarter",
              f"about ₹{fee_extra * 4 / 10_000_000:.1f} cr a year")
    c2.metric(f"Extra revenue from +{extra_points} points weekday occupancy",
              f"₹{weekday_extra / 10_000_000:.2f} cr per quarter",
              f"about ₹{weekday_extra * 4 / 10_000_000:.1f} cr a year")

    st.caption("Move the sliders in the sidebar to test other policies. These are rough estimates: "
               "they assume the same bookings would still be made under a stricter policy.")

st.divider()
st.caption("Data: AtliQ Grands hospitality dataset (Codebasics, simulated). "
           "Built with Python, pandas, NumPy, Matplotlib and Streamlit.")
