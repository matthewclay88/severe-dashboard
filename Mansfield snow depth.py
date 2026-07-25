"""
Mount Mansfield Snow Depth Status
----------------------------------

Fetches the NWS BTV snow-depth time series for Mount Mansfield (current
season, climatological average, and the historical max/min envelope)
and computes the latest observed depth compared to normal for that
date, plus a season-chart PNG matching the one on
https://www.weather.gov/btv/recreation.

Source format note: despite the .xml extension, these are not real XML
data documents - each is a thin <data><text>...</text></data> wrapper
around a JavaScript array literal of [Date.UTC(y,m,d), depth_inches]
pairs (as used directly by a Highcharts series). Date.UTC's month
argument is 0-indexed (0=Jan), unlike Python's date().month.

Data sources:
    https://www.weather.gov/source/btv/rec/mmn/2025-2026depth.xml
    https://www.weather.gov/source/btv/rec/mmn/avgdepth.xml
    https://www.weather.gov/source/btv/rec/mmn/maxdepth.xml
    https://www.weather.gov/source/btv/rec/mmn/mindepth.xml

Requirements:
    pip install requests matplotlib
"""

import json
import os
import re
from datetime import date, datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import requests

CURRENT_URL = "https://www.weather.gov/source/btv/rec/mmn/2025-2026depth.xml"
AVERAGE_URL = "https://www.weather.gov/source/btv/rec/mmn/avgdepth.xml"
MAX_URL = "https://www.weather.gov/source/btv/rec/mmn/maxdepth.xml"
MIN_URL = "https://www.weather.gov/source/btv/rec/mmn/mindepth.xml"

HEADERS = {
    "User-Agent": (
        "MountMansfieldSnowDepth/1.0 (dashboard status card)"
    ),
}

REPO_OUTPUT_DIR = "outputs"

STATUS_OUTPUT_FILE = os.path.join(REPO_OUTPUT_DIR, "vt_snow_depth_status.json")
CHART_OUTPUT_FILE = os.path.join(REPO_OUTPUT_DIR, "vt_snow_depth_chart.png")


def fetch_depth_series(url):
    """
    Download one depth-series file and parse it into a
    {date: depth_inches} dict.
    """

    response = requests.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()

    text = response.text

    match = re.search(r"<text>(.*)</text>", text, re.DOTALL)

    if not match:
        raise ValueError(f"Could not find <text> payload in {url}")

    inner = match.group(1)

    pairs = re.findall(
        r"Date\.UTC\((\d+),(\d+),(\d+)(?:,\d+)?\),\s*(-?\d+(?:\.\d+)?)",
        inner,
    )

    records = {}

    for year_str, month_str, day_str, value_str in pairs:

        year = int(year_str)
        month = int(month_str) + 1  # JS Date.UTC month is 0-indexed
        day = int(day_str)

        try:
            record_date = date(year, month, day)
        except ValueError:
            # A handful of source rows have used an invalid day (e.g.
            # day 31 in a 30-day month) - skip rather than guess.
            continue

        records[record_date] = float(value_str)

    return records


def latest_observed_date(records, as_of):
    """
    Most recent date in `records` that is on or before `as_of`.
    """

    valid_dates = [d for d in records if d <= as_of]

    if not valid_dates:
        raise ValueError("No observed dates on or before as_of.")

    return max(valid_dates)


def normal_for_date(average_records, target_date):
    """
    Climatological average depth for the given month/day, regardless
    of which year the average series happens to encode it under.
    """

    for record_date, value in average_records.items():

        if (
            record_date.month == target_date.month
            and record_date.day == target_date.day
        ):
            return value

    return None


def build_snow_depth_status(as_of=None):
    """
    Fetch both series and return a dict describing the current snow
    depth status, ready to serialize to JSON for a dashboard card.
    """

    as_of = as_of or datetime.now(timezone.utc).date()

    current_series = fetch_depth_series(CURRENT_URL)
    average_series = fetch_depth_series(AVERAGE_URL)

    obs_date = latest_observed_date(current_series, as_of)
    current_depth = current_series[obs_date]

    normal_depth = normal_for_date(average_series, obs_date)

    if normal_depth is None:

        departure = None
        departure_text = "Normal unavailable"

    else:

        departure = current_depth - normal_depth

        if abs(departure) < 0.5:
            departure_text = "Near normal"
        elif departure > 0:
            departure_text = f"+{departure:.0f} in above normal"
        else:
            departure_text = f"{departure:.0f} in below normal"

    return {
        "station": "Mount Mansfield Stake",
        "as_of_date": obs_date.isoformat(),
        "current_depth_in": current_depth,
        "normal_depth_in": normal_depth,
        "departure_in": departure,
        "departure_text": departure_text,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def plot_snow_depth_chart(current_series, average_series, max_series, min_series):
    """
    Render the Max/Min/Avg/Current-season depth lines as a single PNG,
    matching the season chart on weather.gov/btv/recreation.
    """

    def to_xy(series):
        dates = sorted(series.keys())
        return dates, [series[d] for d in dates]

    max_x, max_y = to_xy(max_series)
    min_x, min_y = to_xy(min_series)
    avg_x, avg_y = to_xy(average_series)
    cur_x, cur_y = to_xy(current_series)

    fig, ax = plt.subplots(figsize=(9.5, 3.2))

    ax.plot(max_x, max_y, color="#4dabf7", linewidth=1.5, label="Max Snow Depth")
    ax.plot(min_x, min_y, color="#5c5cd6", linewidth=1.5, label="Min Snow Depth")
    ax.plot(avg_x, avg_y, color="#37b24d", linewidth=1.5, label="Avg Snow Depth")
    ax.plot(cur_x, cur_y, color="#e03131", linewidth=2.5, label="Current Depth")

    ax.set_ylabel("Inches", fontsize=9)
    ax.set_ylim(bottom=0)

    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))

    ax.tick_params(labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#e9ecef", linewidth=0.8)
    ax.set_axisbelow(True)

    ax.legend(
        loc="upper left", fontsize=7.5, frameon=False, ncol=4,
        bbox_to_anchor=(0.0, 1.18),
    )

    fig.tight_layout()

    plt.savefig(CHART_OUTPUT_FILE, dpi=175)
    plt.close(fig)

    print(f"Saved snow depth chart to: {CHART_OUTPUT_FILE}")


def main():

    os.makedirs(REPO_OUTPUT_DIR, exist_ok=True)

    current_series = fetch_depth_series(CURRENT_URL)
    average_series = fetch_depth_series(AVERAGE_URL)
    max_series = fetch_depth_series(MAX_URL)
    min_series = fetch_depth_series(MIN_URL)

    as_of = datetime.now(timezone.utc).date()
    obs_date = latest_observed_date(current_series, as_of)
    current_depth = current_series[obs_date]
    normal_depth = normal_for_date(average_series, obs_date)

    if normal_depth is None:

        departure = None
        departure_text = "Normal unavailable"

    else:

        departure = current_depth - normal_depth

        if abs(departure) < 0.5:
            departure_text = "Near normal"
        elif departure > 0:
            departure_text = f"+{departure:.0f} in above normal"
        else:
            departure_text = f"{departure:.0f} in below normal"

    status = {
        "station": "Mount Mansfield Stake",
        "as_of_date": obs_date.isoformat(),
        "current_depth_in": current_depth,
        "normal_depth_in": normal_depth,
        "departure_in": departure,
        "departure_text": departure_text,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    print(json.dumps(status, indent=2))

    with open(STATUS_OUTPUT_FILE, "w") as f:
        json.dump(status, f, indent=2)

    print(f"\nSaved status to {STATUS_OUTPUT_FILE}")

    plot_snow_depth_chart(current_series, average_series, max_series, min_series)


if __name__ == "__main__":
    main()
