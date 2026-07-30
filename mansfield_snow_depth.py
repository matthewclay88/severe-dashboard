"""
Mount Mansfield Snow Depth
----------------------------------

Two independent data sources, kept separate on purpose:

1. CHART (Max/Min/Avg/Current-season lines) - the 4 NWS BTV feeds,
   maintained by NWS, matching the season chart on
   https://www.weather.gov/btv/recreation. Unchanged.

2. CURRENT DEPTH + DEPARTURE FROM NORMAL - a local station-history
   file, `mmnv1.json`, that lives in this repo alongside this script.
   It's a long-format daily record (one [date, depth_inches] pair per
   calendar day) for the Mount Mansfield stake going back to 1954,
   sourced from IEM (station id MMNV1). This replaces the old
   approach of fetching Matt Parrilla's wide-format CSV
   (https://matthewparrilla.com/mansfield-stake/) over the network -
   same underlying stake, same idea (real season-by-season history
   instead of one blended climatological average curve), but read
   from disk instead of downloaded each run.

   Because mmnv1.json is long-format (no precomputed "Average Season"
   row the way the old CSV had one), the climatological normal,
   record high/low, and rank for a given day-of-season are computed
   here directly from the full history: for a given "M/D" label,
   gather that day's depth across every season on record, then take
   the mean of every *other* season as the normal (so the current,
   possibly-incomplete season doesn't bias its own baseline). Record
   high/low and rank still consider every season, including the
   current one, since a live record should be able to show up as a
   record.

Source format notes:
    - The NWS .xml files are not real XML - each is a thin
      <data><text>...</text></data> wrapper around a JS array literal
      of [Date.UTC(y,m,d), depth_inches] pairs. Date.UTC's month is
      0-indexed (0=Jan), unlike Python's date().month.
    - mmnv1.json is {"meta": {...station info...}, "data": [[date,
      value], ...]}, one row per calendar day, "YYYY-MM-DD" dates,
      string values ("M" = missing, "T" = trace, else a number).
      Both "M" and "T" are treated as unreported, same as the old
      CSV's handling of blank/non-numeric cells.

Data sources:
    https://www.weather.gov/source/btv/rec/mmn/2025-2026depth.xml
    https://www.weather.gov/source/btv/rec/mmn/avgdepth.xml
    https://www.weather.gov/source/btv/rec/mmn/maxdepth.xml
    https://www.weather.gov/source/btv/rec/mmn/mindepth.xml
    mmnv1.json (local, same directory as this script)

Requirements:
    pip install requests matplotlib
"""

import json
import os
import re
from datetime import date, datetime, timedelta, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import requests

# ---- Chart data sources (NWS, unchanged) ----

CURRENT_URL = "https://www.weather.gov/source/btv/rec/mmn/2025-2026depth.xml"
AVERAGE_URL = "https://www.weather.gov/source/btv/rec/mmn/avgdepth.xml"
MAX_URL = "https://www.weather.gov/source/btv/rec/mmn/maxdepth.xml"
MIN_URL = "https://www.weather.gov/source/btv/rec/mmn/mindepth.xml"

# ---- Current depth / departure data source (local station history) ----
#
# mmnv1.json lives at the root of the severe-dashboard repo, alongside
# this script:
# https://github.com/matthewclay88/severe-dashboard/blob/main/mmnv1.json
# Resolved from this script's own location (not the process cwd) so it
# still finds the file correctly if invoked from a different working
# directory, e.g. a GitHub Actions step that cd's elsewhere first.

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_HISTORY_JSON_PATH = os.path.join(SCRIPT_DIR, "mmnv1.json")
LOCAL_HISTORY_MISSING_VALUES = {"", "M", "T"}

HYD_URL = "https://forecast.weather.gov/product.php"
HYD_PARAMS = {"site": "BTV", "issuedby": "BTV", "product": "HYD", "format": "txt", "glossary": "0"}
HYD_MAX_VERSION_FALLBACK = 3

HYD_COLUMN_LABELS = ["24 Hrs", "Max", "Min", "Cur", "Weather", "New", "Total", "SWE"]
HYD_COLUMN_FIELD_NAMES = {
    "24 Hrs": "precip_24hr",
    "Max": "temp_max",
    "Min": "temp_min",
    "Cur": "temp_cur",
    "Weather": "present_weather",
    "New": "snow_new",
    "Total": "snow_total",
    "SWE": "snow_swe",
}

HEADERS = {
    "User-Agent": (
        "MountMansfieldSnowDepth/1.0 (dashboard status card)"
    ),
}

REPO_OUTPUT_DIR = "outputs"

STATUS_OUTPUT_FILE = os.path.join(REPO_OUTPUT_DIR, "vt_snow_depth_status.json")
CHART_OUTPUT_FILE = os.path.join(REPO_OUTPUT_DIR, "vt_snow_depth_chart.png")


# =====================================================================
# CHART (NWS feeds - unchanged)
# =====================================================================

def fetch_depth_series(url):
    """
    Download one NWS depth-series file and parse it into a
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


# =====================================================================
# CURRENT DEPTH / DEPARTURE (local mmnv1.json station history)
# =====================================================================

def load_local_snow_depth_history(path=LOCAL_HISTORY_JSON_PATH):
    """
    Read mmnv1.json and return {date: depth_inches_or_None}. "M"
    (missing) and "T" (trace) both map to None, same treatment as the
    old CSV's blank/non-numeric cells - unreported, not zero.
    """

    with open(path, "r") as f:
        payload = json.load(f)

    records = {}

    for date_str, value_str in payload.get("data", []):

        try:
            record_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError:
            continue

        value_str = (value_str or "").strip()

        if value_str in LOCAL_HISTORY_MISSING_VALUES:
            records[record_date] = None
            continue

        try:
            records[record_date] = float(value_str)
        except ValueError:
            records[record_date] = None

    return records


def season_label_for_date(d):
    """
    Ski-season label ("2025-2026") the given calendar date falls in.
    Mirrors the old CSV's tracked range: Sep 1 - Jun 30. Returns None
    for Jul/Aug (the off-season gap between one season ending and the
    next starting).
    """

    if d.month >= 9:
        return f"{d.year}-{d.year + 1}"

    if d.month <= 6:
        return f"{d.year - 1}-{d.year}"

    return None


def season_start_date(season_label):
    start_year = int(season_label.split("-")[0])
    return date(start_year, 9, 1)


def season_end_date(season_label):
    end_year = int(season_label.split("-")[1])
    return date(end_year, 6, 30)


def mmdd_label(d):
    """Day-of-season label ("9/1".."6/30"), matching the old CSV's
    column headers but derived directly from the date."""
    return f"{d.month}/{d.day}"


def organize_by_season(records):
    """
    {season_label: {mmdd_label: depth_inches}}, numeric values only -
    the per-season, per-day-of-season lookup the old wide CSV gave us
    for free via its column layout.
    """

    season_rows = {}

    for d, value in records.items():

        if value is None:
            continue

        season = season_label_for_date(d)

        if season is None:
            continue

        season_rows.setdefault(season, {})[mmdd_label(d)] = value

    return season_rows


def latest_reported_date(records, season_label, as_of_date):
    """
    Most recent date at or before `as_of_date` (clamped to the
    season's end) that has a non-missing value, walking backward to
    skip gaps in reporting - this dataset has real gaps (the "M" / "T"
    entries), not just zeros.
    """

    start = season_start_date(season_label)
    end = min(as_of_date, season_end_date(season_label))

    if end < start:
        return None

    d = end

    while d >= start:

        if records.get(d) is not None:
            return d

        d -= timedelta(days=1)

    return None


def day_climatology(season_rows, day_label, current_season_label, current_depth):
    """
    Where `current_depth` ranks among all historical seasons' depth on
    this same day-of-season (1 = deepest on record for this date),
    plus the climatological normal (mean of every *other* season on
    record for this day-of-season, so the current season doesn't bias
    its own baseline).

    Returns (rank, rank_of, deepest_season, record_high_in,
    record_low_in, normal_depth_in), or all-None fields if there's
    nothing to compare against.
    """

    comparisons = [
        (season, day_map[day_label])
        for season, day_map in season_rows.items()
        if day_label in day_map
    ]

    if not comparisons or current_depth is None:
        return None, 0, None, None, None, None

    comparisons.sort(key=lambda pair: pair[1], reverse=True)

    deepest_season = comparisons[0][0]
    record_high_in = comparisons[0][1]
    record_low_in = comparisons[-1][1]

    rank = 1 + sum(1 for _, value in comparisons if value > current_depth)
    rank_of = len(comparisons)

    other_season_values = [
        value for season, value in comparisons if season != current_season_label
    ]
    normal_depth_in = (
        round(sum(other_season_values) / len(other_season_values), 1)
        if other_season_values
        else None
    )

    return rank, rank_of, deepest_season, record_high_in, record_low_in, normal_depth_in


# =====================================================================
# CURRENT DEPTH (HYDBTV - Daily Hydrometeorological Data Summary)
# =====================================================================

def extract_hyd_pre_text(html):
    """
    Pull raw text out of the <pre>...</pre> block(s) on a
    forecast.weather.gov product.php page. Concatenates every <pre>
    block rather than assuming there's exactly one, same tolerant
    approach as the RRSBTV parsing on the JS side of the dashboard,
    since these NWS product pages have shown that same quirk.
    """

    matches = re.findall(r"<pre[^>]*>([\s\S]*?)</pre>", html)

    if not matches:
        return None

    text = "\n".join(matches)

    return (
        text.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
    )


def fetch_hyd_product_text(version):
    """
    Fetch one HYDBTV issuance (version=1 is current, version=2/3 are
    progressively older reissues) and return its raw product text, or
    None if the page didn't contain a <pre> block at all.
    """

    params = dict(HYD_PARAMS, version=str(version))

    response = requests.get(HYD_URL, params=params, headers=HEADERS, timeout=30)
    response.raise_for_status()

    return extract_hyd_pre_text(response.text)


def hyd_column_bounds(header_line):
    """
    (start, end) character ranges for each HYD sub-column, derived
    from the product's own header line rather than hardcoded offsets -
    this is boilerplate NWS output, but trusting the actual page over
    an assumed fixed offset costs nothing and protects against a
    future formatting tweak silently breaking this.
    """

    positions = []

    for label in HYD_COLUMN_LABELS:
        idx = header_line.find(label)
        if idx == -1:
            return None
        positions.append((label, idx))

    bounds = {}

    for i, (label, start) in enumerate(positions):
        end = positions[i + 1][1] if i + 1 < len(positions) else None
        bounds[label] = (start, end)

    return bounds


def parse_mansfield_row(product_text):
    """
    Slice the Mount Mansfield row by the header's own column
    positions - Mansfield frequently omits Precip/Present Weather/New/
    SWE, and position-based slicing (instead of assuming a fixed count
    of whitespace-separated tokens) is the only reliable way to land
    on "Total" regardless of which other fields are blank that day.

    Returns a dict of field_name -> raw stripped string, or None if
    this issuance has no Mount Mansfield row at all (the without-
    Mansfield-data reissue case).
    """

    header_match = re.search(r"^\s*24 Hrs.*SWE\s*$", product_text, re.MULTILINE)

    if not header_match:
        return None

    bounds = hyd_column_bounds(header_match.group(0))

    if bounds is None:
        return None

    row_match = re.search(r"^Mount Mansfield.*$", product_text, re.MULTILINE)

    if not row_match:
        return None

    row = row_match.group(0)

    fields = {}

    for label, (start, end) in bounds.items():
        raw = row[start:end] if end is not None else row[start:]
        fields[HYD_COLUMN_FIELD_NAMES[label]] = raw.strip()

    return fields


def parse_snow_total_inches(fields):
    """
    Mansfield's "Total" field as a float, or None if it's blank
    (not reported) or "M" (missing/instrument outage).
    """

    if not fields:
        return None

    raw = fields.get("snow_total", "")

    if raw in ("", "M", "T"):
        return None

    try:
        return float(raw)
    except ValueError:
        return None


def fetch_current_mansfield_depth():
    """
    Mount Mansfield's current total snow depth straight from HYDBTV,
    rather than waiting on the local history file to catch up to it.
    Walks backward through the last few issuances (version=1, 2, 3)
    until one has a usable Mansfield Total - see the
    HYD_MAX_VERSION_FALLBACK comment above for why that's necessary.
    """

    for version in range(1, HYD_MAX_VERSION_FALLBACK + 1):

        try:
            product_text = fetch_hyd_product_text(version)
        except requests.RequestException as error:
            print(f"HYD version={version} fetch failed: {error}")
            continue

        if not product_text:
            continue

        depth = parse_snow_total_inches(parse_mansfield_row(product_text))

        if depth is not None:
            return {"depth_in": depth, "hyd_version": version}

    print(
        "HYD: no usable Mount Mansfield Total depth found in the last "
        f"{HYD_MAX_VERSION_FALLBACK} issuances."
    )

    return None


def build_snow_depth_observation(as_of=None):
    """
    Current depth (preferring live HYDBTV, falling back to the local
    mmnv1.json history's current-season value) + departure from normal
    and rank against 70+ years of history (both only computable when
    the local history has day-of-season data to compare against, i.e.
    during the tracked Sep-Jun season).

    HYDBTV is attempted unconditionally, regardless of season - it's
    a live station reading, not bound to the tracked season range, so
    "off-season" per the local history doesn't mean HYDBTV has nothing
    to say. Departure/normal/rank are the only pieces that gracefully
    degrade to None off-season, since those genuinely depend on a
    day-of-season comparison that doesn't exist for Jul/Aug.
    """

    as_of = as_of or datetime.now(timezone.utc).date()

    hyd_result = fetch_current_mansfield_depth()

    records = load_local_snow_depth_history()
    season_rows = organize_by_season(records)

    season_label = season_label_for_date(as_of)

    base_result = {
        "station": "Mount Mansfield Stake",
        "source": "mmnv1.json station history (IEM) + NWS Daily Hydromet Report (HYDBTV)",
        "season": season_label,
        "as_of_date": as_of.isoformat(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    # No local day-of-season data to compare against (Jul/Aug) - but
    # HYDBTV may still have a real live current-depth reading, so
    # surface that even though departure/normal/rank can't be
    # computed against a day-of-season range that doesn't exist.
    if season_label is None:

        if hyd_result is not None:

            return {
                **base_result,
                "observed_date_label": None,
                "current_depth_in": hyd_result["depth_in"],
                "current_depth_source": f"HYDBTV (version={hyd_result['hyd_version']})",
                "normal_depth_in": None,
                "record_high_in": None,
                "record_low_in": None,
                "departure_in": None,
                "departure_text": "Off-season \u2014 normal unavailable",
                "rank": None,
                "rank_of": None,
            }

        return {
            **base_result,
            "observed_date_label": None,
            "current_depth_in": None,
            "normal_depth_in": None,
            "record_high_in": None,
            "record_low_in": None,
            "departure_in": None,
            "departure_text": "Off-season \u2014 no current tracking",
            "rank": None,
            "rank_of": None,
        }

    latest_date = latest_reported_date(records, season_label, as_of)

    if latest_date is None and hyd_result is None:

        return {
            **base_result,
            "observed_date_label": None,
            "current_depth_in": None,
            "current_depth_source": None,
            "normal_depth_in": None,
            "record_high_in": None,
            "record_low_in": None,
            "departure_in": None,
            "departure_text": "No data reported yet this season",
            "rank": None,
            "rank_of": None,
        }

    # latest_date is still needed (even when HYD has the current
    # depth) to know which day-of-season to compare against for
    # normal/record/rank. Fall back to the most recent day the local
    # history has for this season if today itself has nothing but HYD
    # does.
    if latest_date is None:
        latest_date = latest_reported_date(records, season_label, season_end_date(season_label))

    obs_label = mmdd_label(latest_date) if latest_date else None

    # Prefer the live HYDBTV reading over the local history's
    # current-season value when we can get one - HYDBTV is always the
    # freshest source; the local file only refreshes whenever it's
    # last regenerated.
    if hyd_result is not None:
        current_depth = hyd_result["depth_in"]
        current_depth_source = f"HYDBTV (version={hyd_result['hyd_version']})"
    elif latest_date is not None:
        current_depth = records[latest_date]
        current_depth_source = "mmnv1.json station history (fallback, HYDBTV unavailable)"
    else:
        current_depth = None
        current_depth_source = None

    rank, rank_of, deepest_season, record_high_in, record_low_in, normal_depth = (
        day_climatology(season_rows, obs_label, season_label, current_depth)
        if obs_label is not None
        else (None, None, None, None, None, None)
    )

    if normal_depth is None or current_depth is None:

        departure = None
        departure_text = "Normal unavailable" if current_depth is not None else "No data reported yet this season"

    else:

        departure = current_depth - normal_depth

        if abs(departure) < 0.5:
            departure_text = "Near normal"
        elif departure > 0:
            departure_text = f"+{departure:.0f} in above normal"
        else:
            departure_text = f"{departure:.0f} in below normal"

    return {
        **base_result,
        "observed_date_label": obs_label,
        "current_depth_in": current_depth,
        "current_depth_source": current_depth_source,
        "normal_depth_in": normal_depth,
        "record_high_in": record_high_in,
        "record_low_in": record_low_in,
        "departure_in": departure,
        "departure_text": departure_text,
        "rank": rank,
        "rank_of": rank_of,
        "deepest_season_on_this_date": deepest_season,
    }


# =====================================================================
# MAIN
# =====================================================================

def main():

    os.makedirs(REPO_OUTPUT_DIR, exist_ok=True)

    # Chart: NWS feeds, unchanged.

    current_series = fetch_depth_series(CURRENT_URL)
    average_series = fetch_depth_series(AVERAGE_URL)
    max_series = fetch_depth_series(MAX_URL)
    min_series = fetch_depth_series(MIN_URL)

    plot_snow_depth_chart(current_series, average_series, max_series, min_series)

    # Current depth + departure: HYDBTV live reading preferred, local
    # mmnv1.json station history for normal/rank/fallback.

    status = build_snow_depth_observation()

    print(json.dumps(status, indent=2))

    with open(STATUS_OUTPUT_FILE, "w") as f:
        json.dump(status, f, indent=2)

    print(f"\nSaved status to {STATUS_OUTPUT_FILE}")


if __name__ == "__main__":
    main()
