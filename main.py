import re
import io
import requests
import pandas as pd
import numpy as np
import gspread
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from metpy.units import units
import metpy.calc as mpcalc
from metpy.calc import (
    precipitable_water,
    mixed_layer_cape_cin,
    most_unstable_cape_cin,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.dates as mdates
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pygrib
from PIL import Image

# ============================================================
# GOOGLE SHEETS AUTH
# ============================================================

creds_dict = json.loads(os.environ["GOOGLE_CREDENTIALS"])

scopes = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
gc = gspread.authorize(creds)
drive_service = build("drive", "v3", credentials=creds)

# ============================================================
# MODELS + FORECAST-HOUR CAP
# ============================================================
# IEM/mtarchive BUFKIT layout: .../data/bufkit/{model}/{model}_{site}.buf
# Confirmed available model folders: rap, hrrr, nam, nam4km, namm, gfs, gfsm
# ============================================================

MODELS = ["rap", "hrrr", "nam", "gfs"]
MAX_FORECAST_HOURS = 60  # cap per model; NAM/GFS otherwise run out to 84-384h

# ============================================================
# SITE METADATA
# lat/lon for each BUFKIT site — used for gridded data lookups.
# These are the approximate airport/station coordinates.
# ============================================================

SITE_COORDS = {
    "kbtv": (44.4719, -73.1503),   # Burlington, VT
    "kpbg": (44.6508, -73.4681),   # Plattsburgh, NY
    "kmss": (44.9353, -74.8456),   # Massena, NY
    "kslk": (44.3850, -74.2062),   # Saranac Lake, NY
    "rut":  (43.5294, -72.9497),   # Rutland, VT
    "kmpv": (44.2035, -72.5623),   # Montpelier, VT
    "1v4":  (44.8956, -72.8229),   # Hyde Park, VT  (approximate)
    "kefk": (44.8885, -72.0222),   # Newport, VT
}

# NCEI GHCND station IDs that correspond to each BUFKIT site.
SITE_GHCND = {
    "kbtv": "GHCND:USW00014742",
    "kpbg": "GHCND:USW00094725",
    "kmss": "GHCND:USW00014733",
    "kslk": "GHCND:USW00004745",
    "rut":  "GHCND:USW00014745",
    "kmpv": "GHCND:USW00014742",
    "1v4":  "GHCND:USW00014742",
    "kefk": "GHCND:USW00094746",
}

# ============================================================
# OPEN-METEO SOIL MOISTURE
# ============================================================

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"


def fetch_open_meteo_soil_moisture(site_coords: dict) -> dict:
    """
    Fetch Open-Meteo volumetric soil moisture at three depth layers
    for each site lat/lon. Uses the most recent available hourly value.
    Site-based only — not model-dependent, so fetched once per run.
    """
    results = {site.upper(): {
        "SM_SURFACE_PCT": None,
        "SM_ROOTZONE_PCT": None,
        "SM_VALID_UTC": None,
    } for site in site_coords}

    for site, (lat, lon) in site_coords.items():
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly":
                "soil_moisture_0_to_1cm,"
                "soil_moisture_1_to_3cm,"
                "soil_moisture_3_to_9cm,"
                "soil_moisture_9_to_27cm,"
                "soil_moisture_27_to_81cm",
            "timezone": "UTC",
            "forecast_days": 1,
        }
        try:
            resp = requests.get(OPEN_METEO_URL, params=params, timeout=20)
            resp.raise_for_status()
            data = resp.json()

            hourly = data.get("hourly", {})
            times  = hourly.get("time", [])
            sm_0_1   = hourly.get("soil_moisture_0_to_1cm", [])
            sm_1_3   = hourly.get("soil_moisture_1_to_3cm", [])
            sm_3_9   = hourly.get("soil_moisture_3_to_9cm", [])
            sm_9_27  = hourly.get("soil_moisture_9_to_27cm", [])
            sm_27_81 = hourly.get("soil_moisture_27_to_81cm", [])

            now_utc = datetime.now(timezone.utc)
            best_idx = None
            for i, t_str in enumerate(times):
                t = datetime.fromisoformat(t_str).replace(tzinfo=timezone.utc)
                if t <= now_utc:
                    best_idx = i

            if best_idx is not None:
                valid_str = times[best_idx] + "Z"

                def safe_float(arr, idx):
                    try:
                        v = arr[idx]
                        return round(float(v), 4) if v is not None else None
                    except Exception:
                        return None

                surface_sm = np.mean([
                    safe_float(sm_0_1, best_idx),
                    safe_float(sm_1_3, best_idx),
                    safe_float(sm_3_9, best_idx)
                ])

                rootzone_sm = np.mean([
                    safe_float(sm_9_27, best_idx),
                    safe_float(sm_27_81, best_idx)
                ])

                results[site.upper()] = {
                    "SM_SURFACE_PCT": round(surface_sm * 100, 1),
                    "SM_ROOTZONE_PCT": round(rootzone_sm * 100, 1),
                    "SM_VALID_UTC": valid_str,
                }
                print(
                    f"  Open-Meteo SM {site.upper()}: "
                    f"Surface={results[site.upper()]['SM_SURFACE_PCT']}% "
                    f"Root={results[site.upper()]['SM_ROOTZONE_PCT']}%"
                )
            else:
                print(f"  WARNING: No valid Open-Meteo SM time found for {site.upper()}")

        except Exception as e:
            print(f"  WARNING: Open-Meteo SM fetch failed for {site.upper()}: {e}")

    return results


# ============================================================
# RFC FLASH FLOOD GUIDANCE
# ============================================================

FFG_BASE = (
    "https://mapservices.weather.noaa.gov/raster/rest/services/"
    "precip/rfc_gridded_ffg/MapServer/identify"
)

FFG_LAYERS = {
    "01hr": 3,
    "03hr": 7,
    "06hr": 11,
    "12hr": 15
}


def fetch_ffg(site_coords: dict) -> dict:
    """
    Query the NWS WPC RFC Gridded Flash Flood Guidance for each site.
    Site-based only — not model-dependent, so fetched once per run.
    """
    results = {site.upper(): {
        "FFG_01HR_IN": None,
        "FFG_03HR_IN": None,
        "FFG_06HR_IN": None,
        "FFG_12HR_IN": None,
    } for site in site_coords}

    all_layer_ids = ",".join(str(v) for v in FFG_LAYERS.values())

    OFFSETS = [
        (0.00, 0.00),
        (0.05, 0.00),
        (-0.05, 0.00),
        (0.00, 0.05),
        (0.00, -0.05),
    ]

    for site, (lat, lon) in site_coords.items():
        site_result = {}

        for dlon, dlat in OFFSETS:
            test_lon = lon + dlon
            test_lat = lat + dlat

            params = {
                "geometry": f"{test_lon},{test_lat}",
                "geometryType": "esriGeometryPoint",
                "sr": "4326",
                "layers": f"all:{all_layer_ids}",
                "tolerance": 1,
                "mapExtent": (
                    f"{test_lon-0.01},{test_lat-0.01},"
                    f"{test_lon+0.01},{test_lat+0.01}"
                ),
                "imageDisplay": "100,100,96",
                "returnGeometry": "false",
                "f": "json",
            }

            try:
                resp = requests.get(FFG_BASE, params=params, timeout=20)
                resp.raise_for_status()
                data = resp.json()

                if len(data.get("results", [])) == 0:
                    continue

                layer_map = {str(v): k for k, v in FFG_LAYERS.items()}

                for result in data.get("results", []):
                    lid = str(result.get("layerId", ""))
                    dur = layer_map.get(lid)
                    if dur is None:
                        continue

                    pv = (
                        result.get("attributes", {})
                        .get("Service Pixel Value")
                    )

                    try:
                        val_mm = float(pv)
                        val_in = round(val_mm / 25.4, 2)
                    except Exception:
                        val_in = None

                    site_result[f"FFG_{dur.upper()}_IN"] = val_in

                if site_result:
                    print(f"{site.upper()} FOUND using offset {dlon},{dlat}")
                    break

            except Exception:
                pass

        results[site.upper()].update(site_result)
        print(f"FFG {site.upper()}: {results[site.upper()]}")

    return results


# ============================================================
# MULTI-DAY RAINFALL TOTALS (NCEI GHCND)
# ============================================================

NCEI_BASE = "https://www.ncei.noaa.gov/cdo-web/api/v2/data"


def fetch_rainfall_totals(site_ghcnd: dict) -> dict:
    """
    Fetch 24 h, 72 h, and 7-day precipitation totals from NCEI GHCND.
    Site-based only — not model-dependent, so fetched once per run.
    """
    token = os.environ.get("NCEI_CDO_TOKEN", "")
    if not token:
        print("  WARNING: NCEI_CDO_TOKEN not set; skipping rainfall totals.")
        return {
            site.upper(): {
                "PRECIP_24HR_IN": None,
                "PRECIP_72HR_IN": None,
                "PRECIP_7DAY_IN": None,
            }
            for site in site_ghcnd
        }

    headers = {"token": token}
    results = {}

    today      = datetime.now(timezone.utc).date()
    end_date   = today - timedelta(days=1)
    start_date = today - timedelta(days=7)

    ghcnd_to_sites: dict[str, list[str]] = {}
    for site, ghcnd_id in site_ghcnd.items():
        ghcnd_to_sites.setdefault(ghcnd_id, []).append(site.upper())

    for site in site_ghcnd:
        results[site.upper()] = {
            "PRECIP_24HR_IN": None,
            "PRECIP_72HR_IN": None,
            "PRECIP_7DAY_IN": None,
        }

    for ghcnd_id, sites in ghcnd_to_sites.items():
        params = {
            "datasetid":  "GHCND",
            "stationid":  ghcnd_id,
            "datatypeid": "PRCP",
            "startdate":  start_date.isoformat(),
            "enddate":    end_date.isoformat(),
            "units":      "metric",
            "limit":      10,
        }
        try:
            resp = requests.get(NCEI_BASE, headers=headers, params=params, timeout=20)
            resp.raise_for_status()
            data = resp.json()

            daily: dict[str, float] = {}
            for rec in data.get("results", []):
                date_str      = rec["date"][:10]
                val_tenth_mm  = float(rec.get("value", 0) or 0)
                val_in        = round(val_tenth_mm / 10.0 / 25.4, 3)
                daily[date_str] = daily.get(date_str, 0) + val_in

            sorted_dates = sorted(daily.keys(), reverse=True)

            p24 = daily.get(sorted_dates[0], 0.0) if len(sorted_dates) >= 1 else None
            p72 = sum(daily.get(d, 0.0) for d in sorted_dates[:3]) if len(sorted_dates) >= 1 else None
            p7d = sum(daily.get(d, 0.0) for d in sorted_dates[:7]) if len(sorted_dates) >= 1 else None

            record = {
                "PRECIP_24HR_IN": round(p24, 3) if p24 is not None else None,
                "PRECIP_72HR_IN": round(p72, 3) if p72 is not None else None,
                "PRECIP_7DAY_IN": round(p7d, 3) if p7d is not None else None,
            }
            print(f"  Precip {ghcnd_id}: {record}")
            for site in sites:
                results[site] = record

        except Exception as e:
            print(f"  WARNING: NCEI precip fetch failed for {ghcnd_id}: {e}")

    return results


# ============================================================
# GLWU WAVE HEIGHT + WIND BARBS (Lake Champlain 500m grid)
# ============================================================
# Not site-dependent and not part of the BUFKIT model loop — this is a
# standalone NOMADS pull + plot, run once per script execution. Wrapped
# so that a failure here never takes down the Sheets pipeline above.
# ============================================================

GLWU_BASE_URL = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/glwu/prod"
GLWU_GRID = "grlr_500m_lc"          # Lake Champlain 500m grid; swap for a
                                     # Great Lakes grid (e.g. grlc_2p5km_lc)
                                     # if that's what's actually needed
GLWU_OUTPUT_DIR = Path("./glwu_output")
GLWU_DOWNLOAD_DIR = Path("./glwu_downloads")
GLWU_BARB_SKIP_ROW = 8                # ~15% density reduction from the original
GLWU_BARB_SKIP_COL = 10               # single skip=8 (864 barbs -> 720, -16.7%);
                                       # an integer-only skip can't hit exactly 15%
                                       # (skip=9 both directions is -18.5%), this
                                       # split gets closest while keeping the change
                                       # small — see conversation notes for the math
GLWU_N_HOURS = 6                     # analysis hour + 5 forecast hours
GLWU_FRAME_DURATION_MS = 900          # time each frame is shown in the GIF

# GRIB2 stores these in meters and m/s (WMO convention); convert for display
# since staff work in feet/knots, not metric.
GLWU_M_TO_FT = 3.28084
GLWU_MS_TO_KT = 1.94384

# Fixed color scale instead of a per-run dynamic max — a dynamic scale means
# the same color could represent 0.5ft on a calm day and 3ft on a rough one,
# which is misleading at a glance from across the ops floor. Values above
# this just saturate at the top color (extend="max" in contourf below)
# rather than being hidden/blank.
GLWU_WAVE_HEIGHT_MAX_FT = 5.0

# Tighter map framing to kill the left/right whitespace: the raw grid spans
# the full ~2.0 degrees of latitude from South Bay up into Missisquoi
# Bay/Quebec, far more than the main operational lake body needs, and the
# narrow (~0.45 deg wide) lake ends up tiny in a tall, mostly-empty figure.
# Trimming north/south zooms in on the main body; the figure width is also
# computed FROM the resulting aspect ratio (see glwu_render_frame) instead
# of a fixed 8in — that's what actually eliminates the empty margins,
# clipping alone can't fully fix it on its own.
GLWU_LAT_CLIP_SOUTH_DEG = 0.15
GLWU_LAT_CLIP_NORTH_DEG = 0.35

BTV_LAT, BTV_LON = 44.4719, -73.1503   # Burlington, VT
PBG_LAT, PBG_LON = 44.6508, -73.4681   # Plattsburgh, NY

# NWS-specific seal, NOT the NOAA "meatball". weather.gov's own favicon.ico
# is actually the NOAA logo — the two are distinct, separately-trademarked
# marks. The correct NWS seal is the second of two badges in weather.gov's
# own page-header banner image; NWS_LOGO_CROP_BOX is the pixel-precise crop
# for it (found by scanning the banner for non-white column runs).
NWS_LOGO_URL = "https://www.weather.gov/bundles/templating/images/header/header.png"
NWS_LOGO_CROP_BOX = (56, 0, 104, 60)  # (left, top, right, bottom) in source pixels

# Simple top-down airplane silhouette for the KPBG marker — a hand-built
# vector path rather than a Unicode symbol (e.g. "✈") or a fetched image.
# Deliberate: this runs unattended on a GitHub Actions runner that may not
# share the same font/glyph coverage as wherever it's tested — a vector
# path renders identically everywhere, no font dependency at all.
AIRPLANE_PATH = mpath.Path([
    (0, 1.0), (0.08, 0.55), (0.5, 0.15), (0.5, 0.02), (0.1, 0.12),
    (0.1, -0.35), (0.28, -0.55), (0.28, -0.65), (0, -0.5),
    (-0.28, -0.65), (-0.28, -0.55), (-0.1, -0.35), (-0.1, 0.12),
    (-0.5, 0.02), (-0.5, 0.15), (-0.08, 0.55), (0, 1.0),
])

# Drive folder the plot gets uploaded to (the runner's disk is wiped after
# each GitHub Actions run, so this is what actually persists). The folder
# must be shared with the service account's email (found in
# creds_dict["client_email"]) with Editor access, or the upload will fail
# with a 403/404.
GLWU_DRIVE_FOLDER_ID = os.environ.get("GLWU_DRIVE_FOLDER_ID", "")

# ============================================================
# 8-STATION WAVE HEIGHT FORECAST (vertical stacked chart)
# ============================================================
# Forward forecast only, no observed history — the GRIB2 already gives us
# up to 48h forward from a single already-downloaded cycle, no need to
# accumulate data across runs the way a 10-day observed+forecast chart
# (like GLERL's own site) would require.
#
# Coordinates are the nearest-water-grid-point approximation of each named
# location; two (the NDBC/CDIP buoys) are exact station coordinates, the
# rest are well-known lake landmarks. All landed within ~1km of the actual
# grid point actually used EXCEPT Whitehall, NY (~18km off) — that narrow
# southernmost channel isn't well-resolved by this grid, consistent with
# it showing near-zero wave heights in GLERL's own chart too.
GLWU_STATIONS = [
    ("Burlington, VT", 44.476, -73.221),
    ("Rouses Point, NY", 44.994, -73.367),
    ("Port Henry, NY", 44.048, -73.458),
    ("Essex-Charlotte Ferry", 44.297, -73.320),
    ("Whitehall, NY", 43.554, -73.404),
    ("Philipsburg, QC", 45.033, -73.083),
    ("Inland Sea, Buoy 45166", 44.785, -73.258),
    ("Schuyler Reef, Buoy 251", 44.4877, -73.3391),
]
GLWU_STATION_FORECAST_MAX_HOUR = 48  # matches the short-cycle forecast length


def find_nearest_water_gridpoints(lats, lons, valid_mask, stations):
    """For each (name, lat, lon) in stations, find the closest grid index
    that actually has valid (non-masked/water) data — a naive nearest-point
    lookup could otherwise land on a masked land cell right next to the
    real target and silently return NaN forever."""
    indices = []
    for name, slat, slon in stations:
        dist2 = (lats - slat) ** 2 + (lons - slon) ** 2
        dist2_masked = np.where(valid_mask, dist2, np.inf)
        idx = np.unravel_index(np.argmin(dist2_masked), dist2_masked.shape)
        indices.append(idx)
    return indices


def glwu_render_station_forecast_panel(grib_path: Path):
    """Extract wave height forecast at GLWU_STATIONS from the given GRIB2
    (already downloaded for the main map — no extra fetch needed) and
    render as a 2-column x 4-row grid of time-series panels, one per
    station, each with its own x-axis time labels. Returns a PIL Image."""
    grbs = pygrib.open(str(grib_path))

    swh0 = grbs.select(shortName="swh", forecastTime=0)[0]
    wave0, lats, lons = swh0.data()
    lons = np.where(lons > 180, lons - 360, lons)
    valid_mask = ~np.ma.getmaskarray(wave0) if np.ma.is_masked(wave0) else np.ones_like(wave0, dtype=bool)

    station_idx = find_nearest_water_gridpoints(lats, lons, valid_mask, GLWU_STATIONS)

    all_fhours = sorted(set(m.forecastTime for m in grbs.select(shortName="swh")))
    fhours = [h for h in all_fhours if h <= GLWU_STATION_FORECAST_MAX_HOUR]

    times = []
    series = {name: [] for name, _, _ in GLWU_STATIONS}
    for h in fhours:
        msg = grbs.select(shortName="swh", forecastTime=h)[0]
        wave_h, _, _ = msg.data()
        times.append(msg.validDate)
        for (name, _, _), idx in zip(GLWU_STATIONS, station_idx):
            val = wave_h[idx]
            series[name].append(float(val) * GLWU_M_TO_FT if not np.ma.is_masked(val) else float("nan"))

    grbs.close()

    fig, axes = plt.subplots(4, 2, figsize=(11, 12), sharex=True)
    all_vals = [v for vals in series.values() for v in vals if not np.isnan(v)]
    ymax = max(2.5, np.ceil((max(all_vals) if all_vals else 1.0) * 1.15 * 2) / 2)

    for i, (name, _, _) in enumerate(GLWU_STATIONS):
        row, col = divmod(i, 2)
        ax = axes[row, col]
        vals = series[name]
        ax.plot(times, vals, color="#1a56c4", linewidth=1.3)
        ax.fill_between(times, vals, alpha=0.15, color="#1a56c4")
        ax.set_title(name, fontsize=10, loc="left", fontweight="bold")
        ax.set_ylim(0, ymax)
        ax.set_ylabel("Wave Height (ft)", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)
        # x-axis labels on EVERY panel, not just the bottom row — with
        # sharex=True, matplotlib hides tick labels on all but the bottom
        # subplot by default, which is what was actually missing before.
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %HZ"))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))
        ax.tick_params(axis="x", labelbottom=True, rotation=40)
        ax.set_xlabel("Time (UTC)", fontsize=8)

    fig.suptitle(
        f"Lake Champlain Wave Height Forecast\n"
        f"{times[0]:%Y-%m-%d %H:%M} UTC to +{GLWU_STATION_FORECAST_MAX_HOUR}h",
        fontsize=13, y=1.0,
    )
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def glwu_find_latest_cycle():
    """
    Check today's (and if needed yesterday's) GLWU directory for the
    newest posted cycle of GLWU_GRID. Cycles don't post at a fixed lag
    (some land ~1 min after the hour, others 27+ min late), so this
    scans the actual directory listing rather than assuming a file
    exists for the current hour.
    """
    now = datetime.now(timezone.utc)
    for day_offset in (0, 1):
        day = now - timedelta(days=day_offset)
        date_str = day.strftime("%Y%m%d")
        dir_url = f"{GLWU_BASE_URL}/glwu.{date_str}/"
        try:
            resp = requests.get(dir_url, timeout=30)
            resp.raise_for_status()
            html = resp.text
        except requests.RequestException:
            continue
        pattern = rf'glwu\.{re.escape(GLWU_GRID)}\.t(\d{{2}})z\.grib2(?!\.idx)"'
        hours = sorted(set(re.findall(pattern, html)))
        if hours:
            latest_hour = hours[-1]
            fname = f"glwu.{GLWU_GRID}.t{latest_hour}z.grib2"
            return date_str, latest_hour, f"{dir_url}{fname}"
    raise RuntimeError("Could not find a recent GLWU cycle for this grid.")


def glwu_download(url, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    return dest


_nws_icon_cache = None


def get_nws_icon():
    """Fetch the official NWS seal — cropped from weather.gov's own page
    header banner, not the favicon (that's the NOAA "meatball" instead) —
    once per script run and cache it in memory. Returns None (marker falls
    back to a plain dot) if the fetch ever fails, so a network hiccup here
    can't take down the whole plot."""
    global _nws_icon_cache
    if _nws_icon_cache is not None:
        return _nws_icon_cache
    try:
        resp = requests.get(NWS_LOGO_URL, timeout=15)
        resp.raise_for_status()
        banner = Image.open(io.BytesIO(resp.content))
        badge = banner.crop(NWS_LOGO_CROP_BOX)
        _nws_icon_cache = np.array(badge.convert("RGBA"))
        return _nws_icon_cache
    except Exception as e:
        print(f"  WARNING: could not fetch NWS icon for BTV marker: {e}")
        return None


def glwu_render_frame(swh, u, v, forecast_hour):
    """Render one frame (one forecast hour) as a PIL Image. Converts GRIB2's
    native meters/m-s to feet/knots for display. Uses a fixed color scale
    (GLWU_WAVE_HEIGHT_MAX_FT) rather than a per-run dynamic max, so frames
    are also directly comparable day to day, not just within one animation."""
    wave, lats, lons = swh.data()
    uu, _, _ = u.data()
    vv, _, _ = v.data()
    lons = np.where(lons > 180, lons - 360, lons)
    valid = swh.validDate

    wave_ft = wave * GLWU_M_TO_FT
    uu_kt = uu * GLWU_MS_TO_KT
    vv_kt = vv * GLWU_MS_TO_KT

    # Clip north/south to zoom in on the main lake body (GLWU_LAT_CLIP_*
    # above), then size the figure to actually match the resulting aspect
    # ratio — that's what fills the left/right whitespace, not the clip by
    # itself. The *1.8 leaves a little breathing room rather than a
    # razor-tight fit.
    lon_min, lon_max = lons.min() - 0.05, lons.max() + 0.05
    lat_min = lats.min() + GLWU_LAT_CLIP_SOUTH_DEG - 0.05
    lat_max = lats.max() - GLWU_LAT_CLIP_NORTH_DEG + 0.05
    extent = [lon_min, lon_max, lat_min, lat_max]
    lat_span = lat_max - lat_min
    lon_span = lon_max - lon_min
    fig_h = 10
    fig_w = max(3.5, fig_h * (lon_span / lat_span) * 1.8)

    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent(extent, crs=ccrs.PlateCarree())

    # Land goes down FIRST as background. Drawing it after the contour
    # (with a higher zorder) paints solid gray over the whole domain,
    # including the water - which is what was hiding the wave data.
    ax.add_feature(cfeature.LAND, facecolor="0.85", zorder=0)
    # cfeature has no built-in COUNTIES constant (unlike STATES/BORDERS) —
    # Natural Earth does have this layer, just accessed via
    # NaturalEarthFeature directly. Dotted + thin + gray so it reads as
    # reference detail without competing with state/border lines.
    counties = cfeature.NaturalEarthFeature("cultural", "admin_2_counties", "10m",
                                              facecolor="none", edgecolor="gray")
    ax.add_feature(counties, linewidth=0.4, linestyle=":", zorder=2)
    ax.add_feature(cfeature.STATES, edgecolor="black", linewidth=0.6, zorder=4)
    ax.add_feature(cfeature.BORDERS, edgecolor="black", linewidth=0.9, zorder=4)
    ax.add_feature(cfeature.COASTLINE, zorder=4, linewidth=0.5)

    wave_masked = np.ma.masked_invalid(wave_ft)
    levels = np.linspace(0, GLWU_WAVE_HEIGHT_MAX_FT, 21)
    cf = ax.contourf(lons, lats, wave_masked, levels=levels, cmap="turbo",
                      extend="max", transform=ccrs.PlateCarree(), zorder=3)
    cb = plt.colorbar(cf, ax=ax, orientation="vertical", pad=0.05, shrink=0.7)
    cb.set_label(f"Significant wave height (ft, fixed 0-{GLWU_WAVE_HEIGHT_MAX_FT:.0f}ft scale)")

    sr, sc = GLWU_BARB_SKIP_ROW, GLWU_BARB_SKIP_COL
    ax.barbs(lons[::sr, ::sc], lats[::sr, ::sc], uu_kt[::sr, ::sc], vv_kt[::sr, ::sc],
              length=5, linewidth=0.6, color="white",
              transform=ccrs.PlateCarree(), zorder=5)

    # BTV marker: NWS seal icon if the fetch succeeds, otherwise a plain
    # dot so the marker+label still show up rather than silently vanishing.
    icon = get_nws_icon()
    if icon is not None:
        imagebox = OffsetImage(icon, zoom=0.35)
        ab = AnnotationBbox(imagebox, (BTV_LON, BTV_LAT), frameon=False,
                             box_alignment=(0.5, 0.5), zorder=6)
        ax.add_artist(ab)
    else:
        ax.plot(BTV_LON, BTV_LAT, marker="o", color="black", markersize=5,
                 transform=ccrs.PlateCarree(), zorder=6)
    ax.text(BTV_LON + 0.025, BTV_LAT, "BTV", fontsize=9, fontweight="bold",
            color="black", transform=ccrs.PlateCarree(), zorder=6,
            va="center", ha="left")

    # PBG marker: vector airplane silhouette (see AIRPLANE_PATH above for
    # why this isn't a Unicode symbol or fetched image).
    ax.plot(PBG_LON, PBG_LAT, marker=AIRPLANE_PATH, markersize=16, color="black",
            transform=ccrs.PlateCarree(), zorder=6)
    # Left-aligned (extends west of the marker) rather than right — PBG sits
    # close enough to the shore that offsetting the label east/right (like
    # BTV's) landed the text over the water instead of on solid ground.
    ax.text(PBG_LON - 0.025, PBG_LAT, "PBG", fontsize=9, fontweight="bold",
            color="black", transform=ccrs.PlateCarree(), zorder=6,
            va="center", ha="right")

    label = "Analysis (current)" if forecast_hour == 0 else f"+{forecast_hour}h forecast"
    ax.set_title(f"GLWU ({GLWU_GRID}) wave height (ft) + wind barbs (kt)\n"
                 f"{label} — Valid {valid:%Y-%m-%d %H:%M} UTC")

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    frame_img = Image.open(buf).convert("RGB")

    # Sanity check: confirm the map itself actually rendered, not just the
    # colorbar — added after a run produced a GIF with a real colorbar but a
    # completely blank map area, with no exception anywhere in the log.
    # Restricted to the LEFT portion of the frame to exclude the colorbar,
    # which trivially contains lots of non-white color regardless of
    # whether the map itself drew anything.
    frame_arr = np.array(frame_img)
    map_region = frame_arr[:, :int(frame_img.width * 0.55), :]
    non_white_frac = float(np.mean(np.any(map_region < 240, axis=2)))
    if non_white_frac < 0.05:
        print(f"  WARNING: forecast hour {forecast_hour} frame's map area is "
              f"almost entirely blank ({non_white_frac:.1%} non-white) — the "
              f"map likely failed to render even though no exception was "
              f"raised. matplotlib={matplotlib.__version__}, "
              f"cartopy={cartopy.__version__}")

    return frame_img


def glwu_build_animation(grib_path: Path, out_gif: Path, n_hours: int = GLWU_N_HOURS):
    """
    Build an animated GIF covering the analysis hour plus the next
    (n_hours - 1) forecast hours, all from the single already-downloaded
    GRIB2 (the file already contains a forecast sequence out to 48h, so
    no extra download is needed to get future hours).
    """
    grbs = pygrib.open(str(grib_path))

    frame_msgs = []
    for h in range(n_hours):
        swh = grbs.select(shortName="swh", forecastTime=h)[0]
        u = grbs.select(shortName="u", forecastTime=h)[0]
        v = grbs.select(shortName="v", forecastTime=h)[0]
        frame_msgs.append((h, swh, u, v))

    pil_frames = [
        glwu_render_frame(swh, u, v, h) for h, swh, u, v in frame_msgs
    ]
    grbs.close()

    out_gif.parent.mkdir(parents=True, exist_ok=True)
    pil_frames[0].save(
        out_gif,
        save_all=True,
        append_images=pil_frames[1:],
        duration=GLWU_FRAME_DURATION_MS,
        loop=0,  # loop forever
    )


def glwu_upload_to_drive(local_path: Path, drive_filename: str, mimetype: str):
    """
    Upsert a file into Drive: if a file with this exact name already
    exists in GLWU_DRIVE_FOLDER_ID, overwrite its content (same file ID,
    same shareable link every time). Otherwise create it. Requires the
    target folder to be shared with the service account's client_email,
    AND (since service accounts have no storage quota of their own) a
    file with this exact name must already exist there, pre-uploaded by
    a real Google account - otherwise the "create" branch below will
    fail with a storageQuotaExceeded error.
    """
    if not GLWU_DRIVE_FOLDER_ID:
        print("  WARNING: GLWU_DRIVE_FOLDER_ID not set; skipping Drive upload.")
        return

    query = (
        f"name = '{drive_filename}' "
        f"and '{GLWU_DRIVE_FOLDER_ID}' in parents "
        f"and trashed = false"
    )
    existing = drive_service.files().list(
        q=query, spaces="drive", fields="files(id, name)"
    ).execute().get("files", [])

    media = MediaFileUpload(str(local_path), mimetype=mimetype, resumable=False)

    if existing:
        file_id = existing[0]["id"]
        drive_service.files().update(fileId=file_id, media_body=media).execute()
        print(f"  Updated existing Drive file: {drive_filename} (id={file_id})")
    else:
        file_metadata = {"name": drive_filename, "parents": [GLWU_DRIVE_FOLDER_ID]}
        created = drive_service.files().create(
            body=file_metadata, media_body=media, fields="id"
        ).execute()
        print(f"  Created new Drive file: {drive_filename} (id={created.get('id')})")


def run_glwu_plot():
    """Pull the latest GLWU cycle and render an animated GIF looping
    through the analysis hour plus the next GLWU_N_HOURS-1 forecast
    hours. Any failure here is caught so it can't break the rest of
    the pipeline (Sheets writes, etc.)."""
    try:
        print("\n=== GLWU wave height + wind barbs (animated loop) ===")
        print(f"  matplotlib={matplotlib.__version__}, cartopy={cartopy.__version__}, "
              f"pygrib={pygrib.__version__ if hasattr(pygrib, '__version__') else 'unknown'}")
        date_str, hour_str, url = glwu_find_latest_cycle()
        print(f"  Latest cycle found: {date_str} t{hour_str}z -> {url}")

        grib_dest = GLWU_DOWNLOAD_DIR / f"glwu.{GLWU_GRID}.{date_str}.t{hour_str}z.grib2"
        if not grib_dest.exists():
            glwu_download(url, grib_dest)
            print(f"  Downloaded to {grib_dest}")
        else:
            print("  Already have this cycle locally, skipping download.")

        timestamped_gif = GLWU_OUTPUT_DIR / f"glwu_{date_str}_t{hour_str}z.gif"
        latest_gif = GLWU_OUTPUT_DIR / "latest.gif"
        glwu_build_animation(grib_dest, timestamped_gif)
        glwu_build_animation(grib_dest, latest_gif)
        print(f"  Saved {timestamped_gif} and {latest_gif} "
              f"({GLWU_N_HOURS} frames: analysis + {GLWU_N_HOURS - 1}h forecast)")

        # The GitHub Actions runner disk disappears after this job ends,
        # so push the current animation to Drive to make it actually
        # persist. Only "latest.gif" is uploaded (overwritten each run)
        # so Drive doesn't accumulate one file per run, every 15
        # minutes, forever.
        glwu_upload_to_drive(latest_gif, "glwu_latest.gif", mimetype="image/gif")

        # 8-station vertical forecast chart — reuses the same already-downloaded
        # grib_dest, no extra fetch. Wrapped in its own try/except so a failure
        # here (e.g. a station coordinate landing somewhere unexpected) can't
        # take down the already-working map animation above.
        try:
            station_panel = glwu_render_station_forecast_panel(grib_dest)
            station_panel_path = GLWU_OUTPUT_DIR / "stations_latest.png"
            station_panel.save(station_panel_path)
            print(f"  Saved {station_panel_path} (8-station forecast, "
                  f"+{GLWU_STATION_FORECAST_MAX_HOUR}h forward)")
            glwu_upload_to_drive(station_panel_path, "glwu_stations_latest.png", mimetype="image/png")
        except Exception as e:
            print(f"  WARNING: 8-station forecast panel failed: {e}")

    except Exception as e:
        print(f"  WARNING: GLWU plot step failed: {e}")


# ============================================================
# PARAMETER FUNCTION (unchanged — BUFKIT format is generic
# across models, so this needs no per-model modification)
# ============================================================

def calculate_parameters(sounding):
    results = {}

    sounding = sounding[sounding["DWPC"] > -9000].copy().reset_index(drop=True)

    pressure    = sounding["PRES"].values * units.hPa
    temperature = sounding["TMPC"].values * units.degC
    dewpoint    = sounding["DWPC"].values * units.degC
    heights     = sounding["HGHT"].values * units.meter

    u, v = mpcalc.wind_components(
        sounding["SKNT"].values * units.knots,
        sounding["DRCT"].values * units.degrees,
    )

    sfc_temp = sounding.iloc[0]["TMPC"]
    sfc_hgt  = sounding.iloc[0]["HGHT"]
    sfc_dwpc = sounding.iloc[0]["DWPC"]

    pwat          = precipitable_water(pressure, dewpoint)
    mucape, mucin = most_unstable_cape_cin(pressure, temperature, dewpoint)
    mlcape, mlcin = mixed_layer_cape_cin(pressure, temperature, dewpoint)
    parcel_prof   = mpcalc.parcel_profile(pressure, temperature[0], dewpoint[0])
    sbcape, sbcin = mpcalc.cape_cin(pressure, temperature, dewpoint, parcel_prof)
    dcape         = mpcalc.downdraft_cape(pressure, temperature, dewpoint)[0]

    results["PWAT_MM"]    = round(float(pwat.magnitude), 1)
    results["MUCAPE_JKG"] = round(float(mucape.magnitude), 1)
    results["MLCAPE_JKG"] = round(float(mlcape.magnitude), 1)
    results["MLCIN_JKG"]  = round(float(mlcin.magnitude), 1)
    results["SBCAPE_JKG"] = round(float(sbcape.magnitude), 1)
    results["SBCIN_JKG"]  = round(float(sbcin.magnitude), 1)
    results["DCAPE_JKG"]  = round(float(dcape.magnitude), 1)

    lcl_pressure, _ = mpcalc.lcl(pressure[0], temperature[0], dewpoint[0])
    lcl_idx = np.argmin(np.abs(sounding["PRES"] - lcl_pressure.magnitude))
    results["LCL_M"] = round(float(sounding.iloc[lcl_idx]["HGHT"]), 0)

    idx3 = np.argmin(np.abs(sounding["HGHT"] - (sfc_hgt + 3000)))
    lr03 = (sfc_temp - sounding.iloc[idx3]["TMPC"]) / ((sounding.iloc[idx3]["HGHT"] - sfc_hgt) / 1000)
    results["LR03_CKM"] = round(float(lr03), 2)

    idx700 = np.argmin(np.abs(sounding["PRES"] - 700))
    idx500 = np.argmin(np.abs(sounding["PRES"] - 500))
    lr75 = (
        (sounding.iloc[idx700]["TMPC"] - sounding.iloc[idx500]["TMPC"])
        / ((sounding.iloc[idx500]["HGHT"] - sounding.iloc[idx700]["HGHT"]) / 1000)
    )
    results["LR75_CKM"] = round(float(lr75), 2)

    idx6 = np.argmin(np.abs(sounding["HGHT"] - (sfc_hgt + 6000)))
    bs06 = np.sqrt((u[idx6] - u[0])**2 + (v[idx6] - v[0])**2)
    results["BS06_KT"] = round(float(bs06.to("knots").magnitude), 1)

    rm, lm, mw = mpcalc.bunkers_storm_motion(pressure, u, v, heights)

    _, _, srh_total = mpcalc.storm_relative_helicity(
        heights, u, v, depth=1000 * units.meter,
        storm_u=rm[0], storm_v=rm[1],
    )
    results["SRH01_M2S2"] = round(float(srh_total.magnitude), 1)

    _, _, srh03 = mpcalc.storm_relative_helicity(
        heights, u, v, depth=3000 * units.meter,
        storm_u=rm[0], storm_v=rm[1],
    )
    results["SRH03_M2S2"] = round(float(srh03.magnitude), 1)

    scp = (
        (results["MUCAPE_JKG"] / 1000.0)
        * (results["SRH01_M2S2"] / 50.0)
        * (results["BS06_KT"] / 20.0)
    )
    results["SCP"] = round(scp, 2)

    results["SFC_DWPC"] = round(float(sfc_dwpc), 1)

    idx850 = np.argmin(np.abs(sounding["PRES"] - 850))
    results["DWPC_850"] = round(float(sounding.iloc[idx850]["DWPC"]), 1)

    freezing_level_m = None
    for i in range(len(sounding) - 1):
        t0 = sounding.iloc[i]["TMPC"]
        t1 = sounding.iloc[i + 1]["TMPC"]
        h0 = sounding.iloc[i]["HGHT"]
        h1 = sounding.iloc[i + 1]["HGHT"]
        if t0 >= 0 >= t1:
            frac = t0 / (t0 - t1)
            freezing_level_m = h0 + frac * (h1 - h0)
            break

    if freezing_level_m is not None:
        results["FRZ_LVL_M"] = round(freezing_level_m, 0)
        results["WCD_M"]     = round(freezing_level_m - sfc_hgt, 0)
    else:
        results["FRZ_LVL_M"] = None
        results["WCD_M"]     = None

    try:
        wb_temps = mpcalc.wet_bulb_temperature(pressure, temperature, dewpoint)
        wb_c = wb_temps.to("degC").magnitude
        wbz_m = None
        for i in range(len(sounding) - 1):
            wb0 = wb_c[i]; wb1 = wb_c[i + 1]
            h0  = sounding.iloc[i]["HGHT"]; h1 = sounding.iloc[i + 1]["HGHT"]
            if wb0 >= 0 >= wb1:
                frac  = wb0 / (wb0 - wb1)
                wbz_m = h0 + frac * (h1 - h0)
                break
        results["WBZ_M"] = round(wbz_m, 0) if wbz_m is not None else None
    except Exception:
        results["WBZ_M"] = None

    try:
        li = mpcalc.lifted_index(pressure, temperature, parcel_prof)
        results["LI"] = round(float(li.magnitude), 1)
    except Exception:
        results["LI"] = None

    try:
        rm_spd = np.sqrt(float(rm[0].magnitude)**2 + float(rm[1].magnitude)**2)
        results["RM_SPD_KT"] = round(float((rm_spd * units("m/s")).to("knots").magnitude), 1)
    except Exception:
        results["RM_SPD_KT"] = None

    try:
        idx3km = np.argmin(np.abs(sounding["HGHT"] - (sfc_hgt + 3000)))
        u_03 = float(np.mean(u[:idx3km + 1].magnitude))
        v_03 = float(np.mean(v[:idx3km + 1].magnitude))
        mean_03_spd = np.sqrt(u_03**2 + v_03**2) * units("m/s")
        results["MEAN_WIND_03KM_KT"] = round(float(mean_03_spd.to("knots").magnitude), 1)
    except Exception:
        results["MEAN_WIND_03KM_KT"] = None

    results["WIND_SPD_850_KT"]  = round(float(sounding.iloc[idx850]["SKNT"]), 1)
    results["WIND_DIR_850_DEG"] = round(float(sounding.iloc[idx850]["DRCT"]), 0)
    results["WIND_SPD_500_KT"]  = round(float(sounding.iloc[idx500]["SKNT"]), 1)
    results["WIND_DIR_500_DEG"] = round(float(sounding.iloc[idx500]["DRCT"]), 0)

    cape_pwat = float(mlcape.magnitude) * float(pwat.to("inches").magnitude)
    results["CAPE_PWAT"] = round(cape_pwat, 1)

    if results["WCD_M"] and results["WCD_M"] > 0:
        rrp = (float(mucape.magnitude) * float(pwat.to("mm").magnitude)) / results["WCD_M"]
        results["RRP"] = round(rrp, 3)
    else:
        results["RRP"] = None

    results["SFC_TMPC"]    = round(float(sfc_temp), 1)
    results["SFC_WIND_KT"] = round(float(sounding.iloc[0]["SKNT"]), 1)
    results["T700_TMPC"]   = round(float(sounding.iloc[idx700]["TMPC"]), 1)

    try:
        idx3km_pres = np.argmin(np.abs(sounding["HGHT"] - (sfc_hgt + 3000)))
        pres_03  = pressure[:idx3km_pres + 1]
        temp_03  = temperature[:idx3km_pres + 1]
        dew_03   = dewpoint[:idx3km_pres + 1]
        prof_03  = mpcalc.parcel_profile(pres_03, temp_03[0], dew_03[0])
        llcape, llcin = mpcalc.cape_cin(pres_03, temp_03, dew_03, prof_03)
        results["LLCAPE_JKG"] = round(float(llcape.magnitude), 1)
        results["LLCIN_JKG"]  = round(float(llcin.magnitude), 1)
    except Exception:
        results["LLCAPE_JKG"] = None
        results["LLCIN_JKG"]  = None

    try:
        dgz_top_m    = None
        dgz_bot_m    = None
        dgz_rh_vals  = []

        for i in range(len(sounding) - 1):
            t0   = sounding.iloc[i]["TMPC"]
            t1   = sounding.iloc[i + 1]["TMPC"]
            h0   = sounding.iloc[i]["HGHT"]
            h1   = sounding.iloc[i + 1]["HGHT"]
            td0  = sounding.iloc[i]["DWPC"]

            if t0 >= -12 >= t1 and dgz_top_m is None:
                frac       = (t0 - (-12)) / (t0 - t1)
                dgz_top_m  = h0 + frac * (h1 - h0)

            if t0 >= -18 >= t1 and dgz_bot_m is None:
                frac       = (t0 - (-18)) / (t0 - t1)
                dgz_bot_m  = h0 + frac * (h1 - h0)

            if -18 <= t0 <= -12:
                rh = mpcalc.relative_humidity_from_dewpoint(
                    t0 * units.degC, td0 * units.degC
                )
                dgz_rh_vals.append(float(rh.magnitude) * 100)

        if dgz_top_m is not None and dgz_bot_m is not None:
            results["DGZ_DEPTH_M"]  = round(dgz_bot_m - dgz_top_m, 0)
        else:
            results["DGZ_DEPTH_M"]  = None

        results["DGZ_MEAN_RH_PCT"] = round(float(np.mean(dgz_rh_vals)), 1) if dgz_rh_vals else None

    except Exception:
        results["DGZ_DEPTH_M"]     = None
        results["DGZ_MEAN_RH_PCT"] = None

    try:
        rh850 = mpcalc.relative_humidity_from_dewpoint(
            sounding.iloc[idx850]["TMPC"] * units.degC,
            sounding.iloc[idx850]["DWPC"] * units.degC,
        )
        results["RH850_PCT"] = round(float(rh850.magnitude) * 100, 1)
    except Exception:
        results["RH850_PCT"] = None

    try:
        idx1km = np.argmin(np.abs(sounding["HGHT"] - (sfc_hgt + 1000)))
        bs01   = np.sqrt((u[idx1km] - u[0])**2 + (v[idx1km] - v[0])**2)
        results["BS01_KT"] = round(float(bs01.to("knots").magnitude), 1)
    except Exception:
        results["BS01_KT"] = None

    try:
        bs03 = np.sqrt((u[idx3] - u[0])**2 + (v[idx3] - v[0])**2)
        results["BS03_KT"] = round(float(bs03.to("knots").magnitude), 1)
    except Exception:
        results["BS03_KT"] = None

    try:
        dir_sfc  = float(sounding.iloc[0]["DRCT"])
        dir_850  = float(sounding.iloc[idx850]["DRCT"])
        dir_diff = (dir_850 - dir_sfc + 360) % 360
        if dir_diff > 180:
            dir_diff -= 360
        results["DIR_SHR_SFC_850"] = round(dir_diff, 0)
    except Exception:
        results["DIR_SHR_SFC_850"] = None

    try:
        sbcape_val = float(sbcape.magnitude)
        sbcin_val  = abs(float(sbcin.magnitude))
        bs01_val   = results["BS01_KT"] if results["BS01_KT"] is not None else 0.0

        if sfc_temp <= -4.0:
            t_factor = 1.0
        elif sfc_temp < 0.0:
            t_factor = max(0.0, (0.0 - sfc_temp) / 4.0)
        else:
            t_factor = 0.0

        cin_term = max(0.0, (2000.0 - sbcin_val) / 2000.0)
        ssp = (sbcape_val / 100.0) * (bs01_val / 10.0) * cin_term * t_factor
        results["SSP"] = round(ssp, 2)

    except Exception:
        results["SSP"] = None

    return results


# ============================================================
# SITES
# ============================================================

sites = ["kbtv", "kpbg", "kmss", "kslk", "rut", "kmpv", "1v4", "kefk"]

# ============================================================
# GLWU WAVE + WIND PLOT  (standalone, not site/model-dependent)
# ============================================================

run_glwu_plot()

# ============================================================
# PRE-FETCH GRIDDED / STATION DATA  (once for all sites — not
# model-dependent, so no need to repeat per model)
# ============================================================

print("Fetching Open-Meteo soil moisture …")
sm_data = fetch_open_meteo_soil_moisture(SITE_COORDS)

print("Fetching RFC Flash Flood Guidance …")
ffg_data = fetch_ffg(SITE_COORDS)

print("Fetching NCEI rainfall totals …")
precip_data = fetch_rainfall_totals(SITE_GHCND)

# ============================================================
# SOUNDING LOOP  (now over MODELS x sites)
# ============================================================

all_forecast_results = []
all_current_results  = []

for model in MODELS:
    print(f"\n=== MODEL: {model.upper()} ===")

    for site in sites:
        print(f"Processing {model.upper()} {site.upper()}")

        url = f"https://metfs1.agron.iastate.edu/data/bufkit/{model}/{model}_{site}.buf"

        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
        except Exception as e:
            print(f"  WARNING: {model.upper()} {site.upper()} fetch failed: {e}")
            continue

        lines = response.text.splitlines()

        rap_run = None
        for line in lines[:50]:
            if "TIME =" in line:
                try:
                    rap_run = line.split("TIME =")[1].strip().split()[0]
                except Exception:
                    pass
                break

        stim_locations = [i for i, line in enumerate(lines) if line.startswith("STIM =")]
        print(f"  Found {len(stim_locations)} forecast hours")

        n_hours = len(stim_locations) - 1
        if n_hours > MAX_FORECAST_HOURS:
            n_hours = MAX_FORECAST_HOURS

        for hour in range(n_hours):
            start = stim_locations[hour]
            end   = stim_locations[hour + 1]
            block = lines[start:end]

            valid_time = None
            for line in block:
                if "TIME =" in line:
                    valid_time = line.split("TIME =")[1].strip().split()[0]
                    break

            start_idx = None
            for i, line in enumerate(block):
                if line.startswith("PRES TMPC"):
                    start_idx = i + 2
                    break

            if start_idx is None:
                continue

            data = []
            for i in range(start_idx, len(block) - 1, 2):
                try:
                    line1 = block[i].split()
                    line2 = block[i + 1].split()
                    if len(line1) != 8 or len(line2) != 2:
                        break
                    data.append([float(x) for x in line1 + line2])
                except Exception:
                    break

            if not data:
                continue

            sounding = pd.DataFrame(
                data,
                columns=["PRES", "TMPC", "TMWC", "DWPC", "THTE", "DRCT", "SKNT", "OMEG", "CFRL", "HGHT"],
            )

            try:
                params = calculate_parameters(sounding)
                params["MODEL"]      = model.upper()
                params["SITE"]       = site.upper()
                params["FHOUR"]      = hour
                params["VALID_TIME"] = valid_time

                site_key = site.upper()

                sm = sm_data.get(site_key, {})
                params["SM_SURFACE_PCT"]  = sm.get("SM_SURFACE_PCT")
                params["SM_ROOTZONE_PCT"] = sm.get("SM_ROOTZONE_PCT")
                params["SM_VALID_UTC"]    = sm.get("SM_VALID_UTC")

                ffg = ffg_data.get(site_key, {})
                params["FFG_01HR_IN"] = ffg.get("FFG_01HR_IN")
                params["FFG_03HR_IN"] = ffg.get("FFG_03HR_IN")
                params["FFG_06HR_IN"] = ffg.get("FFG_06HR_IN")
                params["FFG_12HR_IN"] = ffg.get("FFG_12HR_IN")

                pr = precip_data.get(site_key, {})
                params["PRECIP_24HR_IN"] = pr.get("PRECIP_24HR_IN")
                params["PRECIP_72HR_IN"] = pr.get("PRECIP_72HR_IN")
                params["PRECIP_7DAY_IN"] = pr.get("PRECIP_7DAY_IN")

                all_forecast_results.append(params)

                if hour == 0:
                    current = params.copy()
                    current["RUN"] = rap_run
                    all_current_results.append(current)

            except Exception as e:
                print(f"  {model.upper()} {site.upper()} Hour {hour} failed: {e}")

# ============================================================
# BUILD DATAFRAMES
# ============================================================

forecast_df = pd.DataFrame(all_forecast_results)
forecast_df["DISPLAY_TIME"] = forecast_df["VALID_TIME"].str[-4:].str[:2] + "Z"

# put MODEL/SITE/FHOUR first for readability
lead_cols = [c for c in ["MODEL", "SITE", "FHOUR", "VALID_TIME", "DISPLAY_TIME"] if c in forecast_df.columns]
other_cols = [c for c in forecast_df.columns if c not in lead_cols]
forecast_df = forecast_df[lead_cols + other_cols]

current_df = pd.DataFrame(all_current_results)
lead_cols_cur = [c for c in ["MODEL", "SITE", "RUN", "VALID_TIME"] if c in current_df.columns]
other_cols_cur = [c for c in current_df.columns if c not in lead_cols_cur]
current_df = current_df[lead_cols_cur + other_cols_cur]

forecast_df = forecast_df.replace({np.nan: None})
current_df = current_df.replace({np.nan: None})
print(f"\nForecast rows : {len(forecast_df)}")
print(f"Current rows  : {len(current_df)}")

# ============================================================
# WRITE TO GOOGLE SHEETS  (same two tabs, now MODEL-tagged)
# ============================================================

spreadsheet = gc.open_by_key("11FjM4i1s0SpOE5y5_nPDRzLEsoAPA62keyS06a0G3Fo")

forecast_sheet = spreadsheet.worksheet("Forecast")
forecast_sheet.clear()
forecast_sheet.update(
    [forecast_df.columns.tolist()] + forecast_df.values.tolist()
)
print("Forecast sheet updated successfully")

current_sheet = spreadsheet.worksheet("Current")
current_sheet.clear()
current_sheet.update(
    [current_df.columns.tolist()] + current_df.values.tolist()
)
print("Current sheet updated successfully")
