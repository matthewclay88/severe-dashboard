#!/usr/bin/env python3
"""
Pull the latest GLWU Lake Champlain (500m) grid from NOMADS and plot
significant wave height (fill) with 10m wind barbs on top.

Designed to be run hourly via cron/Task Scheduler. Each run:
  1. Finds the newest cycle actually posted (doesn't assume the "current
     hour" file exists yet - cycles post 0-60+ min after the top of the hour).
  2. Downloads that GRIB2 (whole file - it's only ~3-4 MB, no need to
     byte-range subset via the .idx).
  3. Reads HTSGW (sig. wave height) and UGRD/VGRD at the analysis time
     (forecastTime=0).
  4. Renders a PNG (both a timestamped copy and a rolling "latest.png").

Requires: pygrib, cartopy, matplotlib, requests
    pip install pygrib cartopy matplotlib requests --break-system-packages
"""

import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pygrib

# ---- config ----------------------------------------------------------------
BASE_URL = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/glwu/prod"
GRID = "grlr_500m_lc"          # change to e.g. "grlc_2p5km_lc" for the actual Great Lakes
OUTPUT_DIR = Path("./glwu_output")
DOWNLOAD_DIR = Path("./glwu_downloads")
BARB_SKIP = 8                   # subsample wind vectors so barbs aren't overplotted
# -----------------------------------------------------------------------------


def list_dir(url):
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.text


def find_latest_cycle():
    """Check today's (and if needed yesterday's) GLWU dir for the newest
    posted cycle of GRID, returning (date_str, hour_str, full_url)."""
    now = datetime.now(timezone.utc)
    for day_offset in (0, 1):
        day = now - timedelta(days=day_offset)
        date_str = day.strftime("%Y%m%d")
        dir_url = f"{BASE_URL}/glwu.{date_str}/"
        try:
            html = list_dir(dir_url)
        except requests.RequestException:
            continue
        pattern = rf'glwu\.{re.escape(GRID)}\.t(\d{{2}})z\.grib2(?!\.idx)"'
        hours = sorted(set(re.findall(pattern, html)))
        if hours:
            latest_hour = hours[-1]
            fname = f"glwu.{GRID}.t{latest_hour}z.grib2"
            return date_str, latest_hour, f"{dir_url}{fname}"
    raise RuntimeError("Could not find a recent GLWU cycle for this grid.")


def download(url, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    dest.write_bytes(r.content)
    return dest


def plot_wave_and_wind(grib_path: Path, out_png: Path):
    grbs = pygrib.open(str(grib_path))
    swh = grbs.select(shortName="swh", forecastTime=0)[0]
    u = grbs.select(shortName="u", forecastTime=0)[0]
    v = grbs.select(shortName="v", forecastTime=0)[0]

    wave, lats, lons = swh.data()
    uu, _, _ = u.data()
    vv, _, _ = v.data()
    lons = np.where(lons > 180, lons - 360, lons)
    valid = swh.validDate
    grbs.close()

    fig = plt.figure(figsize=(8, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    extent = [lons.min() - 0.05, lons.max() + 0.05,
              lats.min() - 0.05, lats.max() + 0.05]
    ax.set_extent(extent, crs=ccrs.PlateCarree())

    wave_masked = np.ma.masked_invalid(wave)
    cf = ax.contourf(lons, lats, wave_masked, levels=20, cmap="turbo",
                      transform=ccrs.PlateCarree())
    cb = plt.colorbar(cf, ax=ax, orientation="vertical", pad=0.05, shrink=0.7)
    cb.set_label("Significant wave height (m)")

    s = BARB_SKIP
    ax.barbs(lons[::s, ::s], lats[::s, ::s], uu[::s, ::s], vv[::s, ::s],
              length=5, linewidth=0.6, transform=ccrs.PlateCarree())

    ax.add_feature(cfeature.LAND, facecolor="0.85", zorder=2)
    ax.add_feature(cfeature.COASTLINE, zorder=3)
    ax.set_title(f"GLWU ({GRID}) wave height + wind barbs\n"
                 f"Valid {valid:%Y-%m-%d %H:%M} UTC")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    date_str, hour_str, url = find_latest_cycle()
    print(f"Latest cycle found: {date_str} t{hour_str}z -> {url}")

    grib_dest = DOWNLOAD_DIR / f"glwu.{GRID}.{date_str}.t{hour_str}z.grib2"
    if not grib_dest.exists():
        download(url, grib_dest)
        print(f"Downloaded to {grib_dest}")
    else:
        print("Already have this cycle locally, skipping download.")

    timestamped = OUTPUT_DIR / f"glwu_{date_str}_t{hour_str}z.png"
    latest = OUTPUT_DIR / "latest.png"
    plot_wave_and_wind(grib_dest, timestamped)
    plot_wave_and_wind(grib_dest, latest)
    print(f"Saved {timestamped} and {latest}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
