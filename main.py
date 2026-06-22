import requests
import pandas as pd
import numpy as np
import gspread
import json
import os
import netCDF4 as nc
from datetime import datetime, timedelta, timezone

from google.oauth2.service_account import Credentials
from metpy.units import units
import metpy.calc as mpcalc
from metpy.calc import (
    precipitable_water,
    mixed_layer_cape_cin,
    most_unstable_cape_cin,
)

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
# Airport ASOS/AWOS stations give the most complete daily precip records.
# Format: GHCND:USW000XXXXX  (look these up at ncei.noaa.gov/access/homr)
SITE_GHCND = {
    "kbtv": "GHCND:USW00014742",   # Burlington International Airport
    "kpbg": "GHCND:USW00094725",   # Plattsburgh International Airport
    "kmss": "GHCND:USW00014733",   # Massena Airport
    "kslk": "GHCND:USW00004745",   # Adirondack Regional (Saranac Lake)
    "rut":  "GHCND:USW00014745",   # Rutland State Airport
    "kmpv": "GHCND:USW00014742",   # Closest major ASOS (BTV); swap if MPV has own GHCND ID
    "1v4":  "GHCND:USW00014742",   # No dedicated ASOS; use BTV as proxy
    "kefk": "GHCND:USW00094746",   # Newport State Airport
}

# ============================================================
# NEW: NASA SPoRT-LIS SOIL MOISTURE
# ============================================================
# Data source: NASA Earthdata / GHRC DAAC OPeNDAP
#   https://www.earthdata.nasa.gov/data/catalog/ghrc-daac-sportlis-1
#
# The SPoRT-LIS CONUS 3-km product is available via OPeNDAP/HTTPS.
# Set EARTHDATA_TOKEN in your environment (bearer token from
#   https://urs.earthdata.nasa.gov — generate under "Generate Token").
#
# Variables pulled:
#   SoilMoist_0_10cm_inst  → Volumetric SM, 0–10 cm  (m³/m³)
#   SoilMoist_0_200cm_inst → Integrated relative SM,  0–200 cm (%)
#
# File naming convention (6-hourly):
#   LIS_HIST_YYYYMMDDHHOO.d01.nc
# We grab the most recent 00/06/12/18 UTC cycle.
# ============================================================

LIS_BASE = (
    "https://opendap.earthdata.nasa.gov/providers/GHRC_CLOUD/collections/"
    "SPoRT%20Land%20Information%20System%20%28SPoRT-LIS%29/granules/"
)

# Fallback: direct HTTPS file server (no OPeNDAP library needed, uses netCDF4)
LIS_HTTPS_BASE = (
    "https://ghrc.nsstc.nasa.gov/pub/fieldCampaigns/sportlis/data/"
)


def _lis_latest_url() -> str:
    """
    Build the URL for the most recent SPoRT-LIS file.
    Files are 6-hourly; rounds back to the last 00/06/12/18 UTC cycle
    and assumes ~4 h latency so we subtract one extra cycle when near the edge.
    """
    now_utc = datetime.now(timezone.utc)
    # Round down to last 6-hourly cycle with 4-h latency guard
    cycle_hour = ((now_utc.hour - 4) // 6) * 6
    if cycle_hour < 0:
        now_utc -= timedelta(days=1)
        cycle_hour = 18
    cycle_dt = now_utc.replace(hour=cycle_hour, minute=0, second=0, microsecond=0)
    yyyymmdd = cycle_dt.strftime("%Y%m%d")
    hh = cycle_dt.strftime("%H")
    # Public HTTPS granule path (no auth needed for the most recent files
    # at the SPoRT viewer mirror; Earthdata token used as fallback below)
    url = (
        f"https://ghrc.nsstc.nasa.gov/sportlis/data/lis_hist/"
        f"LIS_HIST_{yyyymmdd}{hh}00.d01.nc"
    )
    return url, cycle_dt


def fetch_lis_soil_moisture(site_coords: dict) -> dict:
    """
    Fetch NASA SPoRT-LIS volumetric (0-10 cm) and relative (0-200 cm)
    soil moisture at each site lat/lon.

    Returns a dict keyed by site (uppercase) with sub-keys:
        LIS_VSM_010CM   – volumetric soil moisture 0-10 cm (m³/m³)
        LIS_RSM_0200CM  – column-integrated relative SM 0-200 cm (%)
        LIS_VALID_UTC   – valid time string of the LIS file used

    Requires either:
      - EARTHDATA_TOKEN env var (bearer token from Earthdata URS), OR
      - Anonymous access if the mirror allows it.

    If the fetch fails, all values are set to None.
    """
    results = {site.upper(): {
        "LIS_VSM_010CM":  None,
        "LIS_RSM_0200CM": None,
        "LIS_VALID_UTC":  None,
    } for site in site_coords}

    token = os.environ.get("eyJ0eXAiOiJKV1QiLCJvcmlnaW4iOiJFYXJ0aGRhdGEgTG9naW4iLCJzaWciOiJlZGxqd3RwdWJrZXlfb3BzIiwiYWxnIjoiUlMyNTYifQ.eyJ0eXBlIjoiVXNlciIsInVpZCI6Im1hdHRoZXdjbGF5ODgiLCJleHAiOjE3ODczMjAyMTIsImlhdCI6MTc4MjEzNjIxMiwiaXNzIjoiaHR0cHM6Ly91cnMuZWFydGhkYXRhLm5hc2EuZ292IiwiaWRlbnRpdHlfcHJvdmlkZXIiOiJlZGxfb3BzIiwiYWNyIjoiZWRsIiwiYXNzdXJhbmNlX2xldmVsIjozfQ.HKuQP_e6rdlJVPfN5dP5H50aCiixvyjuoGdVd0XAVvGQZAYpX8dZfLSpIiQPVivXO-hHwOsSj1hLVOG9mMGInCfugQ-t44fsrNYyYhgHzi399ZyZkIN_jzrLIlA_71yoU0lgFdeypO8whVRIBC7tnohx8yHX-GgJED-uK-DOppgPUmTLMBmJnv1wiW6GNfBa_xFmB57STMjFFjgNVX6qWIQlZJOFeIGGo3evWRxtAwBCHbiwFEvqtQTdmBrGwwC-bkQA7OKeUAS5QDtQjWThaUD0_0slRwWWi2GfBpJcg4HSKyCwrqcwlkNuGxfn7oNb5XUs5XTcaFbqe_qL8bR-Hw", "")
    headers = {"Authorization": f"Bearer {token}"} if token else {}

    url, cycle_dt = _lis_latest_url()
    valid_str = cycle_dt.strftime("%Y-%m-%d %HZ")

    print(f"  Fetching SPoRT-LIS: {url}")
    try:
        resp = requests.get(url, headers=headers, timeout=60, stream=True)
        resp.raise_for_status()

        # Write to a temp file so netCDF4 can open it
        tmp_path = "/tmp/lis_latest.nc"
        with open(tmp_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                f.write(chunk)

        with nc.Dataset(tmp_path) as ds:
            lats = ds.variables["lat"][:]   # 1-D or 2-D depending on version
            lons = ds.variables["lon"][:]

            # Flatten to 1-D arrays for nearest-neighbour search
            if lats.ndim == 2:
                lat2d = lats
                lon2d = lons
            else:
                lon2d, lat2d = np.meshgrid(lons, lats)

            # Variable names can differ across LIS versions; try common aliases
            vsm_var = None
            rsm_var = None
            for vname in ds.variables:
                vl = vname.lower()
                if "soilmoist" in vl and ("0_10" in vl or "010" in vl or "0cm" in vl):
                    vsm_var = vname
                if ("soilmoist" in vl or "rsm" in vl) and (
                    "200" in vl or "0_200" in vl or "column" in vl or "int" in vl
                ):
                    rsm_var = vname

            for site, (slat, slon) in site_coords.items():
                dist = np.sqrt((lat2d - slat) ** 2 + (lon2d - slon) ** 2)
                row, col = np.unravel_index(np.argmin(dist), dist.shape)

                vsm_val = None
                rsm_val = None
                try:
                    if vsm_var:
                        v = ds.variables[vsm_var]
                        # Handle time dimension (pick first/only time step)
                        arr = v[:]
                        if arr.ndim == 4:
                            vsm_val = float(arr[0, 0, row, col])
                        elif arr.ndim == 3:
                            vsm_val = float(arr[0, row, col])
                        else:
                            vsm_val = float(arr[row, col])
                        # Mask fill values
                        if vsm_val > 9e+20 or vsm_val < 0:
                            vsm_val = None
                        else:
                            vsm_val = round(vsm_val, 4)
                except Exception:
                    pass

                try:
                    if rsm_var:
                        v = ds.variables[rsm_var]
                        arr = v[:]
                        if arr.ndim == 4:
                            rsm_val = float(arr[0, 0, row, col])
                        elif arr.ndim == 3:
                            rsm_val = float(arr[0, row, col])
                        else:
                            rsm_val = float(arr[row, col])
                        if rsm_val > 9e+20 or rsm_val < 0:
                            rsm_val = None
                        else:
                            rsm_val = round(rsm_val, 2)
                except Exception:
                    pass

                results[site.upper()] = {
                    "LIS_VSM_010CM":  vsm_val,
                    "LIS_RSM_0200CM": rsm_val,
                    "LIS_VALID_UTC":  valid_str,
                }

    except Exception as e:
        print(f"  WARNING: SPoRT-LIS fetch failed: {e}")

    return results


# ============================================================
# NEW: RFC FLASH FLOOD GUIDANCE
# ============================================================
# Source: NWS/WPC CONUS Gridded FFG ArcGIS MapServer
#   https://mapservices.weather.noaa.gov/raster/rest/services/
#          precip/rfc_gridded_ffg/MapServer
#
# Layers:
#   0  = FFG 01-hour
#   4  = FFG 03-hour
#   8  = FFG 06-hour
#   12 = FFG 12-hour  (Mid-Atlantic RFC only; may be null elsewhere)
#
# The "identify" operation returns the raster value at a lon/lat point.
# No authentication required.
# ============================================================

FFG_BASE = (
    "https://mapservices.weather.noaa.gov/raster/rest/services/"
    "precip/rfc_gridded_ffg/MapServer/identify"
)

# Layer IDs for each duration
FFG_LAYERS = {"01hr": 0, "03hr": 4, "06hr": 8, "12hr": 12}


def fetch_ffg(site_coords: dict) -> dict:
    """
    Query the NWS WPC RFC Gridded Flash Flood Guidance for each site.
    Returns dict keyed by UPPERCASE site with sub-keys:
        FFG_01HR_IN, FFG_03HR_IN, FFG_06HR_IN, FFG_12HR_IN  (inches)
    """
    results = {site.upper(): {
        "FFG_01HR_IN": None,
        "FFG_03HR_IN": None,
        "FFG_06HR_IN": None,
        "FFG_12HR_IN": None,
    } for site in site_coords}

    all_layer_ids = ",".join(str(v) for v in FFG_LAYERS.values())

    for site, (lat, lon) in site_coords.items():
        # The identify endpoint needs geometry in WGS84 (SR 4326)
        params = {
            "geometry":     f"{lon},{lat}",
            "geometryType": "esriGeometryPoint",
            "sr":           "4326",
            "layers":       f"all:{all_layer_ids}",
            "tolerance":    1,
            "mapExtent":    f"{lon-0.01},{lat-0.01},{lon+0.01},{lat+0.01}",
            "imageDisplay":  "100,100,96",
            "returnGeometry": "false",
            "f":             "json",
        }
        try:
            resp = requests.get(FFG_BASE, params=params, timeout=20)
            resp.raise_for_status()
            data = resp.json()

            layer_map = {str(v): k for k, v in FFG_LAYERS.items()}
            site_result = {}
            for result in data.get("results", []):
                lid = str(result.get("layerId", ""))
                dur = layer_map.get(lid)
                if dur is None:
                    continue
                # Value is returned in the 'attributes' dict as 'Pixel Value'
                pv = result.get("attributes", {}).get("Pixel Value", None)
                try:
                    val = float(pv)
                    # No-data sentinel varies; WPC uses -9999 or 0 for ocean
                    val = round(val, 2) if val > 0 else None
                except (TypeError, ValueError):
                    val = None
                site_result[f"FFG_{dur.upper()}_IN"] = val

            # Merge, keeping None for any missing duration
            results[site.upper()].update(site_result)
            print(f"  FFG {site.upper()}: {site_result}")

        except Exception as e:
            print(f"  WARNING: FFG fetch failed for {site.upper()}: {e}")

    return results


# ============================================================
# NEW: MULTI-DAY RAINFALL TOTALS (NCEI GHCND)
# ============================================================
# Source: NOAA NCEI Climate Data Online (CDO) API v2
#   https://www.ncei.noaa.gov/cdo-web/api/v2/data
#
# Dataset: GHCND (Global Historical Climatology Network - Daily)
# Datatype: PRCP  (daily precipitation, tenths of mm → convert to inches)
#
# SETUP:
#   export NCEI_CDO_TOKEN="your_token_here"
#   Request a free token at: https://www.ncdc.noaa.gov/cdo-web/token
#
# Returns 24 h, 72 h, and 7-day rainfall accumulations (inches).
# NOTE: GHCND reports daily totals with ~1-day latency; the 24 h total
# is yesterday's observation, 72 h is the sum of the last 3 full days,
# and 7 day is the sum of the last 7 full days.
# ============================================================

NCEI_BASE = "https://www.ncei.noaa.gov/cdo-web/api/v2/data"


def fetch_rainfall_totals(site_ghcnd: dict) -> dict:
    """
    Fetch 24 h, 72 h, and 7-day precipitation totals from NCEI GHCND.

    Requires NCEI_CDO_TOKEN environment variable.

    Returns dict keyed by UPPERCASE site with sub-keys:
        PRECIP_24HR_IN, PRECIP_72HR_IN, PRECIP_7DAY_IN
    """
    token = os.environ.get("fgpdRBwtpUHqwPjfilVKeknGQzAGVdSa", "")
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

    # Date range: last 7 full UTC days ending yesterday
    today = datetime.now(timezone.utc).date()
    end_date = today - timedelta(days=1)
    start_date = today - timedelta(days=7)

    # Deduplicate GHCND IDs → fetch once per unique station, then map back
    ghcnd_to_sites: dict[str, list[str]] = {}
    for site, ghcnd_id in site_ghcnd.items():
        ghcnd_to_sites.setdefault(ghcnd_id, []).append(site.upper())

    # Pre-fill results
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
            "units":      "metric",   # tenths of mm
            "limit":      10,
        }
        try:
            resp = requests.get(NCEI_BASE, headers=headers, params=params, timeout=20)
            resp.raise_for_status()
            data = resp.json()

            # Build a date→precip_mm dict (values are tenths of mm in metric mode)
            daily: dict[str, float] = {}
            for rec in data.get("results", []):
                date_str = rec["date"][:10]   # YYYY-MM-DD
                val_tenth_mm = float(rec.get("value", 0) or 0)
                val_in = round(val_tenth_mm / 10.0 / 25.4, 3)  # tenths mm → inches
                daily[date_str] = daily.get(date_str, 0) + val_in

            # Sort dates descending (most recent first)
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
# PARAMETER FUNCTION
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

    # ----------------------------------------------------------
    # EXISTING: Thermodynamic
    # ----------------------------------------------------------
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

    # ----------------------------------------------------------
    # EXISTING: LCL
    # ----------------------------------------------------------
    lcl_pressure, _ = mpcalc.lcl(pressure[0], temperature[0], dewpoint[0])
    lcl_idx = np.argmin(np.abs(sounding["PRES"] - lcl_pressure.magnitude))
    results["LCL_M"] = round(float(sounding.iloc[lcl_idx]["HGHT"]), 0)

    # ----------------------------------------------------------
    # EXISTING: Lapse rates
    # ----------------------------------------------------------
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

    # ----------------------------------------------------------
    # EXISTING: Wind / shear / SRH
    # ----------------------------------------------------------
    idx6 = np.argmin(np.abs(sounding["HGHT"] - (sfc_hgt + 6000)))
    bs06 = np.sqrt((u[idx6] - u[0])**2 + (v[idx6] - v[0])**2)
    results["BS06_KT"] = round(float(bs06.to("knots").magnitude), 1)

    rm, lm, mw = mpcalc.bunkers_storm_motion(pressure, u, v, heights)

    _, _, srh_total = mpcalc.storm_relative_helicity(
        heights, u, v,
        depth=1000 * units.meter,
        storm_u=rm[0],
        storm_v=rm[1],
    )
    results["SRH01_M2S2"] = round(float(srh_total.magnitude), 1)

    # ----------------------------------------------------------
    # EXISTING: Heavy Rain / Flash Flood Parameters
    # ----------------------------------------------------------
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

    # ----------------------------------------------------------
    # EXISTING: Snow Squall Parameters
    # ----------------------------------------------------------
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
        bs03 = np.sqrt((u[idx3km_pres] - u[0])**2 + (v[idx3km_pres] - v[0])**2)
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
# PRE-FETCH GRIDDED / STATION DATA  (once for all sites)
# ============================================================

print("Fetching NASA SPoRT-LIS soil moisture …")
lis_data = fetch_lis_soil_moisture(SITE_COORDS)

print("Fetching RFC Flash Flood Guidance …")
ffg_data = fetch_ffg(SITE_COORDS)

print("Fetching NCEI rainfall totals …")
precip_data = fetch_rainfall_totals(SITE_GHCND)

# ============================================================
# SOUNDING LOOP
# ============================================================

all_forecast_results = []
all_current_results  = []

for site in sites:
    print(f"Processing {site.upper()}")

    url = f"https://metfs1.agron.iastate.edu/data/bufkit/rap/rap_{site}.buf"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
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

    for hour in range(len(stim_locations) - 1):
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
            params["SITE"]       = site.upper()
            params["FHOUR"]      = hour
            params["VALID_TIME"] = valid_time

            # ---- Attach gridded / station data to EVERY forecast row ----
            site_key = site.upper()

            # NASA LIS soil moisture (current land state, same for all hours)
            lis = lis_data.get(site_key, {})
            params["LIS_VSM_010CM"]  = lis.get("LIS_VSM_010CM")
            params["LIS_RSM_0200CM"] = lis.get("LIS_RSM_0200CM")
            params["LIS_VALID_UTC"]  = lis.get("LIS_VALID_UTC")

            # RFC Flash Flood Guidance (current; same for all forecast hours)
            ffg = ffg_data.get(site_key, {})
            params["FFG_01HR_IN"] = ffg.get("FFG_01HR_IN")
            params["FFG_03HR_IN"] = ffg.get("FFG_03HR_IN")
            params["FFG_06HR_IN"] = ffg.get("FFG_06HR_IN")
            params["FFG_12HR_IN"] = ffg.get("FFG_12HR_IN")

            # Observed rainfall totals (same for all forecast hours)
            pr = precip_data.get(site_key, {})
            params["PRECIP_24HR_IN"] = pr.get("PRECIP_24HR_IN")
            params["PRECIP_72HR_IN"] = pr.get("PRECIP_72HR_IN")
            params["PRECIP_7DAY_IN"] = pr.get("PRECIP_7DAY_IN")

            all_forecast_results.append(params)

            if hour == 0:
                current = params.copy()
                current["RAP_RUN"] = rap_run
                all_current_results.append(current)

        except Exception as e:
            print(f"  {site.upper()} Hour {hour} failed: {e}")

# ============================================================
# BUILD DATAFRAMES
# ============================================================

forecast_df = pd.DataFrame(all_forecast_results)
forecast_df["DISPLAY_TIME"] = forecast_df["VALID_TIME"].str[-4:].str[:2] + "Z"

current_df = pd.DataFrame(all_current_results)

print(f"\nForecast rows : {len(forecast_df)}")
print(f"Current rows  : {len(current_df)}")

# ============================================================
# WRITE TO GOOGLE SHEETS
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
