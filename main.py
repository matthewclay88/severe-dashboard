import requests
import pandas as pd
import numpy as np
import gspread
import json
import os

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

    # ==========================================================
    # NEW: Heavy Rain / Flash Flood Parameters
    # ==========================================================

    # ----------------------------------------------------------
    # MOISTURE
    # ----------------------------------------------------------

    # Surface dewpoint
    results["SFC_DWPC"] = round(float(sfc_dwpc), 1)

    # 850 mb dewpoint — low-level moisture depth
    idx850 = np.argmin(np.abs(sounding["PRES"] - 850))
    results["DWPC_850"] = round(float(sounding.iloc[idx850]["DWPC"]), 1)

    # ----------------------------------------------------------
    # FREEZING LEVEL & WARM CLOUD DEPTH
    # Warm cloud depth (WCD): surface to freezing level.
    # Deeper WCD = more efficient warm-rain coalescence = higher
    # rainfall rates. Critical for flash flood potential.
    # ----------------------------------------------------------
    freezing_level_m = None
    for i in range(len(sounding) - 1):
        t0 = sounding.iloc[i]["TMPC"]
        t1 = sounding.iloc[i + 1]["TMPC"]
        h0 = sounding.iloc[i]["HGHT"]
        h1 = sounding.iloc[i + 1]["HGHT"]
        if t0 >= 0 >= t1:
            # Linear interpolation to the 0°C level
            frac = t0 / (t0 - t1)
            freezing_level_m = h0 + frac * (h1 - h0)
            break

    if freezing_level_m is not None:
        results["FRZ_LVL_M"]  = round(freezing_level_m, 0)
        results["WCD_M"]      = round(freezing_level_m - sfc_hgt, 0)
    else:
        results["FRZ_LVL_M"]  = None
        results["WCD_M"]      = None

    # ----------------------------------------------------------
    # WET BULB ZERO (WBZ)
    # Lower WBZ = more melting zone = heavier surface rain rates.
    # Rule of thumb: WBZ < ~2500 m favors heavy rain over hail.
    # ----------------------------------------------------------
    try:
        wb_temps = mpcalc.wet_bulb_temperature(pressure, temperature, dewpoint)
        wb_c = wb_temps.to("degC").magnitude

        wbz_m = None
        for i in range(len(sounding) - 1):
            wb0 = wb_c[i]
            wb1 = wb_c[i + 1]
            h0  = sounding.iloc[i]["HGHT"]
            h1  = sounding.iloc[i + 1]["HGHT"]
            if wb0 >= 0 >= wb1:
                frac  = wb0 / (wb0 - wb1)
                wbz_m = h0 + frac * (h1 - h0)
                break

        results["WBZ_M"] = round(wbz_m, 0) if wbz_m is not None else None
    except Exception:
        results["WBZ_M"] = None

    # ----------------------------------------------------------
    # LIFTED INDEX (LI)
    # LI < 0 = unstable; < -4 = strongly unstable.
    # ----------------------------------------------------------
    try:
        li = mpcalc.lifted_index(pressure, temperature, parcel_prof)
        results["LI"] = round(float(li.magnitude), 1)
    except Exception:
        results["LI"] = None

    # ----------------------------------------------------------
    # STORM MOTION (Bunkers right-mover) & STORM-RELATIVE WINDS
    # Slow storm motion is the primary driver of flash flooding.
    # ----------------------------------------------------------
    try:
        rm_spd = np.sqrt(float(rm[0].magnitude)**2 + float(rm[1].magnitude)**2)
        results["RM_SPD_KT"] = round(
            float((rm_spd * units("m/s")).to("knots").magnitude), 1
        )
    except Exception:
        results["RM_SPD_KT"] = None

    # 0–3 km mean wind (steering-layer speed) — slow = training risk
    try:
        idx3km = np.argmin(np.abs(sounding["HGHT"] - (sfc_hgt + 3000)))
        u_03 = float(np.mean(u[:idx3km + 1].magnitude))
        v_03 = float(np.mean(v[:idx3km + 1].magnitude))
        mean_03_spd = np.sqrt(u_03**2 + v_03**2) * units("m/s")
        results["MEAN_WIND_03KM_KT"] = round(float(mean_03_spd.to("knots").magnitude), 1)
    except Exception:
        results["MEAN_WIND_03KM_KT"] = None

    # 850 mb wind speed & direction
    results["WIND_SPD_850_KT"]  = round(float(sounding.iloc[idx850]["SKNT"]), 1)
    results["WIND_DIR_850_DEG"] = round(float(sounding.iloc[idx850]["DRCT"]), 0)

    # 500 mb wind speed & direction
    results["WIND_SPD_500_KT"]  = round(float(sounding.iloc[idx500]["SKNT"]), 1)
    results["WIND_DIR_500_DEG"] = round(float(sounding.iloc[idx500]["DRCT"]), 0)

    # ----------------------------------------------------------
    # COMPOSITE INDICES
    # ----------------------------------------------------------

    # CAPE × PWAT product — simple heavy-rain proxy.
    # Higher values = greater rainfall rate potential.
    cape_pwat = float(mlcape.magnitude) * float(pwat.to("inches").magnitude)
    results["CAPE_PWAT"] = round(cape_pwat, 1)

    # Rainfall Rate Proxy (RRP): MUCAPE × PWAT / max(WCD, 1)
    # Incorporates warm cloud depth efficiency.
    if results["WCD_M"] and results["WCD_M"] > 0:
        rrp = (float(mucape.magnitude) * float(pwat.to("mm").magnitude)) / results["WCD_M"]
        results["RRP"] = round(rrp, 3)
    else:
        results["RRP"] = None

    return results

# ============================================================
# SITES
# ============================================================

sites = ["kbtv", "kpbg", "kmss", "kslk", "rut", "kmpv", "1v4", "kefk"]

all_forecast_results = []
all_current_results  = []

for site in sites:
    print(f"Processing {site.upper()}")

    url = f"https://metfs1.agron.iastate.edu/data/bufkit/rap/rap_{site}.buf"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    lines = response.text.splitlines()

    # RAP run time
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
