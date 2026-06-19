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

    sounding = sounding[sounding["DWPC"] > -9000].copy()

    pressure    = sounding["PRES"].values * units.hPa
    temperature = sounding["TMPC"].values * units.degC
    dewpoint    = sounding["DWPC"].values * units.degC

    # Thermodynamic
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

    # LCL
    lcl_pressure, _ = mpcalc.lcl(pressure[0], temperature[0], dewpoint[0])
    lcl_idx = np.argmin(np.abs(sounding["PRES"] - lcl_pressure.magnitude))
    results["LCL_M"] = round(float(sounding.iloc[lcl_idx]["HGHT"]), 0)

    # Lapse rates
    sfc_temp = sounding.iloc[0]["TMPC"]
    sfc_hgt  = sounding.iloc[0]["HGHT"]

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

    # Wind / shear / SRH
    u, v = mpcalc.wind_components(
        sounding["SKNT"].values * units.knots,
        sounding["DRCT"].values * units.degrees,
    )
    heights = sounding["HGHT"].values * units.meter

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

    return results

# ============================================================
# SITES
# ============================================================

sites = ["kbtv", "kpbg", "kmss", "kslk", "rut", "kmpv", "1v4"]

all_forecast_results = []
all_current_results  = []

for site in sites:
    print(f"Processing {site.upper()}")

    url = f"https://metfs1.agron.iastate.edu/data/bufkit/rap/rap_{site}.buf"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    lines = response.text.splitlines()

    # RAP run time (from the very first TIME = line)
    rap_run = None
    for line in lines[:50]:
        if "TIME =" in line:
            try:
                rap_run = line.split("TIME =")[1].strip().split()[0]
            except Exception:
                pass
            break

    # Find all forecast hour locations
    stim_locations = [i for i, line in enumerate(lines) if line.startswith("STIM =")]
    print(f"  Found {len(stim_locations)} forecast hours")

    for hour in range(len(stim_locations) - 1):
        start = stim_locations[hour]
        end   = stim_locations[hour + 1]
        block = lines[start:end]

        # Valid time for this forecast hour
        valid_time = None
        for line in block:
            if "TIME =" in line:
                valid_time = line.split("TIME =")[1].strip().split()[0]
                break

        # Find sounding header
        start_idx = None
        for i, line in enumerate(block):
            if line.startswith("PRES TMPC"):
                start_idx = i + 2
                break

        if start_idx is None:
            continue

        # Build sounding dataframe
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

            # Every hour goes to the forecast sheet
            all_forecast_results.append(params)

            # Only hour 0 (current conditions) goes to the current sheet
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

# Forecast sheet
forecast_sheet = spreadsheet.worksheet("Forecast")
forecast_sheet.clear()
forecast_sheet.update(
    [forecast_df.columns.tolist()] + forecast_df.values.tolist()
)
print("Forecast sheet updated successfully")

# Current sheet
current_sheet = spreadsheet.worksheet("Current")
current_sheet.clear()
current_sheet.update(
    [current_df.columns.tolist()] + current_df.values.tolist()
)
print("Current sheet updated successfully")
