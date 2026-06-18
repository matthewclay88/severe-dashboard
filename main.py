import requests
import pandas as pd
import numpy as np
import metpy.calc as mpcalc

from metpy.units import units
from metpy.calc import (
    precipitable_water,
    mixed_layer_cape_cin,
    most_unstable_cape_cin
)

def get_site_parameters(site):

    results = {}

    # -------------------------
    # DOWNLOAD BUFKIT
    # -------------------------

    url = f"https://metfs1.agron.iastate.edu/data/bufkit/rap/rap_{site.lower()}.buf"

    response = requests.get(url, timeout=30)
response.raise_for_status()

text = response.text
    lines = text.splitlines()

    # -------------------------
    # PARSE SOUNDING
    # -------------------------

    data = []

    start_idx = None

    for i, line in enumerate(lines):

        if line.startswith("PRES TMPC"):
            start_idx = i + 2
            break

    if start_idx is None:
        raise Exception("Could not find sounding")

    for i in range(start_idx, len(lines) - 1, 2):

        try:

            line1 = lines[i].split()
            line2 = lines[i + 1].split()

            if len(line1) != 8:
                break

            if len(line2) != 2:
                break

            row = [float(x) for x in line1 + line2]

            data.append(row)

        except:
            break

    sounding = pd.DataFrame(
        data,
        columns=[
            "PRES",
            "TMPC",
            "TMWC",
            "DWPC",
            "THTE",
            "DRCT",
            "SKNT",
            "OMEG",
            "CFRL",
            "HGHT"
        ]
    )

    # Remove missing dewpoint rows
    sounding = sounding[
        sounding["DWPC"] > -9000
    ].copy()

    # -------------------------
    # METPY ARRAYS
    # -------------------------

    pressure = sounding["PRES"].values * units.hPa
    temperature = sounding["TMPC"].values * units.degC
    dewpoint = sounding["DWPC"].values * units.degC

    # -------------------------
    # PWAT
    # -------------------------

    pwat = precipitable_water(
        pressure,
        dewpoint
    )

    results["PWAT_mm"] = round(
        float(pwat.magnitude),
        1
    )

    # -------------------------
    # MUCAPE
    # -------------------------

    mucape, mucin = most_unstable_cape_cin(
        pressure,
        temperature,
        dewpoint
    )

    results["MUCAPE_Jkg"] = round(
        float(mucape.magnitude),
        1
    )

    # -------------------------
    # MLCAPE
    # -------------------------

    mlcape, mlcin = mixed_layer_cape_cin(
        pressure,
        temperature,
        dewpoint
    )

    results["MLCAPE_Jkg"] = round(
        float(mlcape.magnitude),
        1
    )

    results["MLCIN_Jkg"] = round(
        float(mlcin.magnitude),
        1
    )

    # -------------------------
    # SBCAPE
    # -------------------------

    parcel_prof = mpcalc.parcel_profile(
        pressure,
        temperature[0],
        dewpoint[0]
    )

    sbcape, sbcin = mpcalc.cape_cin(
        pressure,
        temperature,
        dewpoint,
        parcel_prof
    )

    results["SBCAPE_Jkg"] = round(
        float(sbcape.magnitude),
        1
    )

    results["SBCIN_Jkg"] = round(
        float(sbcin.magnitude),
        1
    )

    # -------------------------
    # DCAPE
    # -------------------------

    dcape = mpcalc.downdraft_cape(
        pressure,
        temperature,
        dewpoint
    )[0]

    results["DCAPE_Jkg"] = round(
        float(dcape.magnitude),
        1
    )

    # -------------------------
    # LCL
    # -------------------------

    lcl_pressure, lcl_temp = mpcalc.lcl(
        pressure[0],
        temperature[0],
        dewpoint[0]
    )

    lcl_idx = np.argmin(
        np.abs(
            sounding["PRES"] -
            lcl_pressure.magnitude
        )
    )

    results["LCL_m"] = round(
        float(
            sounding.iloc[lcl_idx]["HGHT"]
        ),
        0
    )

    # -------------------------
    # 0-3 KM LAPSE RATE
    # -------------------------

    sfc_temp = sounding.iloc[0]["TMPC"]
    sfc_hgt = sounding.iloc[0]["HGHT"]

    idx3 = np.argmin(
        np.abs(
            sounding["HGHT"] -
            (sfc_hgt + 3000)
        )
    )

    lr03 = (
        (sfc_temp - sounding.iloc[idx3]["TMPC"])
        /
        (
            (sounding.iloc[idx3]["HGHT"] - sfc_hgt)
            / 1000
        )
    )

    results["LR03_Ckm"] = round(
        float(lr03),
        2
    )

    # -------------------------
    # 700-500 LAPSE RATE
    # -------------------------

    idx700 = np.argmin(
        np.abs(
            sounding["PRES"] - 700
        )
    )

    idx500 = np.argmin(
        np.abs(
            sounding["PRES"] - 500
        )
    )

    lr75 = (
        (
            sounding.iloc[idx700]["TMPC"]
            -
            sounding.iloc[idx500]["TMPC"]
        )
        /
        (
            (
                sounding.iloc[idx500]["HGHT"]
                -
                sounding.iloc[idx700]["HGHT"]
            )
            / 1000
        )
    )

    results["LR75_Ckm"] = round(
        float(lr75),
        2
    )

    # -------------------------
    # WIND COMPONENTS
    # -------------------------

    u, v = mpcalc.wind_components(
        sounding["SKNT"].values * units.knots,
        sounding["DRCT"].values * units.degrees
    )

    heights = (
        sounding["HGHT"].values *
        units.meter
    )

    # -------------------------
    # 0-6 KM BULK SHEAR
    # -------------------------

    idx6 = np.argmin(
        np.abs(
            sounding["HGHT"] -
            (sfc_hgt + 6000)
        )
    )

    bs06 = np.sqrt(
        (u[idx6] - u[0])**2 +
        (v[idx6] - v[0])**2
    )

    results["BS06_kt"] = round(
        float(
            bs06.to("knots").magnitude
        ),
        1
    )

    # -------------------------
    # BUNKERS MOTION
    # -------------------------

    rm, lm, mw = mpcalc.bunkers_storm_motion(
        pressure,
        u,
        v,
        heights
    )

    # -------------------------
    # 0-1 KM SRH
    # -------------------------

    srh_pos, srh_neg, srh_total = (
        mpcalc.storm_relative_helicity(
            heights,
            u,
            v,
            depth=1000 * units.meter,
            storm_u=rm[0],
            storm_v=rm[1]
        )
    )

    results["SRH01_m2s2"] = round(
        float(srh_total.magnitude),
        1
    )

    results["SITE"] = site.upper()

    return results

if __name__ == "__main__":

    sites = [
        "kbtv",
        "kpbg",
        "kmss",
        "kslk",
        "rut",
        "kmpv",
        "1v4"
    ]

    all_results = []

    for site in sites:

        try:

            r = get_site_parameters(site)

            all_results.append(r)

            print(f"{site.upper()} OK")

        except Exception as e:

            print(f"{site.upper()} FAILED")
            print(e)

    df = pd.DataFrame(all_results)

    print(df)
