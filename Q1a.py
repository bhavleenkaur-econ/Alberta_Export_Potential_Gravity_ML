"""
Question 1(a) – Alberta's top export destinations and HS4 products, 2025.

Question answered
-----------------
Using appropriate data, identify Alberta's top ten export destination countries
in 2025 by domestic export value, and for each destination the top five HS4
products (code, name, value).

I keep province AB only, sum all 2025 months, roll US states to country, roll
HS6 to HS4, check HS2/HS6/HS8 Alberta totals match, and attach public HS4 names.

Large CIMT CSVs are not committed. You can run this script to download the
official zip when they are missing.

Run from Final_assignment/:
    python3 Q1a.py
"""
from __future__ import annotations

import re
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "q1a"
OUTPUT_DIR = ROOT / "outputs" / "q1a"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DOMESTIC_EXPORTS_URL = (
    "https://www150.statcan.gc.ca/n1/pub/71-607-x/2021004/zip/CIMT-CICM_Dom_Exp_2025.zip"
)
HS4_NAMES_URL = (
    "https://raw.githubusercontent.com/datasets/harmonized-system/master/data/harmonized-system.csv"
)

ZIP_PATH = DATA_DIR / "CIMT-CICM_Dom_Exp_2025.zip"
EXTRACT_DIR = DATA_DIR / "CIMT-CICM_Dom_Exp_2025"
HS4_NAMES_PATH = DATA_DIR / "harmonized-system.csv"

COLUMN_NAMES = {
    "YearMonth/AnnéeMois": "year_month",
    "HS6": "hs6",
    "HS2": "hs2",
    "HS8": "hs8",
    "Country/Pays": "country_code",
    "Province": "province_code",
    "State/État": "us_state",
    "Value/Valeur": "value_cad",
    "Quantity/Quantité": "quantity",
    "Unit of Measure/Unité de Mesure": "quantity_unit",
}


def download(url: str, destination: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=180) as response:
        destination.write_bytes(response.read())


def ensure_data() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not (EXTRACT_DIR / "ODPFN018_202512N.csv").exists():
        import zipfile

        print("Downloading domestic exports for 2025...")
        download(DOMESTIC_EXPORTS_URL, ZIP_PATH)
        with zipfile.ZipFile(ZIP_PATH) as archive:
            archive.extractall(DATA_DIR)
    if not HS4_NAMES_PATH.exists():
        print("Downloading HS4 product names...")
        download(HS4_NAMES_URL, HS4_NAMES_PATH)


def english_name(text: str) -> str:
    trailer = text.find("ODPF")
    if trailer != -1:
        text = text[:trailer]
    return re.split(r"\s{2,}", text.strip())[0]


def names_in_force(path: Path, code_pattern: str, code_column: str) -> pd.DataFrame:
    rows = []
    unread = []
    for line_number, line in enumerate(path.read_text(encoding="latin-1").splitlines(), start=1):
        match = re.match(code_pattern, line)
        if not match:
            unread.append(line_number)
            continue
        rows.append(
            {
                code_column: match.group("code"),
                "name_start": int(match.group("start")),
                "name_end": int(match.group("end")),
                "name": english_name(match.group("text")),
            }
        )
    if unread:
        raise ValueError(f"{path.name}: could not read lines {unread[:5]}")
    names = pd.DataFrame(rows)
    in_2025 = names[(names["name_start"] <= 202512) & (names["name_end"] >= 202501)]
    in_2025 = in_2025.sort_values("name_start").drop_duplicates(code_column, keep="last")
    return in_2025[[code_column, "name"]]


def alberta_total(filename: str) -> int:
    frame = pd.read_csv(
        EXTRACT_DIR / filename,
        usecols=["Province", "Value/Valeur"],
        dtype={"Province": str},
    )
    values = pd.to_numeric(frame["Value/Valeur"], errors="coerce")
    return int(values.loc[frame["Province"].eq("AB")].sum())


def main() -> None:
    ensure_data()

    COUNTRY_PATTERN = r"^(?P<code>\S{2})\s+\d+\s+(?P<start>\d{6})\s+(?P<end>\d{6})\s(?P<text>.*)$"
    PROVINCE_PATTERN = r"^\d+\s+(?P<code>\S{2})\s+(?P<start>\d{6})\s+(?P<end>\d{6})\s(?P<text>.*)$"
    provinces = names_in_force(EXTRACT_DIR / "ODPF_8_ProvDesc.TXT", PROVINCE_PATTERN, "province_code")
    countries = names_in_force(EXTRACT_DIR / "ODPF_6_CtyDesc.TXT", COUNTRY_PATTERN, "country_code")
    countries = countries.rename(columns={"name": "destination"})
    alberta_name = provinces.loc[provinces["province_code"].eq("AB"), "name"].item()
    print(f"Province code AB is {alberta_name}.")

    hs6 = pd.read_csv(
        EXTRACT_DIR / "ODPFN018_202512N.csv",
        dtype={
            "YearMonth/AnnéeMois": str,
            "HS6": str,
            "Country/Pays": str,
            "Province": str,
            "State/État": str,
            "Unit of Measure/Unité de Mesure": str,
        },
    )
    hs6 = hs6.rename(columns=COLUMN_NAMES)
    hs6["value_cad"] = pd.to_numeric(hs6["value_cad"], errors="coerce")
    hs6["hs6"] = hs6["hs6"].str.zfill(6)

    hs2_total = alberta_total("ODPFN020_202512N.csv")
    hs6_total = int(hs6.loc[hs6["province_code"].eq("AB"), "value_cad"].sum())
    hs8_total = alberta_total("ODPFN016_202512N.csv")
    print("Alberta domestic export value, 2025")
    print(f"  HS2: ${hs2_total:,.0f}  HS6: ${hs6_total:,.0f}  HS8: ${hs8_total:,.0f}")
    if not (hs2_total == hs6_total == hs8_total):
        raise AssertionError("Alberta totals differ by product level.")

    ab = hs6.loc[hs6["province_code"].eq("AB")].copy()
    months = sorted(ab["year_month"].unique())
    if months != [f"2025{month:02d}" for month in range(1, 13)]:
        raise AssertionError("The file does not contain all 12 months of 2025.")

    missing_destination = ab["country_code"].isna() | ab["country_code"].str.strip().eq("")
    key_columns = ["year_month", "hs6", "country_code", "us_state", "quantity_unit"]
    if int(ab.duplicated(key_columns).sum()):
        raise AssertionError("Duplicate Alberta rows would double-count dollars.")

    alberta_total_cad = int(ab["value_cad"].sum())
    exports = ab.loc[~missing_destination].copy()
    exports["hs4"] = exports["hs6"].str[:4]

    destination_totals = (
        exports.groupby("country_code", as_index=False)["value_cad"]
        .sum()
        .merge(countries, on="country_code", how="left")
        .sort_values(["value_cad", "country_code"], ascending=[False, True])
        .reset_index(drop=True)
    )
    destination_totals.insert(0, "destination_rank", destination_totals.index + 1)
    destination_totals["share_of_alberta"] = destination_totals["value_cad"] / alberta_total_cad
    top10 = destination_totals.loc[destination_totals["destination_rank"].le(10)].copy()

    hs_names = pd.read_csv(HS4_NAMES_PATH, dtype=str)
    hs4_names = (
        hs_names.loc[hs_names["level"].eq("4"), ["hscode", "description"]]
        .drop_duplicates("hscode")
        .rename(columns={"hscode": "hs4", "description": "product_name"})
    )
    product_totals = (
        exports.loc[exports["country_code"].isin(top10["country_code"])]
        .groupby(["country_code", "hs4"], as_index=False)["value_cad"]
        .sum()
        .sort_values(["country_code", "value_cad", "hs4"], ascending=[True, False, True])
    )
    product_totals["product_rank"] = product_totals.groupby("country_code").cumcount() + 1
    top_products = product_totals.loc[product_totals["product_rank"].le(5)].copy()
    top_products = top_products.merge(hs4_names, on="hs4", how="left")
    top_products = top_products.merge(
        top10[["destination_rank", "country_code", "destination", "value_cad", "share_of_alberta"]],
        on="country_code",
        how="left",
        suffixes=("", "_destination"),
    )
    top_products = top_products.rename(columns={"value_cad_destination": "destination_value_cad"})
    top_products["share_of_destination"] = top_products["value_cad"] / top_products["destination_value_cad"]
    top_products = top_products.sort_values(["destination_rank", "product_rank"])

    chart = top10.sort_values("value_cad", ascending=True)
    fig, axis = plt.subplots(figsize=(9, 5.5))
    bars = axis.barh(chart["destination"], chart["value_cad"] / 1e9, color="#1f4e79")
    axis.set_xlabel("Domestic export value, 2025 (billion Canadian dollars)")
    axis.set_title("Alberta's top 10 export destinations, 2025")
    for bar, value in zip(bars, chart["value_cad"]):
        billions = value / 1e9
        label = f"${billions:.1f}B" if billions >= 10 else f"${billions:.2f}B"
        axis.text(bar.get_width() + 1.2, bar.get_y() + bar.get_height() / 2, label, va="center", fontsize=9)
    axis.set_xlim(0, chart["value_cad"].max() / 1e9 * 1.16)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "q1a_top10_destinations.png", dpi=150)
    plt.close(fig)

    top10.rename(columns={"value_cad": "domestic_export_value_cad"})[
        ["destination_rank", "country_code", "destination", "domestic_export_value_cad", "share_of_alberta"]
    ].to_csv(OUTPUT_DIR / "q1a_top10_destinations.csv", index=False)

    top_products[
        [
            "destination_rank",
            "country_code",
            "destination",
            "destination_value_cad",
            "share_of_alberta",
            "product_rank",
            "hs4",
            "product_name",
            "value_cad",
            "share_of_destination",
        ]
    ].rename(
        columns={
            "destination_value_cad": "destination_domestic_export_value_cad",
            "share_of_alberta": "destination_share_of_alberta",
            "hs4": "hs4_code",
            "value_cad": "domestic_export_value_cad",
        }
    ).to_csv(OUTPUT_DIR / "q1a_top5_hs4_by_destination.csv", index=False)

    print(f"Alberta domestic exports, 2025: ${alberta_total_cad:,.0f}")
    print(f"Top 10 share: {top10['share_of_alberta'].sum():.1%}")
    print("Wrote", OUTPUT_DIR)


if __name__ == "__main__":
    main()
