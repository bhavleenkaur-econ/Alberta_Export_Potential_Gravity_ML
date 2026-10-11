"""
Question 1(b) – Gravity-informed ML of HS4 trade and Alberta export potential.

Question answered
-----------------
Develop a machine-learning model on 2024 bilateral HS4 trade among USA, Canada,
Mexico, China, Korea, Japan, and the UK. Treat Alberta as a separate exporter.
Report predicted top-ten destinations and top-five HS4 products. Distinguish
observed 2025 exports (Q1a) from model-estimated export potential.

Three models only (submission ladder)
-------------------------------------
  Model 1 – Structural gravity (GDP & trade costs)
      HistGBM: exporter/importer GDP, distance, contiguity, language, FTA.
  Model 2 – Product structure & MFN tariffs (XGBoost)
      Model 1 + importer income/growth, HS2/energy/agri, RCA, importer MFN.
  Model 3 – Market demand & exporter capability (XGBoost)
      Model 2 + LOO HS4 market size, demand growth, complementarity,
      complexity proxy, exporter capability.

I score Alberta potential with Model 3.
Models 1–2 are the validity ladder (pair holdout + GroupKFold).
Fitted .joblib files are written under models/ when you run this script.

Run from Final_assignment/:
    python3 Q1b.py
"""
from __future__ import annotations

import json
import math
import zipfile
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "q1b"
OUT = ROOT / "outputs" / "q1b"
RESULTS = ROOT / "model_results"
MODELS_DIR = ROOT / "models"
for p in (OUT, RESULTS, MODELS_DIR):
    p.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Feature sets (named for the submission story)
# ---------------------------------------------------------------------------
M1_FEATURES = ["ln_gdp_o", "ln_gdp_d", "ln_dist", "contig", "comlang_off", "fta"]
M2_FEATURES = M1_FEATURES + [
    "ln_gdppc_d",
    "growth_d",
    "hs2",
    "energy_hs",
    "agri_hs",
    "ln_rca",
    "ln_mfn",
]
M3_FEATURES = M2_FEATURES + [
    "imp_hs4_demand_growth",
    "ln_imp_hs4_mkt_loo",
    "ln_world_hs4_mkt_loo",
    "trade_complementarity",
    "ln_hs4_comp",
    "pci_proxy",
    "ln_exp_hs4_share",
    "ln_capability",
    "ln_exporter_div",
]

MODEL_SPECS = {
    "M1_structural_gravity": {
        "display": "Model 1 – Structural gravity (GDP & trade costs)",
        "estimator": "HistGBM",
        "features": M1_FEATURES,
        "short": "Structural gravity",
    },
    "M2_product_and_tariff": {
        "display": "Model 2 – Product structure & MFN tariffs (XGBoost)",
        "estimator": "XGBoost",
        "features": M2_FEATURES,
        "short": "Product & tariff XGBoost",
    },
    "M3_demand_and_capability": {
        "display": "Model 3 – Market demand & exporter capability (XGBoost)",
        "estimator": "XGBoost",
        "features": M3_FEATURES,
        "short": "Demand & capability XGBoost",
    },
}

SCORING_MODEL = "M3_demand_and_capability"

NAMES = {
    "USA": "United States of America",
    "CAN": "Canada",
    "MEX": "Mexico",
    "CHN": "China",
    "KOR": "Korea, South",
    "JPN": "Japan",
    "GBR": "United Kingdom",
    "NLD": "Netherlands",
    "IND": "India",
    "DEU": "Germany",
    "FRA": "France",
    "ITA": "Italy",
    "BEL": "Belgium",
    "HKG": "Hong Kong",
    "SGP": "Singapore",
    "CHE": "Switzerland",
    "ESP": "Spain",
    "AUS": "Australia",
    "BRA": "Brazil",
    "IDN": "Indonesia",
    "PER": "Peru",
    "VNM": "Viet Nam",
    "THA": "Thailand",
    "PHL": "Philippines",
    "ARE": "United Arab Emirates",
    "TUR": "Turkiye",
    "POL": "Poland",
    "IRL": "Ireland",
    "BGD": "Bangladesh",
    "COL": "Colombia",
    "CHL": "Chile",
    "SAU": "Saudi Arabia",
    "MYS": "Malaysia",
    "NOR": "Norway",
    "ZAF": "South Africa",
    "NGA": "Nigeria",
    "PAK": "Pakistan",
    "TWN": "Taiwan",
    "ARG": "Argentina",
}

CITY = {
    "AB": (51.05, -114.07),
    "USA": (38.90, -77.04),
    "CAN": (45.42, -75.70),
    "MEX": (19.43, -99.13),
    "CHN": (39.90, 116.40),
    "KOR": (37.57, 126.98),
    "JPN": (35.68, 139.65),
    "GBR": (51.51, -0.13),
    "NLD": (52.37, 4.90),
    "IND": (28.61, 77.21),
    "DEU": (52.52, 13.40),
    "FRA": (48.86, 2.35),
    "ITA": (41.90, 12.50),
    "BEL": (50.85, 4.35),
    "HKG": (22.32, 114.17),
    "SGP": (1.35, 103.82),
    "CHE": (47.38, 8.54),
    "ESP": (40.42, -3.70),
    "AUS": (-33.87, 151.21),
    "BRA": (-23.55, -46.63),
    "IDN": (-6.21, 106.85),
    "PER": (-12.05, -77.04),
    "VNM": (21.03, 105.85),
    "THA": (13.76, 100.50),
    "ARE": (25.20, 55.27),
    "TUR": (41.01, 28.98),
    "POL": (52.23, 21.01),
    "IRL": (53.35, -6.26),
    "BGD": (23.81, 90.41),
    "COL": (4.71, -74.07),
    "PHL": (14.60, 120.98),
}

FTA_2024 = {
    frozenset(("USA", "CAN")): 1,
    frozenset(("USA", "MEX")): 1,
    frozenset(("CAN", "MEX")): 1,
    frozenset(("USA", "KOR")): 1,
    frozenset(("CAN", "KOR")): 1,
    frozenset(("MEX", "KOR")): 1,
    frozenset(("CHN", "KOR")): 1,
    frozenset(("CAN", "JPN")): 1,
    frozenset(("MEX", "JPN")): 1,
    frozenset(("GBR", "JPN")): 1,
    frozenset(("GBR", "CAN")): 1,
    frozenset(("GBR", "MEX")): 1,
    frozenset(("GBR", "KOR")): 1,
    frozenset(("CHN", "JPN")): 1,
    frozenset(("JPN", "KOR")): 1,
}


def ln1p(s):
    if np.isscalar(s):
        return math.log1p(float(s))
    return np.log1p(np.asarray(s, dtype=float))


def fta_flag(o: str, d: str, gravity_fta) -> int:
    key = frozenset((o, d))
    if key in FTA_2024:
        return int(FTA_2024[key])
    if gravity_fta is None or (isinstance(gravity_fta, float) and math.isnan(gravity_fta)):
        return 0
    try:
        return int(float(gravity_fta) > 0)
    except (TypeError, ValueError):
        return 0


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def make_hgb():
    return HistGradientBoostingRegressor(
        max_depth=6, learning_rate=0.08, max_iter=300, min_samples_leaf=20, random_state=42
    )


def make_xgb():
    return xgb.XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.08,
        min_child_weight=20,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    )


def make_estimator(name: str):
    return make_hgb() if name.startswith("M1_") else make_xgb()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _smape(y, yhat) -> float:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    denom = np.abs(y) + np.abs(yhat)
    ok = denom > 0
    return float(np.mean(2.0 * np.abs(yhat[ok] - y[ok]) / denom[ok]) * 100.0)


def _mdape(y, yhat) -> float:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    ok = y > 0
    return float(np.median(np.abs(yhat[ok] - y[ok]) / y[ok]) * 100.0)


def _spearman_by_group(frame, y, yhat, group: str) -> float:
    tmp = frame[[group]].copy()
    tmp["y"] = np.asarray(y, float)
    tmp["yhat"] = np.asarray(yhat, float)
    rhos = []
    for _, g in tmp.groupby(group):
        if len(g) < 5 or g["y"].nunique() < 2 or g["yhat"].nunique() < 2:
            continue
        rhos.append(g["y"].corr(g["yhat"], method="spearman"))
    return float(np.nanmean(rhos)) if rhos else float("nan")


def _top5_hit(frame, y, yhat, group: str) -> float:
    tmp = frame[[group]].copy()
    tmp["y"] = np.asarray(y, float)
    tmp["yhat"] = np.asarray(yhat, float)
    tmp["hs4"] = frame["hs4"].astype(str).str.zfill(4).values
    hits = []
    for _, g in tmp.groupby(group):
        if len(g) < 5:
            continue
        true5 = set(g.nlargest(5, "y")["hs4"])
        pred5 = set(g.nlargest(5, "yhat")["hs4"])
        hits.append(len(true5 & pred5) / 5.0)
    return float(np.mean(hits)) if hits else float("nan")


def _pair_total_spearman(frame, y, yhat) -> float:
    tmp = frame[["pair"]].copy()
    tmp["y"] = np.asarray(y, float)
    tmp["yhat"] = np.asarray(yhat, float)
    agg = tmp.groupby("pair", as_index=False)[["y", "yhat"]].sum()
    if len(agg) < 3 or agg["y"].nunique() < 2:
        return float("nan")
    return float(agg["y"].corr(agg["yhat"], method="spearman"))


def metric_pack(frame, y_ln, pred_ln, y_usd) -> dict:
    pred_usd = np.expm1(pred_ln)
    y_ln = np.asarray(y_ln, float)
    pred_ln = np.asarray(pred_ln, float)
    return {
        "r2_ln": r2_score(y_ln, pred_ln),
        "mae_ln": mean_absolute_error(y_ln, pred_ln),
        "rmse_ln": math.sqrt(mean_squared_error(y_ln, pred_ln)),
        "mae_usd": mean_absolute_error(y_usd, pred_usd),
        "mdape": _mdape(y_usd, pred_usd),
        "smape": _smape(y_usd, pred_usd),
        "spearman_hs4": _spearman_by_group(frame, y_usd, pred_usd, "pair"),
        "top5_hit": _top5_hit(frame, y_usd, pred_usd, "pair"),
        "spearman_pair_total": _pair_total_spearman(frame, y_usd, pred_usd),
    }


# ---------------------------------------------------------------------------
# Fit / CV
# ---------------------------------------------------------------------------
def fit_ladder(df: pd.DataFrame):
    df = df.dropna(subset=["ln_gdp_o", "ln_gdp_d", "ln_dist", "contig", "comlang_off"]).copy()
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    tr_i, te_i = next(gss.split(df, groups=df["pair"]))
    train, test = df.iloc[tr_i].copy(), df.iloc[te_i].copy()

    rows = []
    fitted = {}
    for name, spec in MODEL_SPECS.items():
        cols = spec["features"]
        est = make_estimator(name)
        est.fit(train[cols], train["ln_value"])
        pred_tr = est.predict(train[cols])
        pred_te = est.predict(test[cols])
        row = {
            "model_id": name,
            "display_name": spec["display"],
            "estimator": spec["estimator"],
            "n_features": len(cols),
            "features": ", ".join(cols),
            "train_pairs": int(train["pair"].nunique()),
            "test_pairs": int(test["pair"].nunique()),
            "train_rows": len(train),
            "test_rows": len(test),
        }
        for k, v in metric_pack(train, train["ln_value"], pred_tr, train["value_usd"]).items():
            row[f"train_{k}"] = v
        for k, v in metric_pack(test, test["ln_value"], pred_te, test["value_usd"]).items():
            row[f"test_{k}"] = v
        rows.append(row)
        fitted[name] = (est, cols)
        test[f"pred_ln_{name}"] = pred_te
        joblib.dump(est, MODELS_DIR / f"{name}.joblib")
        print(
            f"  {spec['display']}: holdout R²={row['test_r2_ln']:.3f}  "
            f"top5={row['test_top5_hit']:.3f}  Spearman={row['test_spearman_hs4']:.3f}"
        )

    holdout = pd.DataFrame(rows).sort_values(
        ["test_top5_hit", "test_spearman_hs4", "test_r2_ln"], ascending=False
    )
    return holdout, fitted, train, test


def grouped_cv(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.dropna(subset=["ln_gdp_o", "ln_gdp_d", "ln_dist", "contig", "comlang_off"]).copy()
    gkf = GroupKFold(n_splits=5)
    fold_rows = []
    for fold, (tr_i, te_i) in enumerate(gkf.split(df, groups=df["pair"]), start=1):
        train, test = df.iloc[tr_i], df.iloc[te_i]
        for name, spec in MODEL_SPECS.items():
            cols = spec["features"]
            est = make_estimator(name)
            est.fit(train[cols], train["ln_value"])
            pred = est.predict(test[cols])
            pack = metric_pack(test, test["ln_value"], pred, test["value_usd"])
            fold_rows.append(
                {"fold": fold, "model_id": name, "display_name": spec["display"], **pack}
            )
        print(f"  CV fold {fold}/5")
    folds = pd.DataFrame(fold_rows)
    mean = folds.groupby(["model_id", "display_name"], as_index=False).mean(numeric_only=True)
    mean = mean.drop(columns=["fold"])
    mean["n_folds"] = 5
    return mean.sort_values(["top5_hit", "spearman_hs4", "r2_ln"], ascending=False), folds


def residual_plots(test: pd.DataFrame, model_id: str) -> None:
    col = f"pred_ln_{model_id}"
    if col not in test.columns:
        return
    y = test["ln_value"].to_numpy()
    yhat = test[col].to_numpy()
    resid = y - yhat
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(yhat, y, s=8, alpha=0.25, c="#1f4e79")
    lo, hi = min(y.min(), yhat.min()), max(y.max(), yhat.max())
    axes[0].plot([lo, hi], [lo, hi], color="black", lw=1)
    axes[0].set_xlabel("Predicted ln(1+value)")
    axes[0].set_ylabel("Actual ln(1+value)")
    axes[0].set_title(MODEL_SPECS[model_id]["display"])
    axes[1].scatter(yhat, resid, s=8, alpha=0.25, c="#9c2a2a")
    axes[1].axhline(0, color="black", lw=1)
    axes[1].set_xlabel("Predicted ln(1+value)")
    axes[1].set_ylabel("Log residual")
    fig.tight_layout()
    fig.savefig(OUT / "q1b_residual_diagnostics.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    show = pd.read_csv(RESULTS / "q1b_model_comparison.csv").sort_values("test_r2_ln")
    ax.barh(show["display_name"], show["test_r2_ln"], color="#1f4e79")
    ax.set_xlabel("Holdout R² of ln(1+value)")
    ax.set_title("Submission ladder – pair holdout")
    fig.tight_layout()
    fig.savefig(OUT / "q1b_model_comparison.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Alberta as separate exporter – scored with Model 3
# ---------------------------------------------------------------------------
def alberta_exporter_profile() -> dict:
    gdp = pd.read_csv(DATA / "statcan" / "36100222_alberta_canada.csv", low_memory=False)
    pop = pd.read_csv(DATA / "statcan" / "17100005_alberta.csv", low_memory=False)
    can_wdi = pd.read_csv(DATA / "wdi_macro_for_model.csv").query("iso3 == 'CAN'").iloc[0]

    def current_gdp_millions(geo: str, year: str) -> float:
        sub = gdp[
            (gdp["GEO"] == geo)
            & (gdp["Prices"] == "Current prices")
            & (gdp["Estimates"] == "Gross domestic product at market prices")
            & (gdp["REF_DATE"].astype(str) == year)
        ]
        return float(sub["VALUE"].iloc[0])

    ab_cad_mn = current_gdp_millions("Alberta", "2023")
    ca_cad_mn = current_gdp_millions("Canada", "2023")
    ab_gdp_usd = float(can_wdi["gdp_usd"]) * (ab_cad_mn / ca_cad_mn)
    pop_ab = pop[
        (pop["GEO"] == "Alberta")
        & (pop["Gender"] == "Total - gender")
        & (pop["Age group"] == "All ages")
        & (pop["REF_DATE"].astype(str) == "2023")
    ]
    ab_pop = float(pop_ab["VALUE"].iloc[0])
    return {
        "exporter": "AB",
        "gdp_usd_o": ab_gdp_usd,
        "gdppc_usd_o": ab_gdp_usd / ab_pop,
        "population_o": ab_pop,
        "gdp_growth_o": float(can_wdi["gdp_growth"]),
        "statcan_ab_gdp_cad_millions": ab_cad_mn,
        "statcan_ca_gdp_cad_millions": ca_cad_mn,
    }


def _wb_latest(zip_name: str, colname: str) -> pd.DataFrame:
    zp = DATA / zip_name
    if not zp.exists():
        return pd.DataFrame(columns=["iso3", colname])
    with zipfile.ZipFile(zp) as z:
        name = [n for n in z.namelist() if n.startswith("API_")][0]
        df = pd.read_csv(z.open(name), skiprows=4)
    df = df[["Country Code", "2023", "2024"]].rename(columns={"Country Code": "iso3"})
    df[colname] = df["2023"].fillna(df["2024"])
    return df[["iso3", colname]]


def load_hs4_names() -> pd.Series:
    hs = pd.read_csv(DATA / "harmonized-system.csv", dtype=str)
    four = hs[hs["level"].astype(str) == "4"][["hscode", "description"]].drop_duplicates("hscode")
    return four.set_index("hscode")["description"]


def build_alberta_score_frame(hs_ab: pd.DataFrame, top_n_partners: int = 35) -> tuple[pd.DataFrame, dict]:
    partners = pd.read_csv(DATA / "baci_2024_canada_partners.csv").head(top_n_partners)
    grav = pd.read_csv(DATA / "gravity_can_to_world.csv")
    for c in ["dist", "contig", "comlang_off", "fta_wto"]:
        grav[c] = pd.to_numeric(grav[c], errors="coerce")
    tariffs = pd.read_csv(DATA / "wits_mfn_hs4_2023.csv", dtype={"hs4": str})
    tariffs["hs4"] = tariffs["hs4"].astype(str).str.zfill(4)
    mfn_median = float(tariffs["mfn_hs4"].median())

    gdp_files = list((DATA / "wdi_gdp_raw").glob("API_NY.GDP.MKTP.CD*.csv"))
    gdp_full = pd.read_csv(gdp_files[0], skiprows=4)
    gdp_full = gdp_full[["Country Code", "2023", "2024"]].rename(columns={"Country Code": "iso3"})
    gdp_full["gdp_usd_d"] = gdp_full["2023"].fillna(gdp_full["2024"])
    gdppc = _wb_latest("wb_NY.GDP.PCAP.CD.zip", "gdppc_usd_d")
    growth = _wb_latest("wb_NY.GDP.MKTP.KD.ZG.zip", "growth_d")

    imp_mkt = pd.read_csv(DATA / "lookup_importer_hs4_market.csv", dtype={"hs4": str})
    imp_mkt["hs4"] = imp_mkt["hs4"].astype(str).str.zfill(4)
    world = pd.read_csv(DATA / "lookup_hs4_world_pci.csv", dtype={"hs4": str})
    world["hs4"] = world["hs4"].astype(str).str.zfill(4)
    comp = pd.read_csv(DATA / "lookup_can_importer_complementarity.csv")
    comp_med = float(comp["trade_complementarity"].median()) if len(comp) else 0.0
    world_med = float(world["world_hs4_mkt_usd"].median())
    pci_med = float(world["pci_proxy"].median())
    dem_med = float(imp_mkt["imp_hs4_demand_growth"].median())

    ab = alberta_exporter_profile()
    rows = []
    for _, p in partners.iterrows():
        imp = p["importer"]
        if imp == "CAN":
            continue
        g = grav.loc[grav["importer"] == imp]
        if g.empty:
            continue
        g = g.iloc[0]
        dist = haversine_km(*CITY["AB"], *CITY[imp]) if imp in CITY else (
            float(g["dist"]) if pd.notna(g["dist"]) else np.nan
        )
        contig = 1 if imp == "USA" else 0
        comlang = float(g["comlang_off"]) if pd.notna(g["comlang_off"]) else 0.0
        fta = fta_flag("CAN", imp, g["fta_wto"])
        md = gdp_full.loc[gdp_full["iso3"] == imp]
        if md.empty or pd.isna(md.iloc[0]["gdp_usd_d"]) or pd.isna(dist):
            continue
        gdp_d = float(md.iloc[0]["gdp_usd_d"])
        gdppc_row = gdppc.loc[gdppc.iso3 == imp, "gdppc_usd_d"]
        if gdppc_row.empty or pd.isna(gdppc_row.iloc[0]):
            continue
        gdppc_d = float(gdppc_row.iloc[0])
        gr_d = float(growth.loc[growth.iso3 == imp, "growth_d"].iloc[0]) if (growth.iso3 == imp).any() else 0.0
        tsub = tariffs.loc[tariffs["importer"] == imp] if imp in set(tariffs["importer"]) else pd.DataFrame()
        tc = float(comp.loc[comp["importer"] == imp, "trade_complementarity"].iloc[0]) if (
            comp["importer"] == imp
        ).any() else comp_med

        for _, h in hs_ab.iterrows():
            hs4 = str(h["hs4"]).zfill(4)
            if len(tsub):
                m = tsub.loc[tsub["hs4"] == hs4, "mfn_hs4"]
                mfn = float(m.iloc[0]) if len(m) else mfn_median
            else:
                mfn = mfn_median
            mk = imp_mkt.loc[(imp_mkt["importer"] == imp) & (imp_mkt["hs4"] == hs4)]
            wrow = world.loc[world["hs4"] == hs4]
            w_mkt = float(wrow.iloc[0]["world_hs4_mkt_usd"]) if len(wrow) else world_med
            pci = float(wrow.iloc[0]["pci_proxy"]) if len(wrow) else pci_med
            if len(mk):
                mkt = float(mk.iloc[0]["imp_hs4_mkt_usd"])
                dem = float(mk.iloc[0]["imp_hs4_demand_growth"])
            else:
                # Partners outside the seven-economy panel: scale world HS4 market by importer GDP
                # relative to US GDP so market-size features are not left at zero.
                us_gdp = float(gdp_full.loc[gdp_full["iso3"] == "USA", "gdp_usd_d"].iloc[0])
                mkt = w_mkt * (gdp_d / us_gdp) if us_gdp > 0 else w_mkt * 0.01
                dem = dem_med
            ab_share = float(h.get("ab_share", 0.0))
            n_exp = 7.0
            cap = ab_share / n_exp
            rows.append(
                {
                    "exporter": "AB",
                    "importer": imp,
                    "hs4": hs4,
                    "ln_gdp_o": math.log1p(ab["gdp_usd_o"]),
                    "ln_gdp_d": math.log1p(gdp_d),
                    "ln_gdppc_d": math.log1p(gdppc_d),
                    "ln_dist": math.log1p(dist),
                    "contig": contig,
                    "comlang_off": comlang,
                    "fta": fta,
                    "growth_d": 0.0 if pd.isna(gr_d) else gr_d,
                    "hs2": int(h["hs2"]),
                    "energy_hs": int(h["energy_hs"]),
                    "agri_hs": int(h["agri_hs"]),
                    "ln_rca": float(h["ln_rca"]),
                    "ln_mfn": math.log1p(max(mfn, 0.0)),
                    "imp_hs4_demand_growth": dem,
                    "ln_imp_hs4_mkt_loo": math.log1p(max(mkt, 0.0)),
                    "ln_world_hs4_mkt_loo": math.log1p(max(w_mkt, 0.0)),
                    "trade_complementarity": tc,
                    "ln_hs4_comp": math.log1p(min(ab_share, 0.05) * 1000.0),
                    "pci_proxy": pci,
                    "ln_exp_hs4_share": math.log1p(ab_share * 1000.0),
                    "ln_capability": math.log1p(cap * 1e6),
                    "ln_exporter_div": math.log1p(float(len(hs_ab))),
                }
            )
    return pd.DataFrame(rows), ab


def write_alberta_predictions(score: pd.DataFrame, hs_ab: pd.DataFrame, model_id: str):
    cad_per_usd = 1.35
    ab_obs_cad = float(hs_ab["ab_value"].sum())
    ab_obs_usd = ab_obs_cad / cad_per_usd
    raw_sum = float(score["pred_value_usd_raw"].sum())
    scale = (ab_obs_usd / raw_sum) if raw_sum > 0 else 1.0
    score = score.copy()
    score["pred_value_usd"] = score["pred_value_usd_raw"] * scale

    dest = (
        score.groupby("importer", as_index=False)
        .agg(pred_value_usd=("pred_value_usd", "sum"), pred_value_usd_raw=("pred_value_usd_raw", "sum"))
        .sort_values("pred_value_usd", ascending=False)
    )
    dest["rank"] = range(1, len(dest) + 1)
    dest["country_name"] = dest["importer"].map(NAMES).fillna(dest["importer"])
    dest["share_of_predicted_potential"] = dest["pred_value_usd"] / dest["pred_value_usd"].sum()
    dest["calibration_scale"] = scale
    dest["scoring_model"] = model_id
    top10 = dest.head(10).copy()
    top10.to_csv(OUT / "q1b_predicted_top10_destinations.csv", index=False)

    names = load_hs4_names()
    pieces = []
    for imp in top10["importer"]:
        sub = score[score["importer"] == imp].sort_values("pred_value_usd", ascending=False).head(5).copy()
        sub["country_name"] = NAMES.get(imp, imp)
        sub["hs4"] = sub["hs4"].astype(str).str.zfill(4)
        sub["hs4_name"] = sub["hs4"].map(names).fillna("(name not in HS list)")
        sub["rank_in_destination"] = range(1, len(sub) + 1)
        pieces.append(sub)
    top5 = pd.concat(pieces, ignore_index=True)
    top5[
        ["country_name", "importer", "rank_in_destination", "hs4", "hs4_name", "pred_value_usd"]
    ].to_csv(OUT / "q1b_predicted_top5_hs4_by_destination.csv", index=False)

    # Observed vs potential
    obs_path = ROOT / "outputs" / "q1a" / "q1a_top10_destinations.csv"
    if obs_path.exists():
        obs = pd.read_csv(obs_path)
        iso2_to_iso3 = {
            "US": "USA", "CN": "CHN", "JP": "JPN", "HK": "HKG", "KR": "KOR",
            "SG": "SGP", "MX": "MEX", "NL": "NLD", "PE": "PER", "ID": "IDN", "GB": "GBR",
        }
        obs["importer"] = obs["country_code"].map(iso2_to_iso3)
        compare = top10.merge(
            obs[["importer", "destination", "domestic_export_value_cad", "share_of_alberta"]],
            on="importer",
            how="outer",
        )
        compare.to_csv(OUT / "q1b_observed2025_vs_predicted_potential.csv", index=False)

    return top10, scale, ab_obs_cad, ab_obs_usd, raw_sum


def main():
    print("=== Load training panel (M1–M3 features) ===")
    panel_path = DATA / "training_panel_m1_m2_m3.csv.gz"
    if not panel_path.exists():
        panel_path = DATA / "training_panel_m1_m2_m3.csv"
    df = pd.read_csv(panel_path, dtype={"hs4": str})
    df["hs4"] = df["hs4"].astype(str).str.zfill(4)
    print(f"  {len(df):,} rows, {df['pair'].nunique()} corridors")

    print("=== Fit submission ladder ===")
    holdout, fitted, train, test = fit_ladder(df)
    holdout.to_csv(RESULTS / "q1b_model_comparison.csv", index=False)

    print("=== 5-fold GroupKFold ===")
    cv_mean, cv_folds = grouped_cv(df)
    cv_mean.to_csv(RESULTS / "q1b_cv5_mean_metrics.csv", index=False)
    cv_folds.to_csv(RESULTS / "q1b_cv5_fold_metrics.csv", index=False)
    print(cv_mean[["display_name", "r2_ln", "spearman_hs4", "top5_hit"]].to_string(index=False))

    residual_plots(test, SCORING_MODEL)

    print("=== Alberta scoring with", MODEL_SPECS[SCORING_MODEL]["display"], "===")
    hs_ab = pd.read_csv(DATA / "alberta_hs4_specialization.csv", dtype={"hs4": str})
    hs_ab["hs4"] = hs_ab["hs4"].astype(str).str.zfill(4)
    score, ab_profile = build_alberta_score_frame(hs_ab)
    est, cols = fitted[SCORING_MODEL]
    score["pred_ln_raw"] = est.predict(score[cols])
    score["pred_value_usd_raw"] = np.expm1(score["pred_ln_raw"])
    top10, scale, ab_cad, ab_usd, raw_sum = write_alberta_predictions(score, hs_ab, SCORING_MODEL)

    meta = {
        "scoring_model": SCORING_MODEL,
        "scoring_display": MODEL_SPECS[SCORING_MODEL]["display"],
        "scoring_rule": "Alberta tables use Model 3. Models 1–2 are the validity ladder.",
        "models": {k: {"display": v["display"], "n_features": len(v["features"]), "features": v["features"]} for k, v in MODEL_SPECS.items()},
        "train_test": "GroupShuffleSplit 75/25 by pair, random_state=42; 5-fold GroupKFold by pair.",
        "calibration": {
            "cad_per_usd": 1.35,
            "alberta_observed_total_cad": ab_cad,
            "alberta_observed_total_usd": ab_usd,
            "raw_predicted_sum_usd": raw_sum,
            "scale": scale,
            "note": "Calibration rescales mix to Alberta observed total. Ranks are the model.",
        },
        "alberta_profile": ab_profile,
    }
    (RESULTS / "q1b_run_meta.json").write_text(json.dumps(meta, indent=2, default=str))
    print(top10[["rank", "country_name", "pred_value_usd"]].to_string(index=False))
    print("Done →", OUT, RESULTS)


if __name__ == "__main__":
    main()
