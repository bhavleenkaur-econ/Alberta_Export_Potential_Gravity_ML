# Alberta export potential with gravity-informed ML

Structural gravity plus machine learning to rank Alberta's export **potential** by destination and HS4 product – separate from observed 2025 shipments.

**Skills demonstrated.** Bilateral HS4 trade (CEPII BACI), CEPII Gravity, WDI macros, WITS MFN tariffs, StatCan provincial mass, Balassa RCA, leave-one-out market size, Finger–Kreinin complementarity, HistGBM / XGBoost, corridor-grouped holdout and GroupKFold, calibrated potential scoring.

I do not commit CIMT mega-CSVs, CEPII zips, or fitted `.joblib` files. The repo keeps a gzipped training panel, slim StatCan slices, result tables, and figures.

## Key results (from my last `Q1b.py` run)

| Model | Holdout R² (log) | Holdout top-5 hit | Holdout Spearman | CV top-5 |
| --- | ---: | ---: | ---: | ---: |
| Model 1 Structural gravity | 0.076 | 0.000 | n/a | ~0.00 |
| Model 2 Product & tariff XGBoost | 0.511 | 0.255 | 0.708 | 0.168 |
| Model 3 Demand & capability XGBoost | 0.732 | 0.673 | 0.897 | 0.586 |

I score Alberta potential with **Model 3**. See `model_results/q1b_model_comparison.csv` and `q1b_cv5_mean_metrics.csv` for the full metric pack. Walkthrough notebook: [`Q1_Alberta_Trade_Model.ipynb`](Q1_Alberta_Trade_Model.ipynb).

## Quick start

```bash
cd <this-repo>
python3 -m pip install -r requirements.txt
python3 Q1a.py    # downloads CIMT if needed; writes outputs/q1a/
python3 Q1b.py    # fits M1–M3; writes model_results/, outputs/q1b/, models/
```

On macOS, if XGBoost fails to import, you can install OpenMP with `brew install libomp`.

Or open `Q1_Alberta_Trade_Model.ipynb` with `REFIT = False` to read the saved tables. You can set `REFIT = True` to call the scripts from the notebook.

## Questions answered

**Question 1(a).** Using appropriate data, identify Alberta's top ten export destination countries in 2025, ranked by total domestic export value. For each destination, report the top five HS4 products (code, name, and Alberta export value).

**Question 1(b).** Develop a machine-learning model using 2024 bilateral HS4 merchandise trade among the United States, Canada, Mexico, China, South Korea, Japan, and the United Kingdom, with macroeconomic, geographic, and other measurable factors. Treat Alberta as a separate economic entity. Report Alberta's predicted top ten export destinations and, for each, the predicted top five HS4 products. Distinguish observed 2025 Alberta exports (Q1a) from model-estimated export **potential** (Q1b).

Alberta potential is not the same as observed shipments. I calibrate dollar levels to Alberta's observed 2025 total for presentation; ranks and product mix come from the model.

## Three-model story (Q1b)

| ID | Display name | What it contains |
| --- | --- | --- |
| `M1_structural_gravity` | **Model 1 – Structural gravity (GDP & trade costs)** | HistGBM: exporter/importer GDP, distance, contiguity, language, FTA. No product structure. |
| `M2_product_and_tariff` | **Model 2 – Product structure & MFN tariffs (XGBoost)** | Model 1 + importer income/growth, HS2/energy/agri flags, RCA, importer MFN. |
| `M3_demand_and_capability` | **Model 3 – Market demand & exporter capability (XGBoost)** | Model 2 + LOO HS4 market size, demand growth, complementarity, complexity, capability. |

I keep Models 1–2 as the validity ladder (same pair holdout and GroupKFold).

For partners outside the seven-economy training set, Model 3 importer HS4 market size is proxied as world HS4 market × (importer GDP / US GDP). Details are in [`methodology/Q1b_methodology.md`](methodology/Q1b_methodology.md).

I trialled other estimators (Random Forest, CatBoost, stacking ensemble, openness and HHI variants). I summarise why I did not keep them in [`methodology/APPENDIX_other_models.md`](methodology/APPENDIX_other_models.md). They are not in `Q1b.py`.

## Folder map

```
.
  Q1_Alberta_Trade_Model.ipynb   # combined walkthrough (Q1a + Q1b)
  Q1a.py / Q1b.py
  README.md / DATA_TERMS.md / requirements.txt
  methodology/                   # methods + appendix of discarded trials
  data/q1a/                      # small name lists; CIMT CSVs download on demand
  data/q1b/                      # gzipped training panel + slim scoring inputs
  outputs/q1a/                   # observed top10 / top5 + chart
  outputs/q1b/                   # potential tables, residual & comparison plots
  model_results/                 # holdout + CV metrics, run meta
  models/                        # joblib files created when you run Q1b.py
```

You can download CIMT by running `Q1a.py`, and you can regenerate models by running `Q1b.py`.

## Reproducibility

- Target: `ln(1 + HS4 value USD)`.
- Split: `GroupShuffleSplit` 75/25 by exporter–importer corridor, `random_state=42`.
- CV: 5-fold `GroupKFold` by corridor.
- Scoring rule I used when choosing Model 3: highest holdout top-5 HS4 hit, then Spearman, then log R².
- Alberta mass: StatCan provincial GDP and population (2023); FTA/language/MFN use Canadian proxies.

## Data provenance

See [`DATA_TERMS.md`](DATA_TERMS.md) for links and citation notes. Summary:

| Source | Role |
| --- | --- |
| Statistics Canada CIMT domestic exports 2025 | Q1a observed values |
| CEPII BACI HS22 (seven-economy slice) | Q1b training panel (gzipped in repo) |
| CEPII Gravity | Distance, contig, language, FTA vintage |
| World Bank WDI | GDP, GDP/capita, growth |
| WITS / UNCTAD TRAINS | Importer MFN (HS6→HS4) |
| Public HS list | HS4 product names |

## Extending the work

You can rebuild features from full CEPII BACI / Gravity if you want a from-scratch panel instead of the gzipped slice. You can reuse the three-model ladder and Alberta scoring pattern for other provinces or years with a rebuilt panel and mass inputs.
