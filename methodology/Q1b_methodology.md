# Q1b methodology – three submission models

## Question this answers

Develop a machine-learning model using 2024 bilateral HS4 merchandise trade among the United States, Canada, Mexico, China, South Korea, Japan, and the United Kingdom, together with macroeconomic, geographic, and other measurable factors. Treat Alberta as a separate economic entity. Report Alberta’s predicted top ten export destinations and, for each, the predicted top five HS4 products. Distinguish observed 2025 Alberta exports (Q1a) from model-estimated export **potential**.

## Data I used

- **Dependent variable.** CEPII BACI HS22 bilateral values, HS6 rolled to HS4, among the seven economies (2024). I store the feature panel as `data/q1b/training_panel_m1_m2_m3.csv.gz`.
- **Geography / agreements.** CEPII Gravity distance, contiguity, common official language; FTA flag with a 2024 overlay where Gravity’s vintage ends in 2021.
- **Macros (2023 lag).** WDI GDP, GDP per capita, real GDP growth.
- **Tariffs.** WITS TRAINS importer MFN (partner 000), HS6 averaged to HS4. Preferential partner codes were empty, so I keep the FTA dummy as the preference channel.
- **Alberta mass.** StatCan 36-10-0222 (provincial GDP) and 17-10-0005 (population), 2023; CAD to USD via Canada WDI GDP. I commit slim Alberta/Canada slices only.
- **Alberta product prior.** CIMT 2025 Alberta HS4 mix (share / RCA) for scoring. In development I also scored a no-RCA variant to show basket-leakage risk.

## Target and validation

- Target: \(y = \ln(1 + \text{HS4 value USD})\). I convert predictions back with `expm1` for dollar rankings.
- **Holdout.** `GroupShuffleSplit` 75/25 by exporter–importer **pair**, `random_state=42` (no HS4 row from a test corridor appears in training).
- **CV.** 5-fold `GroupKFold` by pair.
- Metric pack: log R², log MAE/RMSE, USD MAE, MdAPE, sMAPE, Spearman of HS4 ranks within corridors, top-5 HS4 hit rate, Spearman of pair totals.
- Gravity-only models have undefined within-corridor Spearman (same prediction for every HS4 in a corridor).

## Models I kept

### Model 1 – Structural gravity (GDP & trade costs) (`M1_structural_gravity`)

HistGBM. Features: `ln_gdp_o`, `ln_gdp_d`, `ln_dist`, `contig`, `comlang_off`, `fta`.  
I use this as the classical gravity baseline before any product structure.

### Model 2 – Product structure & MFN tariffs (`M2_product_and_tariff`)

XGBoost (depth 6, 300 trees, learning rate 0.08, and the other settings in `Q1b.py`).  
Adds: `ln_gdppc_d`, `growth_d`, `hs2`, `energy_hs`, `agri_hs`, `ln_rca`, `ln_mfn`.  
I use this as the product- and tariff-aware benchmark.

### Model 3 – Market demand & exporter capability (`M3_demand_and_capability`)

Same learner as Model 2, plus:

| Feature | Definition |
| --- | --- |
| `imp_hs4_demand_growth` | Importer HS4 import growth 2023→2024 (seven basket), winsorized |
| `ln_imp_hs4_mkt_loo` | ln(1 + importer HS4 imports − this bilateral flow) |
| `ln_world_hs4_mkt_loo` | ln(1 + seven-basket HS4 total − this flow) |
| `trade_complementarity` | Finger–Kreinin index (exporter shares vs importer shares) |
| `ln_hs4_comp` | ln(1 + 1000 × min(export share, import share) at HS4) |
| `pci_proxy` | −ln(ubiquity) + 0.25 ln(1 + #importers) |
| `ln_exp_hs4_share`, `ln_capability`, `ln_exporter_div` | Exporter capability block |

I use leave-one-out market size so the outcome flow is not fed straight back into the feature.

## Alberta scoring assumptions

- Distance from Calgary; contiguity only with the United States.
- FTA, language, and MFN facing Canadian origin (Alberta proxies).
- Model 3 market-size lookups cover the seven training importers. For other Canada partners, I proxy importer HS4 market as world HS4 market × (importer GDP / US GDP).
- Complementarity uses the Canada–importer value when available, else the seven-basket median.
- I calibrate dollar levels with one scale factor so predicted potential sums to Alberta’s observed 2025 domestic-export total (CAD converted at 1.35 CAD per USD). Ranks and mix are the model; the scale is presentation only.

Alberta potential ≠ observed 2025 shipments. Potential can surface destinations that look under-served relative to the training structure.

## Key assumptions I made

1. 2024 seven-economy trade is a useful training ground for Alberta potential in 2025.
2. Pair-based splits prevent product-level leakage across corridors.
3. Ranking quality (top-5 hit, Spearman) matters more than dollar MAPE for this assignment.
4. Observed Q1a shipments and Q1b potential answer different questions and need not match country-by-country.

## How you can replicate or extend

You can do the following to replicate or extend my setup:

- You can refit Models 1–3 with `python3 Q1b.py` using the gzipped panel.
- You can clear and re-run the notebook with `REFIT = True`.
- You can rebuild features from full CEPII BACI / Gravity if you want a from-scratch pipeline outside this slim repo.
