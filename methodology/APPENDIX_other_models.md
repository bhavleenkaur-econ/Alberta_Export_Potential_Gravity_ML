# Appendix – estimators and specifications I did not keep

## Question context

Same Q1b question as the main methodology: model seven-economy HS4 trade and score Alberta export potential. Here I record trials that informed the three-model ladder but are **not** shipped in `Q1b.py`.

I kept only:

1. Structural gravity (GDP & trade costs)
2. Product structure & MFN tariffs (XGBoost) – formerly labelled XGB_M5 in development
3. Market demand & exporter capability (XGBoost) – formerly labelled XGB_M5_safe_new

## Offline learners on product-structure features

| Estimator | Holdout log R² | Holdout top-5 hit | What I decided |
| --- | ---: | ---: | --- |
| HistGBM | 0.486 | 0.236 | Kept as the Model 1 learner family; XGBoost preferred for Models 2–3 |
| XGBoost | 0.529 | 0.218 | Promoted (stronger CV R² / ranking rule) |
| LightGBM | 0.488 | 0.255 | Promising on one holdout top-5 reading; I did not stabilise it once XGBoost covered boosting |
| ExtraTrees | 0.408 | 0.055 | Dropped – no gain on ranking/R² |
| sklearn GBM | 0.477 | 0.164 | Dropped – no gain on ranking/R² |
| Random Forest | Weaker than HistGBM/XGBoost on fit and ranking | Dropped from production code |

## CatBoost

I fitted CatBoost on several feature sets. It was competitive on log fit but weaker on top-5 hit and Spearman than XGBoost Model 2 (about 0.18 vs 0.25 top-5 hit on the MFN set). Tuned CatBoost (train-pair search only) improved on default CatBoost but still lost the ranking rule. I did not put it in the submission ladder.

## Stacking ensemble

I stacked HistGBM + XGBoost + CatBoost with a Ridge blender fitted on **train out-of-fold** predictions only (holdout labels never used for weights). It matched XGBoost Model 2 on top-5 hit (~0.255) but had lower log R² (~0.49 vs ~0.51). Extra complexity without a clear ranking win, so I left it out.

## Openness trial (WDI trade/GDP)

I added importer trade/GDP and real export/import growth on top of the product-and-tariff set. It tied on top-5 hit but lost Spearman and log R². I did not promote it.

## HHI and Alberta classic RCA trial

I tested leave-one-out importer×HS4 HHI and Balassa Alberta RCA vs Canada / the seven basket. I also clarified that an earlier Alberta `rca_ab` was only `ab_share × n_hs4` (not Balassa). Alberta capability helped the product-and-tariff model alone; HHI alone hurt ranking. A kitchen-sink stack beat Model 2 but did not beat Model 3 (demand & capability) on holdout. I left it as a documented trial.

## LOVO / SHAP on Model 2

- Must-keep features: `ln_rca`, `hs2`, `ln_dist`.
- Weak: `comlang_off`, `contig`.
- I do not drop `energy_hs` despite low SHAP – top-5 hit collapses without it.
- These diagnostics are why I kept the Model 2 feature set intact inside Model 3 rather than aggressively pruning flags.

## Collinear macro trial

Adding GDP per capita and population on both sides together with GDP (`ln(GDP) ≈ ln(GDP/capita) + ln(pop)`) collapsed holdout log R² (~0.04). I kept that only as a failed trial in development notes.

## How you can dig deeper

You can do the following if you want the full trial trail:

- You can reproduce these comparisons from the broader `Alberta_Assignment` development scripts and `output/` tables.
- You can keep this Git package limited to the three submission models and treat this appendix as the decision log only.
