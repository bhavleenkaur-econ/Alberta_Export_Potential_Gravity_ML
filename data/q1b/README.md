# Q1b data

**Question this answers.** Build a machine-learning model of 2024 bilateral HS4 trade among seven economies, treat Alberta as a separate exporter, and report predicted top-ten destinations and top-five HS4 products (export **potential**, not Q1a observed shipments).

| File | Role |
| --- | --- |
| `training_panel_m1_m2_m3.csv.gz` | Slim seven-economy HS4 panel with features for Models 1–3 (gzipped for Git) |
| `lookup_*.csv` | Market size, complementarity, complexity helpers for Alberta scoring |
| `alberta_hs4_specialization.csv` | Alberta CIMT HS4 shares / RCA prior |
| `baci_2024_canada_partners.csv` | Partner list for Alberta potential |
| `gravity_can_to_world.csv` | Canada–partner gravity fields |
| `wits_mfn_hs4_2023.csv` | Importer MFN by HS4 |
| `wdi_*` / `wb_*.zip` | Macro inputs |
| `statcan/*_alberta*.csv` | Slim Alberta / Canada GDP and population slices (not full StatCan dumps) |
| `harmonized-system.csv` | HS4 names |

I do not commit CEPII BACI or Gravity mega-zips. You can rebuild a panel from those sources if you extend the pipeline; for replication of Models 1–3, the gzipped panel is enough.
