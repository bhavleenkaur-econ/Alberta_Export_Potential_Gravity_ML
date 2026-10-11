# Data terms and how to cite sources

I used third-party open data for this assignment. You should check each provider’s current licence before redistributing raw files.

| Source | What I used it for | Starting point |
| --- | --- | --- |
| Statistics Canada CIMT domestic exports 2025 | Q1a observed Alberta shipments | [Open Government record](https://open.canada.ca/data/en/dataset/2909a648-5753-4924-878a-b069392d9cde); [2025 zip](https://www150.statcan.gc.ca/n1/pub/71-607-x/2021004/zip/CIMT-CICM_Dom_Exp_2025.zip) |
| Statistics Canada tables 36-10-0222 and 17-10-0005 | Alberta GDP and population for Q1b scoring | StatCan website / CSV bulk |
| CEPII BACI HS22 | Q1b bilateral HS4 training panel (pre-built slice in this repo) | [CEPII BACI](http://www.cepii.fr/CEPII/en/bdd_modele/bdd_modele_item.asp?id=37) |
| CEPII Gravity | Distance, contig, language, FTA vintage | [CEPII Gravity](http://www.cepii.fr/CEPII/en/bdd_modele/bdd_modele_item.asp?id=8) |
| World Bank WDI | GDP, GDP per capita, growth | [WDI](https://datatopics.worldbank.org/world-development-indicators/) |
| WITS / UNCTAD TRAINS | Importer MFN tariffs | [WITS](https://wits.worldbank.org/) |
| Harmonized System names | HS4 product labels | [datasets/harmonized-system](https://github.com/datasets/harmonized-system) (ODC-PDDL) |

I commit only slim processed slices and result tables so the Git repo stays small. You can re-download the full StatCan CIMT zip by running `python3 Q1a.py` (it fetches the file when the large CSVs are missing). You can rebuild the Q1b panel from CEPII BACI / Gravity if you need a from-scratch pipeline; the committed `training_panel_m1_m2_m3.csv.gz` is enough to refit Models 1–3.
