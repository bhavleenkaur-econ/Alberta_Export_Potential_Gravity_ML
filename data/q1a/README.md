# Q1a data

**Question this answers.** Using appropriate data, identify Alberta’s top ten export destination countries in 2025 by domestic export value, and for each destination the top five HS4 products.

I keep only the small StatCan name lists (`ODPF_6_CtyDesc.TXT`, `ODPF_8_ProvDesc.TXT`) and the public HS4 name file here. I do **not** commit the large monthly CIMT CSVs (~125 MB).

You can restore the full extract by running from the repo root:

```bash
python3 Q1a.py
```

That downloads the official 2025 domestic-exports zip into this folder when `ODPFN018_202512N.csv` is missing, then rebuilds `outputs/q1a/`. The committed output CSVs and chart are enough if you only want to read results.
