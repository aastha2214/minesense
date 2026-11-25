# Data

Data files are not committed. Recreate them from the project root:

```bash
python -m minesense.download              # Kaggle copy
python -m minesense.download --with-uci   # plus the original source (for verification)
python -m minesense.audit                 # quality and leakage checks
```

## Sources

- Kaggle: <https://www.kaggle.com/datasets/ritwikb3/land-mines-detection> (listed as CC BY-SA 4.0)
- Original: UCI Machine Learning Repository, *Land Mines*, <https://doi.org/10.24432/C54C8Z> (listed as CC BY 4.0)
- Paper: C. Yilmaz, H. T. Kahraman and S. Söyler, "Passive Mine Detection and Classification
  Method Based on Hybrid Model," *IEEE Access*, vol. 6, pp. 47870-47888, 2018,
  <https://doi.org/10.1109/ACCESS.2018.2866538>

## Columns

| Column | Meaning | Stored as | Physical range in the paper (Table 1) |
|---|---|---|---|
| V | Fluxgate (FLC) sensor output voltage | normalised, at most 1 | 0 to 10.6 V |
| H | Sensor height above ground | 12 levels, k/11 | 0 to 20 cm |
| S | Soil type | integer 1-6 | see below |
| M | Mine type (target) | integer 1-5 | see below |

The files do not document how V and H were normalised.

| S | Soil | M | Class |
|---|---|---|---|
| 1 | Dry and sandy | 1 | No mine (null) |
| 2 | Dry and humus | 2 | Anti-tank |
| 3 | Dry and limy | 3 | Anti-personnel |
| 4 | Humid and sandy | 4 | Booby-trapped anti-personnel |
| 5 | Humid and humus | 5 | M14 anti-personnel |
| 6 | Humid and limy | | |

The original UCI spreadsheet stores soil as 0, 0.2, ..., 1.0 instead of 1-6.
