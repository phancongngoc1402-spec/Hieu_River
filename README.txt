Hieu River robustness reanalysis bundle

Files
-----
Stats_Ready_robustness_input.csv
    Minimal analysis-ready input extracted from the Stats_Ready sheet of the supplied 2025 workbook.

Data_dictionary_robustness_input.csv
    Variable definitions for the minimal analysis input.

robustness_reanalysis.py
    Standalone Python implementation of:
    - station fixed-effects regressions;
    - Bell-McCaffrey CR2 bias-reduced cluster-robust variance;
    - Satterthwaite degrees of freedom;
    - restricted wild cluster bootstrap-t with station-level Rademacher weights;
    - exact enumeration of sign patterns (1,024 for 10 clusters; 512 for leave-one-out; 256 for 8-cluster proxy-excluded models);
    - leave-one-station-out influence analysis;
    - alternative rainfall-window sensitivity.

robustness_results.csv
    Machine-readable results generated for the manuscript.

Reproduction
------------
Python 3 with numpy and scipy:
    python robustness_reanalysis.py

Methodological references
-------------------------
Bell RM, McCaffrey DF (2002) Survey Methodology 28(2):169-181.
Cameron AC, Gelbach JB, Miller DL (2008) Review of Economics and Statistics 90(3):414-427.
Pustejovsky JE, Tipton E (2018) Journal of Business & Economic Statistics 36(4):672-683.

Important
---------
This bundle does not resolve the pending laboratory-method/fraction verification or author metadata.
Those fields remain for the authors to confirm before submission.
