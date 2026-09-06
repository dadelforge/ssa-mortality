# SSA-Based Mortality Baselines and Excess Deaths

> [!NOTE]
> This repository is associated with a manuscript under finalization and under
> construction, and made public for co-author review. This note will be updated
> with the preprint link and citation once available.

## Overview

This repository contains notebooks, scripts, and data snapshots used to explore
and document Extreme Mortality Events (EME) and their relationship with Extreme
Temperature Events (ETE) for Belgium and Greece.

The study units are six NUTS‑3 metropolitan regions: Brussels, Antwerp and
Liège in Belgium, Athens (4 NUTS-3 units combined), Thessaloniki and Larisa in 
Greece. Each covers the wider metropolitan area rather than the municipality alone, 
and therefore includes, to some lesser extent, suburban and rural population. 
Throughout this repository, a region is labeled by the name of its principal city.

## Contents

- `ssa_mortality_tutorial.ipynb`— tutorial illustrating SSA-based mortality
  preprocessing/gap filling, and excess-mortality detection via a
  quasi-Poisson z-score threshold.
- `mortality_modelling/ssa_fill_na.py` — utilities expanding the SSALib package
  to
  handle outlier or missing value interpolation in mortality time series.
- `mortality_modelling/quasi_poisson.py` — quasi-Poisson dispersion model
  (Farrington et al., 1996; Be-MOMO; EuroMOMO-style) used to standardize
  mortality residuals into z-scores/thresholds that scale with the local
  baseline level instead of a constant offset.
- `data/input/df_mortality_TOTAL.csv` — contains mortality time series used in
  the manuscript and in the notebook tutorial.
  Source: [Eurostat](https://doi.org/10.2908/DEMO_R_MWEEK3).
- `data/output/eme_catalog.xlsx` — catalog of Extreme Mortality Events (EMEs)
  and their associated Extreme Temperature Events (ETEs), following various
  percentile-based definitions.
- `supplementary_figures/` — per-region figures supporting the manuscript and its
  Supplementary Digital Content:
    - `ssa_examples/` — four panels per region illustrating how the baseline is
      obtained: the singular spectrum against its Monte-Carlo confidence band,
      the components retained, the reconstructed baseline over the observed
      series, and the residual. Deliberately simple (no COVID masking, no gap
      filling) so that the technique is visible rather than the full pipeline.
    - `ssa_count_baselines/` — Monte-Carlo SSA baselines and quasi-Poisson
      z-score bands fitted on weekly death **counts**, for every combination of
      age group (total, 80+), window length (130, 261 weeks) and maximum AR
      order of the surrogate ensemble (0, 1).
    - `ssa_rate_baselines/` — the same grid fitted on weekly death **rates**,
      which is what the manuscript results are based on.

  Filenames encode the parameters as `{age group}_w{window}_ar{maximum AR
  order}_f{maximum component frequency}_ns{number of surrogates}`.

## Description of `eme_catalog.xlsx`

The catalog aggregates base EME fields with matching ETE and derived
indicators. Column descriptions are provided below. EME are defined as events
relative to the SSA baseline, using a z-score >= 2 threshold with tolerances of
1 week < z-score 2. We refer to the main manuscript and its Supplementary
Digital Content for more details on the methodology.

### Identity and timing

| Column     | Type     | Description                                                        |
|------------|----------|--------------------------------------------------------------------|
| city       | str      | Metropolitan region, named after its principal city (Brussels, Antwerp, Liège, Athens, Larisa, Thessaloniki) |
| country    | str      | Country of the region (Belgium or Greece)                          |
| nuts3_code | str      | NUTS‑3 code of the metropolitan region                              |
| start_date | datetime | EME start date (UTC ISO‑8601 in source; Excel datetime in file)    |
| end_date   | datetime | EME end date (UTC ISO‑8601 in source; Excel datetime in file)      |
| duration   | int      | Event duration in weeks                                            |
| season     | str      | Season label of the event (winter, spring, summer, autumn)         |

### Mortality metrics

| Column                         | Type        | Description                                                      |
|--------------------------------|-------------|------------------------------------------------------------------|
| mortality                      | list[int]   | Weekly observed deaths during the event window (serialized list) |
| zscores                        | list[float] | Weekly z‑scores corresponding to mortality (serialized list)     |
| z2_limit                       | list[float] | Weekly z=2 threshold values (serialized list)                    |
| excess_deaths                  | list[float] | Weekly excess deaths above the baseline (serialized list)        |
| total_excess_deaths            | float       | Sum of weekly excess deaths over the event window                |
| population_jan1                | int         | Population at January 1st for the corresponding region/year      |
| excess_mortality_rate_per_100k | float       | Total excess deaths per 100,000 population                       |
| z_score_mean                   | float       | Mean z‑score over the event window                               |
| z_score_max                    | float       | Maximum z‑score during the event                                 |
| z_score_min                    | float       | Minimum z‑score during the event                                 |

### Event flag

| Column     | Type | Description                                                   |
|------------|------|---------------------------------------------------------------|
| is_covid19 | bool | True if the event falls in a COVID‑19 period; otherwise False |

### Matched temperature events (per percentile pair)

Column names follow this convention: for each percentile pair CW–HW in {(0.5,
99.5), (1.0, 99.0), …, (5.0, 95.0)}
with three‑digit labels CCC (cold) and HHH (heat), e.g., (e.g., 99.0 → `990`,
2.0 →
`020`):

| Column pattern | Type       | Description                                                                                                                         |
|----------------|------------|-------------------------------------------------------------------------------------------------------------------------------------|
| ete_CCC_HHH    | list[dict] | Overlapping ETEs for CW=CCC and HW=HHH. Each item is an ETE record (start/end, event_type, city, etc.). Serialized as text in Excel |
| n_ete_CCC_HHH  | int        | Count of overlapping ETEs for that percentile pair                                                                                  |

### Derived temperature indicators (by percentile)

With HHH and CCC labels as above:

| Column pattern | Type | Description                                            |
|----------------|------|--------------------------------------------------------|
| hw_count_HHH   | int  | Number of matched heatwave (HW) events at threshold H  |
| cw_count_CCC   | int  | Number of matched cold‑wave (CW) events at threshold C |
| is_hw_HHH      | bool | True if `hw_count_H > 0`                               |
| is_cw_CCC      | bool | True if `cw_count_C > 0`                               |

## Requirements

- Python 3.9–3.13 (tested with ssalib, later versions may work too but not
  guaranteed).
- Git (optional, if you clone the repository)
- SSALib (see [installation instructions](https://github.com/adscian/ssalib))
- Jupyter Notebook (
  see [installation instructions](https://jupyter.org/install))
