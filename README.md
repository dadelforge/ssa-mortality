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
- `data/output/eme_catalog.xlsx` — the full catalog of 170 Excess Mortality
  Events (EMEs), each with the extreme temperature events (ETEs), COVID-19
  waves and influenza epidemics it coincides with.
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

The catalog holds every EME detected in the study, ranked by excess mortality
rate, together with the extreme temperature events, COVID-19 waves and
influenza epidemics each one coincides with. There are 170 events, spanning 456
excess weeks and 14,210 excess deaths across the six regions.

An EME is a run of weeks in which observed mortality reaches the z-score-2
threshold of a quasi-Poisson variance model fitted around the SSA baseline,
with a one-week gap tolerance: two runs separated by a single week below the
threshold are recorded as one event. Baselines are fitted on mortality
**rates**, which is what the manuscript results are based on. We refer to the
main manuscript and its Supplementary Digital Content for the full methodology.

Dates are the Monday of their ISO week, and a week is labelled by the Monday it
starts on. Columns typed `list[...]` hold one value per week of the event, in
order, and are serialized as text in the Excel file.

### Identity and timing

| Column     | Type     | Description                                                        |
|------------|----------|--------------------------------------------------------------------|
| rank       | int      | Rank by total excess mortality rate, 1 being the highest. Ranks are assigned across the whole catalog, so a rank cited in the manuscript's ranked table but falling below its cut-off is found at that row here |
| city       | str      | Metropolitan region, named after its principal city (Brussels, Antwerp, Liège, Athens, Larisa, Thessaloniki) |
| country    | str      | Country of the region (Belgium or Greece)                          |
| nuts3_code | str      | NUTS‑3 code of the metropolitan region                              |
| start_date | datetime | Monday of the event's first excess week                            |
| end_date   | datetime | Monday of the event's last excess week                             |
| duration   | int      | Event duration in weeks                                            |
| season     | str      | Season label of the event (winter, spring, summer, autumn)         |

### Mortality metrics

| Column                     | Type        | Description                                                      |
|----------------------------|-------------|------------------------------------------------------------------|
| mortality                  | list[float] | Weekly observed deaths                                           |
| rate                       | list[float] | Weekly observed mortality rate per 100,000                       |
| baseline                   | list[float] | Weekly expected deaths from the SSA baseline                     |
| baseline_rate              | list[float] | Weekly expected mortality rate per 100,000                       |
| population                 | list[float] | Weekly population denominator, interpolated from January 1 figures |
| zscores                    | list[float] | Weekly quasi-Poisson z‑scores                                    |
| z2_limit                   | list[float] | Weekly z=2 threshold, on the count scale                         |
| z2_rate_limit              | list[float] | Weekly z=2 threshold, on the rate scale                          |
| excess_deaths              | list[float] | Weekly deaths above the baseline                                 |
| excess_rate_per_100k       | list[float] | Weekly excess rate per 100,000                                   |
| total_excess_deaths        | float       | Sum of weekly excess deaths over the event                       |
| total_excess_rate_per_100k | float       | Sum of weekly excess rates per 100,000 over the event            |
| population_mean            | float       | Mean weekly population over the event                            |
| z_score_mean               | float       | Mean z‑score over the event window                               |
| z_score_max                | float       | Maximum z‑score during the event                                 |
| z_score_min                | float       | Minimum z‑score during the event                                 |

### Coincidence flags

Each flag records whether the event's weeks overlap the hazard in question. A
flag is a statement of temporal coincidence, not of attribution: an event may
carry several, and 22 of the 30 largest carry more than one.

| Column       | Type | Description                                                        |
|--------------|------|--------------------------------------------------------------------|
| is_hw        | bool | Overlaps at least one heat wave                                    |
| is_cw        | bool | Overlaps at least one cold wave                                    |
| is_covid19   | bool | Falls within a national COVID-19 wave period                       |
| is_influenza | bool | Overlaps a national influenza epidemic period, as reported by Sciensano (Belgium) or EODY (Greece) |
| related      | str  | Rank and region of every other event in the catalog whose weeks overlap this one, e.g. `#6 (Larisa); #9 (Athens)`. Excess in one region is often excess in the same week elsewhere, and ranking by rate separates those rows |

### Matched temperature events

Heat and cold waves are scored independently, each on the variable and minimum
duration adopted in the manuscript: a **heat wave** is at least 2 consecutive
days with daily maximum temperature at or above the 99th percentile, a **cold
wave** at least 4 consecutive days with daily minimum temperature at or below
the 2.5th percentile. Percentiles are computed per region over the 1991–2020
climate reference period. A temperature event counts as coincident when it
overlaps the event's weeks or ended within the preceding seven days.

| Column            | Type       | Description                                                                    |
|-------------------|------------|--------------------------------------------------------------------------------|
| ete               | list[dict] | The matched temperature events. Each record carries the event type, its dates and the daily temperatures over it. Serialized as text in Excel |
| n_ete             | int        | Number of matched temperature events                                           |
| n_ete_HW          | int        | Number of matched heat waves                                                   |
| n_ete_CW          | int        | Number of matched cold waves                                                   |
| influenza_seasons | list[str]  | The influenza seasons the event overlaps, e.g. `['2021-2022']`                 |

## Requirements

- Python 3.9–3.13 (tested with ssalib, later versions may work too but not
  guaranteed).
- Git (optional, if you clone the repository)
- SSALib v0.1.3 (see
  [installation instructions](https://github.com/adscian/ssalib))
- Jupyter Notebook (
  see [installation instructions](https://jupyter.org/install))
