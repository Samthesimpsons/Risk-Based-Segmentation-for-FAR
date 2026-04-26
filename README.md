# SMU Capstone: Profile Coherence as a Diagnostic and Design Lens for Financial Asset Recommendation

A measurement framework and a profile-aware extension of LightGCN for the FAR-Trans dataset. The thesis introduces **Profile Coherence at k (PC@k)** as a new evaluation axis, audits how profile-coherent existing FAR baselines actually are, and proposes a minimal **Profile-Coherent LightGCN** that uses regulatory profile signals as both a conditioning input and a learning constraint.

This README is the single source of truth for the project: thesis framing, methodology, code architecture, and expected outputs all live here.

## Table of Contents

1. [Thesis](#thesis) (full writeup in [thesis.md](thesis.md))
2. [Paper Summary: FAR-Trans](#paper-summary-far-trans)
3. [Problem Statement](#problem-statement)
4. [Research Questions](#research-questions)
5. [Dataset Audit Findings](#dataset-audit-findings)
6. [Novelty and Research Contributions](#novelty-and-research-contributions)
7. [Proposed Approach](#proposed-approach)
8. [Source Code Architecture](#source-code-architecture)
9. [Expected Outputs](#expected-outputs)
10. [Risks and Mitigations](#risks-and-mitigations)
11. [Working with this Repository](#working-with-this-repository)
12. [GPU Cluster](#gpu-cluster)
13. [References](#references)

## Thesis

The full thesis writeup (Abstract, Introduction, Methodology, Findings, Ablation Studies, Discussion, Conclusion) lives in [thesis.md](thesis.md). The numerics, tables, and figures there come from prior runs and will be regenerated when the pipeline is rerun fresh on the cluster; methodology, ablation design, and panel-regression specification are stable.

## Paper Summary: FAR-Trans

This section summarises the FAR-Trans paper that forms the foundation of this project.

I have also taken the liberty of converting the original paper from `TeX` into `markdown` format so that LLM-based tooling can ingest it as grounding context. The markdown copy lives under `papers/FAR-Trans An Investment Dataset for Financial Asset Recommendation/paper.md`.

### What is FAR?

Financial Asset Recommendation (FAR) identifies and ranks financial securities for investors based on their suitability.

Suitability depends on:

| Factor | Examples |
|---|---|
| **Investor-side** | Past transactions, risk tolerance, investment capacity, personal goals |
| **Market-side** | Asset returns, currency value, inflation |

FAR systems analyse multiple data sources:

- Time series pricing data
- Customer profile data
- Past investment transactions

### Paper Contribution

Most existing FAR models are developed over **proprietary or simulated datasets**, making fair comparison across methods impossible. The only prior public dataset (ObjectWay, Musto et al. 2014) has 1,172 users but **lacks pricing data and asset identifiers**, which prevents price-based approaches from being tested.

**FAR-Trans fills this gap**: the first public dataset for FAR that contains both real asset pricing information and real retail investor transactions. The paper also provides a **benchmark comparison of 11 FAR algorithms** as baselines for future research.

### Dataset

[FAR-Trans](https://doi.org/10.5525/gla.researchdata.1658) (Sanz-Cruzado et al., 2024): the first public dataset for financial asset recommendation containing both pricing information and retail investor transactions, collected from a large European financial institution (the National Bank of Greece) and covering January 2018 to November 2022.

#### What the Dataset Contains

| Component | Description |
|---|---|
| **Price time series** | Daily closing prices for 806 financial assets (stocks, bonds, mutual funds) across 38 markets. 703,303 price data points total. |
| **Investment transactions** | 388,049 buy/sell records from 29,090 retail investors. |
| **Asset metadata** | Per-asset asset type, sub-type, market, sector, industry. |
| **Customer profiles** | Per-customer segment, MiFID II risk profile, and investment capacity. |

#### Dataset Statistics at a Glance

| Property | Value |
|---|---|
| Unique assets | 806 |
| Assets with at least one transaction | 321 |
| Unique customers | 29,090 |
| Total transactions | 388,049 (154,103 unique user-asset pairs) |
| Acquisitions (buys) | 228,913 (89,884 unique) |
| Sales (sells) | 159,136 (64,219 unique) |
| Time span | Jan 2018 - Nov 2022 |
| Average return (by assets, whole period) | 37.16% |
| % profitable assets | 54.28% |
| Average return (by customers, whole period) | 22.89% |
| % customers with profits | 54.56% |

#### Customer Segments and MiFID Risk Profiles

The two regulatory signals that this thesis treats as load-bearing:

- **MiFID II risk profile** (`riskLevel`): four declared bands (`Conservative`, `Income`, `Balanced`, `Aggressive`) plus regression-imputed `Predicted_*` variants for customers who never completed the questionnaire.
- **Investment capacity** (`investmentCapacity`): four tiers (`CAP_LT30K`, `CAP_30K_80K`, `CAP_80K_300K`, `CAP_GT300K`).
- **Customer segment** (`customerType`): five segments (`Mass`, `Premium`, `Professional`, `Legal Entity`, `Inactive`).

These three categorical signals form the **profile vector** that conditioning and PC@k rely on.

#### Cleaning and Pre-processing

All cleaning has been done by the FAR-Trans authors. See the original [dataset documentation](https://researchdata.gla.ac.uk/1658/) for the full schedule of price normalisation, transaction synthesis, and stock-split adjustment.

### Existing FAR Methods

The FAR-Trans paper benchmarks three families:

1. **Price-based** (Linear Regression, Random Forest, LightGBM): predict asset profitability from technical indicators. Best at ROI@10 (Random Forest reaches 0.0259 monthly), worst at nDCG@10 (near-random preference accuracy).
2. **Transaction-based** (Popularity, Matrix Factorisation, LightGCN, ARM, UB-kNN): predict the next purchase from interaction history. Best at nDCG@10 (LightGCN reaches 0.3404), worst at ROI@10 (near-zero realised return).
3. **Hybrid** (Hybrid-nDCG, Hybrid-regression): two-stage pipelines combining the above. Neither dominates either axis.

The headline finding is the **negative correlation between nDCG@10 and ROI@10**: methods that win one objective lose the other. This is the gap the original sequential-modelling thesis tried to close. The revised thesis treats that conflict as a *symptom* of a deeper problem: customer transactions encode systematic behavioural bias, and current FAR systems have no reason to filter that out because the regulatory profile signal is not part of any baseline's loss or evaluation.

### Benchmark Results (Paper Table 2)

| Data Source | Algorithm | nDCG@10 | ROI@10 |
|---|---|---|---|
| None | Random | 0.0106 | 0.0071 |
| Prices | Random Forest | 0.0237 | **0.0259** |
| Prices | Linear Regression | 0.0215 | 0.0249 |
| Prices | LightGBM | 0.0221 | 0.0225 |
| Transactions | Popularity | 0.2710 | 0.0006 |
| Transactions | LightGCN | **0.3404** | 0.0004 |
| Transactions | ARM | 0.2556 | 0.0007 |
| Transactions | MF | 0.1780 | 0.0038 |
| Transactions | UB kNN | 0.1599 | 0.0119 |
| Hybrid | Hybrid-nDCG | 0.2313 | 0.0063 |
| Hybrid | Hybrid-regression | 0.0261 | 0.0132 |
| - | Market average | - | 0.0079 |
| - | Customer average | - | 0.0018 |

## Problem Statement

> **Existing FAR systems train against observed user transactions, but those transactions encode systematic behavioural bias documented in the finance literature (Barber and Odean 2000 / 2008; Kumar 2009). FAR-Trans contains a regulatory ground-truth signal that is almost entirely ignored: the customer's MiFID II risk profile. Using this signal as both an evaluation axis (does the recommender match the user's declared risk band?) and a learning signal (penalise high scores assigned to profile-discordant assets) reframes the FAR problem as a profile-coherence question rather than a pure preference vs profit trade-off.**

The original thesis (sequential SASRec / TiSASRec / Hybrid Dual-Head models for FAR) was dropped as it was a score-chasing exercise rather than a problem-finding contribution. The revised thesis frames the FAR problem around an under-exploited regulatory signal in the FAR-Trans dataset.

## Research Questions

1. **RQ1 (Diagnostic):** What is the distribution of profile-discordance in the FAR-Trans transaction record, broken down by `customerType`, `riskLevel`, and market regime? *Answered: see [Dataset Audit Findings](#dataset-audit-findings).*
2. **RQ2 (Causal):** Controlling for asset volatility, customer segment, and time, do profile-discordant transactions earn lower realised 6-month excess return than profile-coherent ones? *Tested via panel regression.*
3. **RQ3 (Audit):** What is the Profile Coherence at 10 (PC@10) of each FAR-Trans baseline (Random Forest, LightGCN)? Does the nDCG-ROI tradeoff explain itself partially as a profile-coherence axis? *Tested via the baseline grid sweep.*
4. **RQ4 (Method):** Does adding (a) profile conditioning and (b) a profile-coherence regulariser to LightGCN improve PC@10 and ROI@10 with minimal nDCG@10 cost? Which of the two components carries the gain? *Tested via Profile-Coherent LightGCN.*

## Dataset Audit Findings

The dataset audit (`uv run poe eda`, source in `src/pipeline/eda.py`, full numbers in `outputs/eda/summary.json`) tests whether profile-coherence is a load-bearing signal in FAR-Trans. The headline numbers below all come from the hierarchical asset risk-class mapping (mutual funds via `assetSubCategory`, bonds via subcategory, stocks via 252-day annualised volatility quartile).

### Coverage

| Quantity | Value | Notes |
|---|---|---|
| Total customers | 29,090 | Matches FAR-Trans paper |
| Customers with usable MiFID band | 28,770 (98.9%) | 7,141 (24.5%) regression-imputed `Predicted_*` |
| Customers with `Not_Available` band | 320 (1.1%) | Excluded from PC@k |
| Total assets classified | 806 | 100% mapping coverage |
| Total Buy transactions | 228,913 | |
| Buy transactions scoreable for PC | 228,241 (99.7%) | |

The framework can score 99.7% of Buy transactions, so any sample-selection bias from missing profile signals is negligible.

### Finding 1: 18.6% of FAR-Trans Buy transactions are profile-discordant

Across 228,241 scoreable Buy transactions, the discordance distribution is:

| Discordance `d` | Count | Share |
|---|---|---|
| `d = 0` (exact band match) | 77,925 | 34.1% |
| `d = 1` (within tolerance) | 107,899 | 47.3% |
| `d = 2` | 35,581 | 15.6% |
| `d = 3` (extreme mismatch) | 6,836 | 3.0% |

Under the default tolerance (`d <= 1`) **81.4% of transactions are profile-coherent**, leaving **18.6% (42,417 transactions) that violate the user's declared MiFID band by 2 or more steps**. Mean discordance is 0.87 bands. See `outputs/eda/transaction_discordance_distribution.png`.

### Finding 2: Self-discordance is a customer-level trait, not transaction-level noise

The per-customer fraction of discordant transactions is bimodal, not normal:

- **64.4% of customers are fully profile-coherent** (every transaction within `d <= 1`).
- **17.2% of customers are fully discordant** (every transaction at `d >= 2`).
- 20.3% have a majority-discordant trading record.
- The middle of the distribution is sparse.

This U-shape (`outputs/eda/customer_self_discordance_histogram.png`) is the strongest empirical motivation for the thesis: discordance is *systematic at the customer level*, not random noise. A recommender that ignores it inherits the same bias. Behavioural-finance literature (Barber and Odean 2000 / 2008; Kumar 2009) predicts exactly this concentration.

### Finding 3: Extreme-band customers reach toward the centre

Decomposing transaction discordance by the customer's declared MiFID band reveals a regression-toward-the-mean pattern:

| Declared MiFID Band | Transactions | Coherent share (`d <= 1`) |
|---|---|---|
| Conservative | 14,193 | **45.1%** |
| Income | 73,468 | 90.9% |
| Balanced | 99,901 | 89.9% |
| Aggressive | 40,679 | **56.2%** |

Mid-band customers (Income, Balanced) are roughly 90% profile-coherent. The two extreme bands are dramatically less coherent: more than half of Conservative customers' Buy transactions land at `d >= 2` (chasing risk), and a similar fraction of Aggressive customers' purchases land in safer assets than their profile permits (loss-aversion or yield-chasing). See `outputs/eda/discordance_by_risk_level.png`.

### Finding 4: Discordance is stable across regimes

Mean discordance per calendar year stays in the 0.83-0.98 band across 2018-2022 (`outputs/eda/discordance_by_year.png`). The COVID-19 crash and recovery do not visibly shift the pattern. This rules out a "panic trading drove the discordance" explanation and supports the customer-level-trait reading from Finding 2.

### Finding 5: Hierarchical mapping is more lenient than pure volatility, but not by much

The sensitivity check uses pure volatility quartiles for *all* assets (no metadata):

| Mapping | `d <= 1` share | Strict `d == 0` share | Mean `d` |
|---|---|---|---|
| Hierarchical (default) | 81.4% | 34.1% | 0.87 |
| Pure volatility quartiles | 73.2% | 26.8% | 1.03 |

Switching to a pure-volatility mapping shifts the headline `d <= 1` rate down by 8 pp. The qualitative findings (1-4) hold under both mappings; we report the hierarchical numbers as the primary result because using regulatory metadata where it exists is a regulator-aligned design choice, and report the volatility numbers in the sensitivity table.

### What this implies for the thesis

The 18.6% discordant share with a customer-level concentration justifies treating profile-coherence as a load-bearing signal. The remaining empirical questions:

- **Do profile-discordant transactions earn lower realised excess returns?** Tested via a panel regression (RQ2).
- **Do high-nDCG FAR baselines reproduce this discordance?** Tested by computing PC@10 on the existing baselines (RQ3).
- **Can a minimal architectural change correct it?** Tested via Profile-Coherent LightGCN (RQ4).

## Novelty and Research Contributions

1. **Profile Coherence at k (PC@k).** A new evaluation metric that scores the share of recommended top-k assets whose risk class is within one MiFID band of the user's declared profile. Reported alongside nDCG, ROI, and Recall on every model.
2. **Self-discordance audit of FAR-Trans.** Empirical decomposition (above) of how often customers transact outside their declared risk band, and how that decomposes by segment, declared band, and time. The U-shaped per-customer distribution (Finding 2) and the band-asymmetry pattern (Finding 3) are new findings about the dataset itself.
3. **Profile-Coherent LightGCN.** A minimal extension of LightGCN that conditions on `(riskLevel, customerType, investmentCapacity)` via a summed embedding and adds a profile-coherence regulariser to the BPR loss. The architecture delta is one embedding sum and one extra loss term. This shows profile coherence can be injected as a *learning* signal, not only as a post-hoc re-ranking constraint (the path taken by RURA, Kim et al. 2025).

## Proposed Approach

### Risk-class assignment for assets

Each asset is mapped to one of the four MiFID bands using a hierarchical rule:

1. **Mutual funds (`assetCategory == 'MTF'`)** carry an `assetSubCategory` mapped directly: `Money Market` -> Conservative, `Bond` / `Bonds` -> Income, `Balanced` -> Balanced, `Equity` / `Large Cap` -> Aggressive. `Other` and `Structured` fall through.
2. **Bonds (`assetCategory == 'Bond'`)** with `Government` -> Conservative, `Corporate` -> Income, otherwise Income.
3. **Stocks and remaining assets** are binned by trailing 252-trading-day annualised volatility quartile: lowest quartile -> Conservative, ..., top quartile -> Aggressive.

The bands are encoded as ordinals: Conservative=0, Income=1, Balanced=2, Aggressive=3.

### Profile-discordance and PC@k

For a customer `u` with declared band `b_u` and an asset `i` with band `b_i`:

```
discordance(u, i) = |b_u - b_i|
PC@k = (1/k) * |{i in top_k : discordance(u, i) <= 1}|
```

Customers whose `risk_band` is `None` (raw `Not_Available`) contribute 0.0 to PC@k. The strict variant (`discordance == 0`) is reported as a sensitivity row.

### Profile-Coherent LightGCN

Two minimal additions on top of `LightGCNBaseline`:

1. **Profile embedding.** Three small `nn.Embedding` tables keyed on `(risk_band, customer_type, investment_capacity)`. The lookup vectors are summed, projected to the LightGCN embedding dimension, and added to the user embedding *before* the LGConv stack. Toggle via `profile_embedding_enabled`.
2. **Profile-coherence regulariser.** The total loss becomes
   ```
   L = L_BPR + lambda_pc * E_{(u, i_pos) in batch} [ d(u, i_pos) * sigmoid(score(u, i_pos)) ]
   ```
   This penalises high scores assigned to profile-discordant positives, weighted by the discordance distance. Toggle via `profile_coherence_enabled`. The strength is `profile_coherence_lambda`.

Setting both toggles to False reduces the model exactly to vanilla LightGCN, which is the cleanest possible 2x2 ablation: each row of the ablation table corresponds to one cell of `(profile_embedding_enabled, profile_coherence_enabled)`.

### Evaluation Schedule

The original FAR-Trans evaluation schedule is preserved unchanged: 69 evenly-spaced evaluation splits at ~9-trading-day intervals, each with a 6-month test window. See `src/data/splitting.py` and the FAR-Trans paper's Section 4 for the full mechanics.

## Source Code Architecture

### Module Structure

```
src/
    config/
        settings.py             # Pydantic BaseSettings: hyperparameters and data paths
        schemas.py              # TemporalSplitData, CustomerProfile, EvaluationResult
    data/
        loading.py              # Load raw CSVs (FAR-Trans cleaning conventions)
        splitting.py            # 69 temporal train/test splits
    features/
        technical_indicators.py # 30-column indicator set for the Random Forest baseline
    utils/
        profile_coherence.py    # Risk-band assignment, customer profile parsing, discordance, coherence predicate, calendar/trading-day constants, RISK_BAND_NAMES
        metrics.py              # nDCG@k, ROI@k, Recall@k, PC@k, PC-lift@k
    models/
        protocol.py             # Recommender Protocol (structural type only)
        random_forest.py        # Price-based RF regressor (paper baseline)
        light_gcn.py            # LightGCN collaborative filtering (paper baseline)
        pc_light_gcn.py         # Profile-Coherent LightGCN (method-contribution model)
    pipeline/
        preprocessing.py    # Generate splits, save to disk
        eda.py              # Dataset audit (figures + summary.json); the transaction-discordance annotator lives here since it only has one consumer
        tune.py             # End-to-end pipeline: RF + LightGCN grid + PC-LGCN sweep + 3-model decomposition + PC-LGCN ablation tables + both canonical panel regressions, all in one driver
```

The `analysis/` subpackage was retired in an earlier consolidation: the post-evaluation analyses (`run_decomposition`, `run_profile_coherent_decomposition`, `run_panel_regression`) are now module-private functions inside `pipeline/tune.py`, called sequentially by `run_tune` after the two grid sweeps finish.

### Configuration

- `src/config/settings.py`: `RandomForestConfig`, `LightGCNConfig`, `ProfileCoherentLightGCNConfig`, `ExperimentConfig`, `DataPaths`. The new model's config carries six profile-related fields (the embedding dimension, the two boolean toggles, the regulariser scale, and the squared-distance flag).
- `src/config/schemas.py`: `TemporalSplitData` (one split), `CustomerProfile` (regulatory record per customer), `EvaluationResult` (now includes `profile_coherence_at_k`).

### Models

All three models implement the `Recommender` Protocol:

```python
class Recommender(Protocol):
    @property
    def name(self) -> str: ...
    def train_on_split(self, split: TemporalSplitData, **kwargs: object) -> None: ...
    def recommend_for_user(self, user_id: str, excluded_assets: set[str], k: int = 10) -> list[str]: ...
```

There is deliberately no central registry. With three known recommenders, each pipeline (baselines, profile-coherent) constructs its models directly: the registry indirection of the legacy code added complexity without buying anything.

| Model | Category | Data Source | Key Mechanism |
|---|---|---|---|
| **Random Forest** | Price-based | Technical indicators | sklearn regressor predicts forward 126-trading-day ROI; ranks all assets identically for every user |
| **LightGCN** | Transaction-based | User-asset bipartite graph | PyTorch Geometric `LGConv` (canonical symmetric `D^{-1/2}AD^{-1/2}`, no self-loops); BPR loss with L2 on initial embeddings |
| **Profile-Coherent LightGCN** | Transaction + regulatory profile | Interaction graph + customer profiles + asset risk classes | LightGCN backbone with summed `(risk, segment, capacity)` profile embedding and a profile-coherence regulariser added to BPR |

### Profile-Coherence Module

`src/utils/profile_coherence.py` is the shared module that owns the measurement framework:

- Asset risk classification: `build_asset_risk_classes` returns `dict[asset_id, ordinal_band]` from `asset_information.csv` and `close_prices.csv`. The current default uses a single (full-period) volatility-quartile mapping for stocks; per-time-point refitting is a planned sensitivity check. The annualised-volatility helper `compute_annualised_volatility` is exported for the dataset audit's volatility-distribution figure.
- Customer profile parsing: `build_customer_profile_lookup` returns `dict[customer_id, CustomerProfile]` from `customer_information.csv`. Maps `Predicted_*` risk levels onto the same ordinal scale as declared ones, with a flag.
- Discordance and coherence: `compute_pairwise_discordance` returns `d(b_u, b_i)`; `is_profile_coherent` is the `d <= 1` predicate (with a strict `d == 0` mode). Single-consumer extensions live with their consumer; the transaction-level annotator that the dataset audit uses sits in `src/pipeline/eda.py`.

### Pipeline Orchestration

```sh
uv run poe preprocess           # Step 0: generate splits to data/splits/
uv run poe eda                  # Step 1: dataset audit -> outputs/eda/
uv run poe tune                 # Step 2: end-to-end pipeline (baseline grid + PC-LGCN sweep + 3-model decomposition + canonical panel regressions)
```

`tune` is a single end-to-end driver: it runs the RF + LightGCN grid sweep, saves the baseline best-config JSON, runs the 7-trial PC-LGCN sweep at the winning LightGCN backbone, writes the 3-model decomposition and PC-LGCN ablation + lambda-sensitivity tables, and produces both canonical panel regressions (PC-LGCN pinned to the `L_PC`-only cell and to the full-method cell). Wall-time budget is roughly 22 hours on a single L40s.

The post-evaluation analyses (`run_decomposition`, `run_profile_coherent_decomposition`, `run_panel_regression`) are module-private functions inside `src/pipeline/tune.py` and are invoked sequentially by `run_tune` after the two grid sweeps finish; they can also be called directly as Python imports against an existing run timestamp if a single artefact needs to be regenerated.

### Baseline Grid Sweep

The baseline grid stage of `src/pipeline/tune.py` (functions `run_baseline_grid_search` and the `_run_baseline_grid_for_spec` family) runs a Ray-driven grid sweep across RF (12 trials) and LightGCN (8 trials). Each trial is a **full 69-split evaluation**, so the per-trial summary is directly comparable to the FAR-Trans paper's Table 2.

| Model | Grid axes | Trials | Primary metric |
|---|---|---|---|
| Random Forest | `number_of_estimators in {20, 30, 40, 50}` x `max_depth in {None, 15, 25}`; other axes pinned to paper | 12 | ROI@10 |
| LightGCN | `embedding_dimension in {64, 128}` x `number_of_layers in {2, 3}` x `learning_rate in {1e-2, 1e-3}`; other axes pinned to paper | 8 | nDCG@10 |

**Paper defaults are explicitly included** in both grids (`n_estimators=20, max_depth=None` for RF; `emb_dim=64, layers=3, lr=1e-2` for LightGCN), so benchmark replication is always one of the trial points. The RF grid was expanded after a first run showed `max_depth=None` consistently outperforming capped depths and `n_estimators=100` adding cost without ROI gain; the current grid drops `n=100`, fills in `n in {30, 40}`, and adds `max_depth in {15, 25}` as sensitivity rows. The LightGCN grid is kept tight (8 trials) because the Profile-Coherent LightGCN pipeline reuses its winning backbone rather than re-sweeping every axis.

Outputs after one run:

- `outputs/results/evaluation/{model}/{timestamp}/{trial_id}/per_split_metrics.csv`: per-trial per-split scalar metrics.
- `outputs/results/evaluation/{model}/{timestamp}/{trial_id}/recommendations.parquet`: flat per-recommendation rows with `monthly_return` and `is_relevant` precomputed; the source of truth for any future decomposition or re-aggregation.
- `outputs/results/tuning/{model}/{timestamp}.csv`: per-trial roll-up (one row per trial, four averaged metrics).
- `outputs/configs/{timestamp}/best_hyperparameters.json`: best trial per primary metric.
- `outputs/analysis/baseline_decomposition/{timestamp}/`:
    - `main_results.csv`: best trial per model with means and standard deviations across splits.
    - `decomposition.csv`: profile-coherent vs profile-discordant ROI breakdown per model.
    - `scatter_ndcg_vs_pc.png`, `scatter_pc_vs_roi.png`: trade-off scatter plots.
    - `summary.json`: machine-readable headline numbers.

#### Resource model

The cluster job (`scripts/tune.sh`) requests the SMU studentqos envelope: 1 L40s GPU, 4 CPUs, 16 GB RAM, 1-day cap. Ray distributes:

- RF: 4 concurrent CPU trials (1 CPU each, sklearn single-threaded), so the 12 trials run as three waves of 4.
- LightGCN: 4 concurrent GPU trials with `gpu = 1 / max_concurrent_trials = 0.25` fractional sharing on the single L40s, so the 8 trials run as two waves of 4.

#### Validation/Evaluation Window Overlap (Known Caveat)

The legacy validation-split tuning had a known overlap issue between validation and early evaluation splits. The new design eliminates this entirely: every trial is a full 69-split evaluation, so there is no separate validation set whose splits could overlap with the benchmark.

### Profile-Coherent LightGCN Sweep

The PC-LGCN stage of `src/pipeline/tune.py` (function `run_profile_coherent_grid`) runs a focused 7-trial sweep on top of the winning LightGCN backbone. The backbone hyperparameters are read from `outputs/configs/<latest>/best_hyperparameters.json` (selected by mean nDCG@10 in the baseline stage). All seven trials use the same backbone; only the profile-conditioning components vary.

| Sub-sweep | Axes | Trials | Purpose |
|---|---|---|---|
| 2x2 ablation | `profile_embedding_enabled in {False, True}` x `profile_coherence_enabled in {False, True}`, `lambda_pc = 0.5` (paper default) | 4 | Attribute gains to each component independently. The (False, False) cell reduces exactly to vanilla LightGCN, so the table reads as a delta against the baseline. |
| Lambda sensitivity | `profile_coherence_lambda in {0.1, 1.0, 2.0}` with both flags True (`0.5` is reused from the ablation cell) | 3 | Quantify robustness of the regulariser strength. |

Selecting the backbone first on nDCG@10 means the ablation answers the question *"given the best plain backbone, do profile-conditioning components add coherence without hurting accuracy?"* This avoids confounding ablation with hyperparameter sensitivity. The thesis flags this as a deliberate scope choice; if the winning cell underperforms, the runner-up backbone can be retried at low cost since each PC-LGCN run is short.

Outputs after one run:

- `outputs/results/evaluation/profile_coherent_light_gcn/{timestamp}/{trial_id}/per_split_metrics.csv` and `recommendations.parquet`: same schema as the baseline pipeline, so the decomposition module consumes them unchanged.
- `outputs/results/tuning/profile_coherent_light_gcn/{timestamp}.csv`: per-trial roll-up.
- `outputs/analysis/baseline_decomposition/{timestamp}/main_results.csv`: now includes a Profile-Coherent LightGCN row alongside the two baselines.
- `outputs/analysis/profile_coherent_decomposition/{timestamp}/`:
    - `ablation.csv`: 4 rows, one per (profile-embedding, L_PC) cell, with means + std + per-trial coherent/discordant ROI.
    - `lambda_sensitivity.csv`: 4 rows (the ablation's both-on cell + 3 sensitivity points) ordered by lambda.
    - `summary.json`: machine-readable headline numbers.

Resource model: the combined SLURM job (`scripts/tune.sh`) runs the SMU studentqos envelope (1xL40s, 4 CPUs, 16 GB RAM, 1-day cap) across all stages. Ray distributes 4 concurrent GPU trials at 0.25 GPU each during the LightGCN and PC-LGCN stages, so the 8 LightGCN trials run as two waves and the 7 PC-LGCN trials run as roughly two waves.

### Panel Regression

The panel-regression analysis (function `run_panel_regression` inside `src/pipeline/tune.py`) tests RQ2/RQ3 jointly: does the per-band coherence gap close or widen between models? Each model's best-trial recommendations parquet is reduced to one row per (customer, split) carrying the customer's coherent share for that split. The model fits

```
coherent_share ~ C(declared_band) * C(model) + C(split_index)
```

with cluster-robust standard errors on `customer_id`. The interaction terms are the headline coefficients: each one quantifies how a given model shifts a given band's coherence relative to the reference cell (Conservative band, alphabetically-first model).

Outputs under `outputs/analysis/panel_regression/{timestamp}/`:

- `coefficients.csv`: every coefficient with point estimate, SE, t, p, and 95% CI.
- `predicted_pc_by_band_model.csv`: model-implied PC@10 per (band, model) cell at the median split, with 95% CIs.
- `forest_predicted_pc.png`: forest figure of the predicted PC@10 grid, which is the headline visualization for the thesis's structural-discordance claim.
- `regression_summary.txt`: the full statsmodels OLS summary.
- `panel.csv`: the assembled (customer, split, model) panel for replication.

## Expected Outputs

### Tables

1. **Table 1** (main results): `Model x {nDCG@10, ROI@10, Recall@10, PC@10}` for RF, LightGCN, Profile-Coherent LightGCN over 69 splits, mean +/- standard deviation across splits.
2. **Table 2** (ablation): `LightGCN-variant x metrics` with 2x2 toggle of profile-embedding and L_PC, plus a 3-point lambda sensitivity sweep on the both-flags-on cell.
3. **Table 3** (panel regression): per-band predicted PC@10 with cluster-robust SEs on `customer_id`.

### Figures

1. **Figure 1**: distribution of asset risk-class assignments under the hierarchical mapping.
2. **Figure 2**: per-customer self-discordance histogram, by `customerType` and `riskLevel`.
3. **Figure 3**: ROI by discordance-bin (0, 1, 2, 3) on actual transactions.
4. **Figure 4**: decomposition: per-baseline ROI@10 on profile-coherent vs discordant top-10 subsets.
5. **Figure 5**: (nDCG@10, PC@10) scatter and (PC@10, ROI@10) scatter, with one point per baseline.
6. **Figure 6**: `lambda_pc` sweep on Profile-Coherent LightGCN: Pareto curve over (nDCG, ROI, PC).

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Volatility-quartile risk-class binning is sensitive to window length | Medium | Could weaken claim | Window fixed at 252 trading days before looking at headline numbers; Audit Finding 5 shows hierarchical mapping vs pure volatility agree within 1-2 pp of cross-band assignment |
| Profile-Coherent LightGCN has worse PC@10 than expected | Medium | Could weaken claim | The 2x2 component ablation (profile-embedding x `L_PC`) isolates each component independently; if either component fails to improve PC@k, the negative result itself is publishable as evidence on the role of profile signals in BPR-trained recommenders |
| Mutual-fund subcategory mapping is too coarse (e.g., "Other" / "Structured" categories) | Medium | Adds noise to risk-class | Volatility tiebreaker for "Other"/"Structured" rows; the residual noise is absorbed by the sample size (806 assets, 1.37M recommendations evaluated) |
| `Predicted_*` risk levels behave differently from declared | Medium | Could bias PC@10 | Both declared and `Predicted_*` bands are pooled into the customer profile lookup with an explicit `is_predicted` flag, so a strict-declared-only sensitivity check is one boolean filter away |

## Working with this Repository

### Prerequisites

- **[uv](https://docs.astral.sh/uv/getting-started/installation/)**: package and dependency manager

### Setup

```sh
uv sync                               # install dependencies
uv run poe setup                      # install lefthook git hooks
source .venv/bin/activate
pip install graphifyy                 # optional: knowledge-graph integration
graphify claude install               # optional: Claude Code integration
```

### Running the Full Pipeline

A complete reproduction of the thesis paper's tables and figures runs in three commands.

**Step 1: Preprocess (local, one-off).** Loads raw FAR-Trans CSVs and emits the 69 monthly temporal splits and preprocessed close prices to `data/splits/`.

```sh
uv run poe preprocess
```

**Step 2: Dataset audit (local).** Produces the discordance histograms, per-customer self-discordance distributions, and segment / year breakdowns under `outputs/eda/`. Backs the [Dataset Audit Findings](#dataset-audit-findings) section.

```sh
uv run poe eda
```

**Step 3: End-to-end evaluation pipeline (cluster).** A single GPU job that runs the baseline grid (RF + LightGCN), the PC-LGCN ablation and lambda sweep, the 3-model decomposition, and the two canonical panel regressions in one invocation.

```sh
sbatch scripts/tune.sh
```

The script runs `uv run poe tune` under SLURM. Wall-clock budget is ~22 hours on a single L40s, dominated by the LightGCN grid (~16 h for 8 trials at 60 min each) and the PC-LGCN sweep (~7 trials at the same backbone). For a local CPU smoke test on a few splits:

```sh
uv run poe tune --splits-limit 2 --max-concurrent-trials 1 --device cpu
```

After step 3, all artefacts the thesis paper cites are on disk:

| Artefact | Location |
|---|---|
| Dataset audit figures | `outputs/eda/` |
| Per-trial metrics + recommendations | `outputs/results/evaluation/{model}/<ts>/<trial>/` |
| Tuning summaries | `outputs/results/tuning/{model}/<ts>.csv` |
| Best baseline configs | `outputs/configs/<ts>/best_hyperparameters.json` |
| Three-model main results + decomposition | `outputs/analysis/baseline_decomposition/<ts>/` |
| PC-LGCN ablation + lambda sensitivity | `outputs/analysis/profile_coherent_decomposition/<ts>/` |
| Three-model panel (`L_PC`-only canonical cell) | `outputs/analysis/panel_regression_three_models/<ts>/` |
| Three-model panel (full-method canonical cell) | `outputs/analysis/panel_regression_three_models_full_method/<ts>/` |

#### Re-running individual analysis steps

The post-evaluation analyses (`run_decomposition`, `run_profile_coherent_decomposition`, `run_panel_regression`) are module-private functions inside `src/pipeline/tune.py`. To recompute a single artefact from existing per-trial outputs without retraining, import the function directly and pass the relevant `run_timestamp`:

```python
from src.pipeline.tune import run_decomposition, run_panel_regression
run_decomposition(evaluation_directory=Path("outputs/results/evaluation"), run_timestamp="<ts>")
```

For a partial GPU re-run (e.g., reproducing only the PC-LGCN sweep against an existing baseline best-config JSON), call `run_profile_coherent_grid` from `src.pipeline.tune` programmatically.

### Common Tasks

```sh
uv run poe lint                                                  # ruff
uv run poe type                                                  # ty
uv run poe format                                                # ruff format

uv run poe tune --splits-limit 2 --max-concurrent-trials 1 --device cpu             # end-to-end smoke test
```

### Git Hooks

[Lefthook](https://github.com/evilmartians/lefthook) manages the hooks:

- **Pre-commit**: lint, format, typecheck.
- **Post-commit / post-checkout**: rebuild the graphify knowledge graph for changed code files.

## GPU Cluster

The SMU `msc` partition (1 L40s GPU, 4 CPUs, 32 GB RAM, 5-day max job time, `studentqos`) is the standard target for the grid sweep. SSH via `samuel.sim.2024@origami.smu.edu.sg` (GlobalVPN required).

### Submitting the Pipeline

A single SBATCH job runs the full thesis pipeline end-to-end:

```bash
sbatch scripts/tune.sh    # ~22h on 1xL40s; produces every artefact the thesis paper cites
```

`scripts/tune.sh` loads the cluster Python and CUDA modules, activates the venv, and runs `uv run poe tune --device cuda`. That command sequentially: (i) runs the RF + LightGCN grid sweep, (ii) saves the baseline best-config JSON, (iii) runs the 7-trial PC-LGCN sweep using that backbone, (iv) writes the 3-model decomposition and the PC-LGCN ablation + lambda-sensitivity tables, and (v) runs both canonical panel regressions (PC-LGCN pinned to the `L_PC`-only cell and to the full-method cell).

For a partial GPU re-run (e.g., recomputing only the PC-LGCN sweep against an existing baseline best-config), invoke `run_profile_coherent_grid` from `src.pipeline.tune` programmatically; for analysis-only re-runs, see [Re-running individual analysis steps](#re-running-individual-analysis-steps) above.

Job email notifications go to the addresses listed in the `#SBATCH --mail-user` line. Live job output streams to `outputs/{user}.{jobid}.out` on the cluster filesystem.

### File Transfers

```bash
# Local file -> cluster
scp /path/to/file samuel.sim.2024@origami.smu.edu.sg:~/path/to/destination

# Local folder -> cluster
scp -r /path/to/folder samuel.sim.2024@origami.smu.edu.sg:~/path/to/destination

# Pull artefacts back from cluster -> local for analysis
scp -r samuel.sim.2024@origami.smu.edu.sg:~/SMU-Capstone/outputs ./outputs
```

### Useful Commands

```bash
myinfo                  # Account details, quotas, partition info
myqueue                 # Status of current jobs
myjob <jobid>           # Detailed info on a running/recent job (last 5 min)
mypastjob <days>        # Job history for the past N days (max 30)
```

## References

- Sanz-Cruzado, J., Droukas, N., and McCreadie, R. (2024). *FAR-Trans: An Investment Dataset for Financial Asset Recommendation*. arXiv:2407.08692.
- Sanz-Cruzado, J., McCreadie, R., Droukas, N., Macdonald, C., and Ounis, I. (2022). *On Transaction-Based Metrics as a Proxy for Profitability of Financial Asset Recommendations*. FinRec @ RecSys 2022.
- He, X., Deng, K., Wang, X., Li, Y., Zhang, Y., and Wang, M. (2020). *LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation*. SIGIR 2020.
- Barber, B. M., and Odean, T. (2000). *Trading Is Hazardous to Your Wealth: The Common Stock Investment Performance of Individual Investors*. Journal of Finance 55(2).
- Barber, B. M., and Odean, T. (2008). *All That Glitters: The Effect of Attention and News on the Buying Behavior of Individual and Institutional Investors*. Review of Financial Studies 21(2).
- Kumar, A. (2009). *Who Gambles in the Stock Market?* Journal of Finance 64(4).
- Kim, J., Lee, S., et al. (2025). *Risk-Aware Utility Re-Ranking for Financial Asset Recommendation*. (Working paper / proceedings; see `papers/Risk-Aware Utility Re-Ranking for Financial Asset Recommendation/paper.md`.)
