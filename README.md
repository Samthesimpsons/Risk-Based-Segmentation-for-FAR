# SMU Capstone: Profile Coherence as a Diagnostic and Design Lens for Financial Asset Recommendation

A measurement framework and a profile-aware extension of LightGCN for the FAR-Trans dataset. The thesis introduces **Profile Coherence at k (PC@k)** as a new evaluation axis, audits how profile-coherent existing FAR baselines actually are, and proposes a minimal **Profile-Coherent LightGCN** that uses regulatory profile signals as both a conditioning input and a learning constraint.

This README is the single source of truth for the project: thesis framing, methodology, code architecture, task roadmap, and expected outputs all live here.

## Table of Contents

1. [Paper Summary: FAR-Trans](#paper-summary-far-trans)
2. [Problem Statement](#problem-statement)
3. [Research Questions](#research-questions)
4. [Dataset Audit Findings](#dataset-audit-findings)
5. [Novelty and Research Contributions](#novelty-and-research-contributions)
6. [Proposed Approach](#proposed-approach)
7. [Source Code Architecture](#source-code-architecture)
8. [Project Roadmap](#project-roadmap)
9. [Expected Outputs](#expected-outputs)
10. [Risks and Mitigations](#risks-and-mitigations)
11. [Working with this Repository](#working-with-this-repository)
12. [GPU Cluster](#gpu-cluster)
13. [References](#references)

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

The dataset audit (`uv run poe eda`, source in `src/analysis/eda_profile_coherence.py`, full numbers in `outputs/eda/profile_coherence/summary.json`) tests whether profile-coherence is a load-bearing signal in FAR-Trans. The headline numbers below all come from the hierarchical asset risk-class mapping (mutual funds via `assetSubCategory`, bonds via subcategory, stocks via 252-day annualised volatility quartile).

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

Under the default tolerance (`d <= 1`) **81.4% of transactions are profile-coherent**, leaving **18.6% (42,417 transactions) that violate the user's declared MiFID band by 2 or more steps**. Mean discordance is 0.87 bands. See `outputs/eda/profile_coherence/transaction_discordance_distribution.png`.

### Finding 2: Self-discordance is a customer-level trait, not transaction-level noise

The per-customer fraction of discordant transactions is bimodal, not normal:

- **64.4% of customers are fully profile-coherent** (every transaction within `d <= 1`).
- **17.2% of customers are fully discordant** (every transaction at `d >= 2`).
- 20.3% have a majority-discordant trading record.
- The middle of the distribution is sparse.

This U-shape (`outputs/eda/profile_coherence/customer_self_discordance_histogram.png`) is the strongest empirical motivation for the thesis: discordance is *systematic at the customer level*, not random noise. A recommender that ignores it inherits the same bias. Behavioural-finance literature (Barber and Odean 2000 / 2008; Kumar 2009) predicts exactly this concentration.

### Finding 3: Extreme-band customers reach toward the centre

Decomposing transaction discordance by the customer's declared MiFID band reveals a regression-toward-the-mean pattern:

| Declared MiFID Band | Transactions | Coherent share (`d <= 1`) |
|---|---|---|
| Conservative | 14,193 | **45.1%** |
| Income | 73,468 | 90.9% |
| Balanced | 99,901 | 89.9% |
| Aggressive | 40,679 | **56.2%** |

Mid-band customers (Income, Balanced) are roughly 90% profile-coherent. The two extreme bands are dramatically less coherent: more than half of Conservative customers' Buy transactions land at `d >= 2` (chasing risk), and a similar fraction of Aggressive customers' purchases land in safer assets than their profile permits (loss-aversion or yield-chasing). See `outputs/eda/profile_coherence/discordance_by_risk_level.png`.

### Finding 4: Discordance is stable across regimes

Mean discordance per calendar year stays in the 0.83-0.98 band across 2018-2022 (`outputs/eda/profile_coherence/discordance_by_year.png`). The COVID-19 crash and recovery do not visibly shift the pattern. This rules out a "panic trading drove the discordance" explanation and supports the customer-level-trait reading from Finding 2.

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
    profile_coherence/
        risk_classification.py  # Asset to MiFID 4-band mapping (hierarchical + volatility quartile)
        customer_profile.py     # Customer profile lookup builder (riskLevel + segment + capacity)
        discordance.py          # Per-transaction and per-recommendation discordance scoring
    evaluation/
        metrics.py              # nDCG@k, ROI@k, Recall@k, PC@k
    models/
        protocol.py             # Recommender Protocol (structural type only)
        random_forest.py        # Price-based RF regressor (paper baseline)
        light_gcn.py            # LightGCN collaborative filtering (paper baseline)
        profile_coherent_lgcn.py# Profile-Coherent LightGCN (method-contribution model)
    pipeline/
        preprocessing.py                # Generate splits, save to disk
        baseline_evaluation.py          # Ray-driven baseline grid sweep (6 RF + 8 LightGCN trials), each trial = full 69-split eval
        profile_coherent_evaluation.py  # Ray-driven PC-LGCN ablation + lambda-sensitivity sweep at the winning LightGCN backbone (7 trials)
    analysis/
        eda_profile_coherence.py            # Dataset audit (figures + summary.json)
        baseline_decomposition.py           # Post-evaluation: best-trial selection, decomposition, scatter
        profile_coherent_decomposition.py   # PC-LGCN ablation + lambda-sensitivity tables and per-cell ROI breakdown
        panel_regression.py                 # OLS of PC@k on declared band x model with cluster-robust SEs
```

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

`src/profile_coherence/` is the new package that owns the measurement framework:

- `risk_classification.py`: builds `dict[asset_id, ordinal_band]` from `asset_information.csv` and `close_prices.csv`. The current default uses a single (full-period) volatility-quartile mapping for stocks; per-time-point refitting is a planned sensitivity check.
- `customer_profile.py`: builds `dict[customer_id, CustomerProfile]` from `customer_information.csv`. Maps `Predicted_*` risk levels onto the same ordinal scale as declared ones, with a flag.
- `discordance.py`: pairwise `d(b_u, b_i)`, `is_profile_coherent` predicate, and a transaction-level annotator used by the EDA notebook and the panel regression.

### Pipeline Orchestration

```sh
uv run poe preprocess                  # Step 0: generate splits to data/splits/
uv run poe eda                         # Step 1: dataset audit -> outputs/eda/profile_coherence/
uv run poe evaluate-baselines          # Step 2: baseline Ray grid sweep + decomposition
uv run poe evaluate-profile-coherent   # Step 3: PC-LGCN ablation + lambda-sensitivity sweep + decomposition
uv run poe panel-regression            # Step 4: cluster-robust OLS of PC@k on declared band x model
```

Each pipeline step is end-to-end: it runs the Ray grid sweep, dumps per-trial parquets, picks the best trial per primary metric, runs its decomposition, and saves headline tables. The decomposition steps run automatically at the end of each evaluation. To re-run only a decomposition without retraining, call it directly:

```sh
uv run python -m src.analysis.baseline_decomposition --run-timestamp <YYYYMMDD_HHMMSS>
uv run python -m src.analysis.profile_coherent_decomposition --run-timestamp <YYYYMMDD_HHMMSS>
uv run python -m src.analysis.panel_regression --run-timestamp <YYYYMMDD_HHMMSS>
```

### Baseline Grid Sweep

`src/pipeline/baseline_evaluation.py` runs a Ray-driven grid sweep across RF (6 trials) and LightGCN (8 trials). Each trial is a **full 69-split evaluation**, so the per-trial summary is directly comparable to the FAR-Trans paper's Table 2.

| Model | Grid axes | Trials | Primary metric |
|---|---|---|---|
| Random Forest | `number_of_estimators in {20, 50, 100}` x `max_depth in {None, 15}`; other axes pinned to paper | 6 | ROI@10 |
| LightGCN | `embedding_dimension in {64, 128}` x `number_of_layers in {2, 3}` x `learning_rate in {1e-2, 1e-3}`; other axes pinned to paper | 8 | nDCG@10 |

**Paper defaults are explicitly included** in both grids (`n_estimators=20, max_depth=None` for RF; `emb_dim=64, layers=3, lr=1e-2` for LightGCN), so benchmark replication is always one of the trial points. The LightGCN grid is kept tight (8 trials) because the Profile-Coherent LightGCN pipeline reuses its winning backbone rather than re-sweeping every axis: the ablation runs at the chosen backbone, not on the full cartesian product.

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

The cluster job (`scripts/evaluate-baselines.sh`) requests `--gres=gpu:1` on an L40s. Ray distributes:

- RF: 3 concurrent CPU trials (1 CPU each), so the 6 trials run as two waves of 3.
- LightGCN: 4 concurrent GPU trials with `gpu = 1 / max_concurrent_trials = 0.25` fractional sharing on the single L40s, so the 8 trials run as two waves of 4.

#### Validation/Evaluation Window Overlap (Known Caveat)

The legacy validation-split tuning had a known overlap issue between validation and early evaluation splits. The new design eliminates this entirely: every trial is a full 69-split evaluation, so there is no separate validation set whose splits could overlap with the benchmark.

### Profile-Coherent LightGCN Sweep

`src/pipeline/profile_coherent_evaluation.py` runs a focused 7-trial sweep on top of the winning LightGCN backbone. The backbone hyperparameters are read from `outputs/configs/<latest>/best_hyperparameters.json` (selected by mean nDCG@10 in the baseline run). All seven trials use the same backbone; only the profile-conditioning components vary.

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

Resource model: the SLURM script (`scripts/evaluate-profile-coherent.sh`) requests the same 1xL40s envelope as the baseline run with a 24h wall-clock cap. Ray distributes 4 concurrent GPU trials at 0.25 GPU each, so the 7 trials run as roughly two waves.

### Panel Regression

`src/analysis/panel_regression.py` tests RQ2/RQ3 jointly: does the per-band coherence gap close or widen between models? Each model's best-trial recommendations parquet is reduced to one row per (customer, split) carrying the customer's coherent share for that split. The model fits

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

## Project Roadmap

Tasks for the thesis. Items marked `[x]` are done; `[ ]` is pending.

- [x] `src/profile_coherence/risk_classification.py`: hierarchical asset to MiFID 4-band mapping (mutual-fund subcategory, bond subcategory, stock volatility-quartile fallback).
- [x] `src/profile_coherence/discordance.py`: pairwise discordance, coherence predicate, transaction-level annotator.
- [x] `src/profile_coherence/customer_profile.py`: parses declared / predicted / `Not_Available` MiFID risk levels.
- [x] `src/analysis/eda_profile_coherence.py`: dataset audit (asset and customer band distributions, transaction-level discordance, per-customer self-discordance, segment and year breakdowns, hierarchy vs pure-volatility sensitivity). See [Dataset Audit Findings](#dataset-audit-findings).
- [x] `compute_profile_coherence_at_k` in `src/evaluation/metrics.py`, wired into `EvaluationResult` and `evaluate_model_on_split`.
- [x] `src/pipeline/baseline_evaluation.py`: Ray-driven grid where each trial is a full 69-split eval. Paper defaults are guaranteed grid points (RF: `n_estimators in {20, 50, 100}` x `max_depth in {None, 15}`, ROI@10 primary, 6 trials; LightGCN: `embedding_dimension in {64, 128}` x `number_of_layers in {2, 3}` x `learning_rate in {1e-2, 1e-3}`, nDCG@10 primary, 8 trials).
- [x] `src/analysis/baseline_decomposition.py`: best-trial selection, profile-coherent vs profile-discordant ROI breakdown, scatter plots, summary JSON. Runs automatically as the final step of `evaluate-baselines`.
- [x] `ProfileCoherentLightGCNBaseline` implemented (`src/models/profile_coherent_lgcn.py`).
- [x] `src/pipeline/profile_coherent_evaluation.py`: reads the winning LightGCN backbone from the baseline best-config JSON and runs a 4-cell ablation (`profile_embedding_enabled` x `profile_coherence_enabled`) plus a 3-point lambda sensitivity sweep on top. Each trial = full 69-split evaluation. Decomposition runs automatically.
- [x] `src/analysis/profile_coherent_decomposition.py`: ablation + lambda-sensitivity tables with per-cell coherent/discordant ROI breakdown. Runs automatically as the final step of `evaluate-profile-coherent`.
- [x] `src/analysis/panel_regression.py`: cluster-robust OLS of PC@10 on declared band x model with predicted-PC forest figure. Available as `uv run poe panel-regression`.
- [ ] Submit `scripts/evaluate-baselines.sh` to the SMU L40s GPU. One sbatch produces per-trial artefacts, the per-trial roll-up CSV, the best-config JSON, the main results table, the decomposition table, and the scatter plots.
- [ ] Update this README's "Dataset Audit Findings" section with the baseline-audit table once the GPU job completes.
- [ ] Submit `scripts/evaluate-profile-coherent.sh` to the SMU L40s GPU once baselines are in. One sbatch produces the PC-LGCN per-trial artefacts, the ablation table, the lambda-sensitivity table, and the updated headline main-results table.
- [ ] Run `uv run poe panel-regression` after both pipelines have completed to produce the per-band coherence-gap table and forest figure.
- [ ] Main results table (3 rows: RF, LightGCN, Profile-Coherent LightGCN x 4 metrics).
- [ ] Ablation table (4 rows: 2x2 of profile-embedding x L_PC x 4 metrics).
- [ ] Sensitivity table: ordinal vs squared discordance, profile-feature subsets, volatility-window length, declared-only vs declared+predicted profiles.
- [ ] Final writeup polish.
- [ ] Defense slides.

## Expected Outputs

### Tables

1. **Table 1** (main results): `Model x {nDCG@10, ROI@10, Recall@10, PC@10}` for RF, LightGCN, Profile-Coherent LightGCN over 69 splits, mean +/- standard deviation across splits.
2. **Table 2** (ablation): `LightGCN-variant x metrics` with 2x2 toggle of profile-embedding and L_PC.
3. **Table 3** (sensitivity): ordinal vs squared `d`, profile-feature subsets, volatility window, declared-only vs declared+predicted MiFID.
4. **Table 4** (panel regression): coefficient on `d`, with controls and fixed effects, cluster-robust SE.

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
| Volatility-quartile risk-class binning is sensitive to window length | Medium | Could weaken claim | Sensitivity row in Table 3, fix window before looking at headline numbers to avoid p-hacking |
| Discordance-vs-return regression coefficient is statistically insignificant | Medium | Weakens RQ2 | Report direction of effect even if non-significant; the audit (RQ1) and method (RQ4) stand independently |
| Profile-Coherent LightGCN has worse PC@10 than expected (the loss may be too soft) | Low-Medium | Weakens RQ4 | The 2x2 ablation will isolate which component is at fault; can fall back to a re-ranker variant for the actionability proof |
| Mutual-fund subcategory mapping is too coarse (e.g., "Other" / "Structured" categories) | Medium | Adds noise to risk-class | Volatility tiebreaker for "Other"/"Structured" rows; report % of universe affected |
| `Predicted_*` risk levels behave differently from declared | Medium | Could bias PC@10 | Sensitivity row pooling vs separating; default keeps both pooled |
| Time available is not enough for the model contribution | Medium | Truncates RQ4 | Audit (RQ1-3) + metric is publishable on its own; method is the cherry on top, not the foundation |

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

The thesis pipeline is five commands. Each evaluation step is a single end-to-end driver: it runs the Ray grid sweep, dumps per-trial metrics + per-recommendation parquet, picks the best trial per primary metric, runs the decomposition, prints the headline tables, and saves the scatter plots, all in one invocation.

```sh
uv run poe preprocess                  # one-off: generate 69 evaluation splits
uv run poe eda                         # dataset audit -> outputs/eda/profile_coherence/
uv run poe evaluate-baselines          # baseline grid sweep + decomposition
uv run poe evaluate-profile-coherent   # PC-LGCN ablation + lambda sweep + decomposition (depends on baseline best-config JSON)
uv run poe panel-regression            # cluster-robust OLS across all completed runs
```

The recommended way to run both `evaluate-*` tasks is on the GPU cluster (see below); locally on CPU they are workable for smoke tests with `--splits-limit` and `--max-concurrent-trials`.

### Common Tasks

```sh
uv run poe lint                                                  # ruff
uv run poe type                                                  # ty
uv run poe test                                                  # pytest
uv run poe format                                                # ruff format

uv run poe evaluate-baselines --splits-limit 2 --max-concurrent-trials 1          # baseline smoke test
uv run poe evaluate-profile-coherent --splits-limit 2 --max-concurrent-trials 1   # PC-LGCN smoke test
uv run python -m src.analysis.baseline_decomposition --run-timestamp <ts>            # re-run baseline decomposition only
uv run python -m src.analysis.profile_coherent_decomposition --run-timestamp <ts>    # re-run PC-LGCN ablation tables only
uv run python -m src.analysis.panel_regression --run-timestamp <ts>                  # re-run panel regression only
```

### Git Hooks

[Lefthook](https://github.com/evilmartians/lefthook) manages the hooks:

- **Pre-commit**: lint, format, typecheck.
- **Post-commit / post-checkout**: rebuild the graphify knowledge graph for changed code files.

## GPU Cluster

The SMU `msc` partition (1 L40s GPU, 4 CPUs, 32 GB RAM, 5-day max job time, `studentqos`) is the standard target for the grid sweep. SSH via `samuel.sim.2024@origami.smu.edu.sg` (GlobalVPN required).

### Submitting the Pipeline

Two sbatch jobs, run in order:

```bash
sbatch scripts/evaluate-baselines.sh           # waits ~24-34h on 1xL40s; produces baseline outputs + best_hyperparameters.json
sbatch scripts/evaluate-profile-coherent.sh    # depends on the baseline best-config JSON; ~12-18h on 1xL40s
```

`scripts/evaluate-baselines.sh` is the baseline pipeline driver. It loads the cluster Python and CUDA modules, activates the venv, and runs `uv run poe evaluate-baselines --device cuda`. That command produces every baseline artefact under `outputs/` (see "Pipeline Orchestration" above for the full list), including the headline tables and scatter plots from the baseline decomposition step.

`scripts/evaluate-profile-coherent.sh` is the PC-LGCN pipeline driver. It reads the latest baseline best-config JSON, runs the 4-cell ablation plus 3-point lambda sensitivity (7 trials), updates the main-results table to include a Profile-Coherent LightGCN row, and writes the ablation + sensitivity tables under `outputs/analysis/profile_coherent_decomposition/`.

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
