# SMU Capstone: Profile Coherence as a Diagnostic and Design Lens for Financial Asset Recommendation

A measurement framework for profile-aware financial asset recommendation on the FAR-Trans dataset. The thesis introduces **Profile Coherence at k (PC@k)** as a new evaluation axis, audits how profile-coherent existing FAR baselines (Random Forest, LightGCN) actually are (RQ1-RQ3), and proposes a stratified profile-coherent LightGCN extension as a prescriptive intervention (RQ4). This README is the single source of truth for the project.

## Table of Contents

1. [Thesis](#thesis) (full writeup in [thesis.md](thesis.md))
2. [Paper Summary: FAR-Trans](#paper-summary-far-trans)
3. [Problem Statement](#problem-statement)
4. [Research Questions](#research-questions)
5. [Working with this Repository](#working-with-this-repository)
6. [GPU Cluster](#gpu-cluster)
7. [References](#references)

## Thesis

The full writeup (Abstract, Introduction, Methodology, Findings for RQ1/RQ2/RQ3, Discussion, Conclusion) lives in [thesis.md](thesis.md). All findings, audit numbers, regression tables, and the per-band lift profile are reported there. This README is the engineering counterpart: project context, code architecture, reproduction instructions.

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

**MiFID II** (Markets in Financial Instruments Directive II) is the EU regulation in force since January 2018 that requires investment firms to verify a product is suitable for a client's risk tolerance, knowledge, financial situation, and objectives before recommending it. Risk profiles in FAR-Trans come from a real MiFID II 25-question questionnaire administered by the bank; for the subset of customers who never completed it, the band is regression-imputed from correlated demographic features like estimated salary.

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

The headline finding is the **negative correlation between nDCG@10 and ROI@10**: methods that win one objective lose the other.

A second observation, which motivates this thesis: none of these baselines take the customer's MiFID risk profile or the suitability of the assets they buy as a signal. Price-based models fit returns, transaction-based models fit buy history, hybrids combine the two, but none read `riskLevel` or measure how aligned a recommendation is with the customer's profile. Any mismatch between profile and actual buying behaviour passes straight through into the recommendations, and standard metrics (nDCG, ROI) don't penalise it, so the model has no reason to correct it.

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

The FAR-Trans benchmark exposes a sharp tradeoff between two evaluation axes. Transaction-based methods (LightGCN, Matrix Factorisation) achieve high nDCG@10 by fitting customer buy history but earn near-zero ROI@10. Price-based methods (Random Forest, LightGBM) achieve the highest ROI@10 by predicting asset profitability but score near-random on nDCG@10. No baseline simultaneously wins both axes, and the field has typically framed this as a fundamental "preference versus profit" tension to be balanced via hybrid objectives.

FAR-Trans contains a third signal that every existing baseline ignores: the customer's MiFID II risk profile. None of the FAR-Trans baselines read this signal as input, optimise against it during training, or are evaluated on whether their top-k satisfies it.

The consequence is a measurement gap and a method gap. The measurement gap: nDCG and ROI cannot tell whether a recommendation is suitable, so a model that produces failing recommendations can still score well on the established benchmark. Any mismatch between a customer's profile and their actual buying behaviour passes straight through into the model's output.

This thesis treats the gap as the load-bearing problem. It introduces **Profile Coherence at k (PC@k)** as an evaluation axis and uses it both to audit the FAR-Trans transaction record and to relocate existing baselines on a third axis. The reframing positions FAR not as preference-versus-profit, but as profit-within-the-suitable-universe.

## Research Questions

| # | Question | Test method |
|---|---|---|
| **RQ1** *(Diagnostic)* | What is the distribution of profile-discordance in the FAR-Trans transaction record, broken down by `customerType`, `riskLevel`, and market regime? | Dataset audit (`uv run poe eda`, source in `src/pipeline/eda.py`, full numerics in `outputs/eda/summary.json`); written up in [thesis.md, section 3.1](thesis.md#31-rq1-profile-discordance-is-prevalent-and-structural). |
| **RQ2** *(Quasi-causal)* | Do profile-discordant transactions earn lower realised 6-month return than profile-coherent ones? | Transaction-level OLS on the FAR-Trans Buy record with asset volatility, customer segment, and year as controls and standard errors clustered on `customerID`; written up in [thesis.md, section 3.2](thesis.md#32-rq2-profile-coherent-transactions-earn-higher-realised-return). |
| **RQ3** *(Audit)* | Where do the FAR-Trans baselines (Random Forest, LightGCN) sit on the Profile Coherence (PC@10) axis relative to the band-conditional random baseline π (the per-band PC@10 a uniformly-random recommender would achieve, computed as the share of the asset menu within ±1 band of the customer's risk band)? | Baseline grid sweep across 69 time-based splits + a band-conditional model panel regression; written up in [thesis.md, section 3.3](thesis.md#33-rq3-both-far-trans-baselines-under-serve-declared-band-coherence). |
| **RQ4** *(Method)* | Can a stratified, profile-coherent LightGCN extension (one sub-model per MiFID risk band, trained with a profile-coherent margin loss) improve PC@10 over the global LightGCN baseline, and at what cost to nDCG@10, Recall@10, and ROI@10? Does the gain vary by declared risk segment? | Two stratified configurations (λ=0 ablation, λ=1 treatment) evaluated on the same 69 evaluation splits as RQ1-RQ3 via `uv run poe stratify`; comparison against the RQ1 LightGCN baseline reuses the existing `outputs/results/evaluation/light_gcn/` artefacts. Source in `src/pipeline/stratified_training.py` and `src/models/profile_coherent_light_gcn.py`. |

## Working with this Repository

### Prerequisites

- **[uv](https://docs.astral.sh/uv/getting-started/installation/)**: package and dependency manager

### Setup

```sh
uv sync                               # Install dependencies
uv run poe setup                      # Install lefthook git hooks
source .venv/bin/activate             # Activate the virtual environment
pip install graphifyy                 # optional: knowledge-graph integration
graphify claude install               # optional: Claude Code integration
```

### Common Tasks

```sh
uv run poe preprocess                                                         # generate temporal evaluation splits to data/splits/
uv run poe eda                                                                # dataset audit -> outputs/eda/
uv run poe tune --splits-limit 2 --device cpu                                 # RQ1-RQ3 smoke test using cpu and small subset of data
uv run poe stratify --splits-limit 2 --device cpu                             # RQ4 stratified PC-LightGCN smoke test
uv run poe lint                                                               # ruff linting
uv run poe type                                                               # ty type checks
uv run poe format                                                             # ruff format
```

### Outputs

A full pipeline run (`preprocess` + `eda` + `tune`) produces:

- `outputs/eda/`: dataset audit (`summary.json` plus seven figures).
- `outputs/results/evaluation/{model}/{timestamp}/{trial_id}/`:
    - `per_split_metrics.csv`: per-split scalar metrics for one trial.
    - `recommendations.parquet`: flat per-recommendation rows with `monthly_return` and `is_relevant` precomputed.
- `outputs/results/tuning/{model}/{timestamp}.csv`: per-trial roll-up (averaged nDCG@10, ROI@10, Recall@10, PC@10, and PC-lift@10).
- `outputs/configs/{timestamp}/best_hyperparameters.json`: best trial per model, by primary metric.
- `outputs/analysis/baseline_decomposition/{timestamp}/`: `main_results.csv`, `decomposition.csv`, scatter plots, `summary.json`.
- `outputs/analysis/transaction_return_regression/{timestamp}/` (RQ2): `coefficients.csv`, `panel.csv`, `regression_summary.txt`, `summary.json`.
- `outputs/analysis/panel_regression/{timestamp}/` (RQ3): `coefficients.csv`, `predicted_pc_by_band_model.csv`, `forest_predicted_pc.png`, `panel.csv`, `regression_summary.txt`.

A separate RQ4 run (`stratify`) produces, alongside the same `evaluation/{model}/{timestamp}/{trial_id}/` and `tuning/{model}/{timestamp}.csv` shape:

- `outputs/results/evaluation/pc_lgcn/{timestamp}/{trial_id}/`: per-split metrics and recommendations for the two stratified configurations (`stratified_lambda_0.0`, `stratified_lambda_1.0`).
- `outputs/results/tuning/pc_lgcn/{timestamp}.csv`: per-configuration roll-up of averaged metrics.

### Git Hooks

[Lefthook](https://github.com/evilmartians/lefthook) manages the hooks:

- **Pre-commit**: lint, format, typecheck.
- **Post-commit / post-checkout**: rebuild the graphify knowledge graph for changed code files.

## GPU Cluster

The SMU `msc` partition under `studentqos` is the standard target for the grid sweep. SSH via my personal email: `samuel.sim.2024@origami.smu.edu.sg` (GlobalVPN set-up required).

### Resource Model

The cluster job (`scripts/tune.sh`) requests 1 L40s GPU, 4 CPUs, 16 GB RAM, with a 2-day wall-clock cap. Each `GridSpec` in `src/config/registry.py` declares its own `max_concurrent_trials`, currently set to 1 for every model so each trial runs serially:

- Random Forest: 12 trials run sequentially. Each trial saturates all 4 CPU cores via `RandomForestRegressor(n_jobs=-1)`, so the work is parallelised inside the trial rather than across trials. This was the fix for the OOM at 16 GB: 4 concurrent forks each materialised a private copy of the 69-split `EvaluationContext`, totalling roughly 5 GB per worker, which the cgroup killed.
- LightGCN: 8 trials run sequentially with `gpu=1.0` per trial. Same memory rationale.

### Validation/Evaluation Window Overlap (Known Caveat)

The legacy validation-split tuning had a known overlap issue between validation and early evaluation splits. The new design eliminates this entirely: every trial is a full 69-split evaluation, so there is no separate validation set whose splits could overlap with the benchmark.

### Submitting the Pipeline

```bash
sbatch scripts/tune.sh         # RQ1-RQ3: baseline grid + decomposition + RQ2 + RQ3
sbatch scripts/stratify.sh     # RQ4: stratified profile-coherent LightGCN
```

Both scripts load the cluster Python and CUDA modules, activate the venv, and invoke the matching `poe` task (`tune` or `stratify`).

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
