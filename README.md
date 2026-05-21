# Profile Coherence at k: A Suitability Axis for Financial Asset Recommendation

A measurement framework for profile-aware financial asset recommendation on the FAR-Trans dataset. The thesis introduces **Profile Coherence at k (PC@k)** as a new evaluation axis, audits how profile-coherent existing FAR baselines (Random Forest, LightGCN) actually are, and proposes a stratified profile-coherent LightGCN extension as a prescriptive intervention.

## Table of Contents

1. [Thesis](#thesis) (full writeup in [thesis.md](thesis.md))
2. [Paper Summary: FAR-Trans](#paper-summary-far-trans)
3. [Working with this Repository](#working-with-this-repository)
4. [GPU Cluster](#gpu-cluster)
5. [References](#references)

## Thesis

The full writeup lives in [thesis.md](thesis.md). This README is the engineering counterpart: project context, code architecture, reproduction instructions.

## Paper Context: FAR-Trans

This section summarises the FAR-Trans paper that forms the foundation of this project.

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

**[FAR-Trans](https://doi.org/10.5525/gla.researchdata.1658) (Sanz-Cruzado et al., 2024) fills this gap**: the first public dataset for FAR that contains both real asset pricing information and real retail investor transactions, collected from a large European financial institution (the National Bank of Greece) and covering January 2018 to November 2022. The paper also provides a **benchmark comparison of 11 FAR algorithms** as baselines for future research.

### Dataset

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

These three categorical signals form the **profile vector** that our profile coherence will rely on.

#### Cleaning and Pre-processing

All cleaning has been done by the FAR-Trans authors. See the original [dataset documentation](https://researchdata.gla.ac.uk/1658/) for the full schedule of price normalisation, transaction synthesis, and stock-split adjustment.

### Existing FAR Methods

The FAR-Trans paper benchmarks three families:

1. **Price-based** (Linear Regression, Random Forest, LightGBM): predict asset profitability from technical indicators. Best at ROI@10 (Random Forest reaches 0.0259 monthly), worst at nDCG@10 (near-random preference accuracy).
2. **Transaction-based** (Popularity, Matrix Factorisation, LightGCN, ARM, UB-kNN): predict the next purchase from interaction history. Best at nDCG@10 (LightGCN reaches 0.3404), worst at ROI@10 (near-zero realised return).
3. **Hybrid** (Hybrid-nDCG, Hybrid-regression): two-stage pipelines combining the above. Neither dominates either axis.

#### Benchmark Results

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

The headline finding is the **negative correlation between nDCG@10 and ROI@10**: methods that win one objective lose the other.

A second observation, which motivates this thesis: none of these baselines take the customer's MiFID risk profile or the suitability of the assets they buy as a signal.

Price-based models fit returns, transaction-based models fit buy history, hybrids combine the two, but none read `riskLevel` or measure how aligned a recommendation is with the customer's profile. Any mismatch between profile and actual buying behaviour passes straight through into the recommendations, and standard metrics (nDCG, ROI) don't penalise it, so the model has no reason to correct it.

## Working with this Repository

### Prerequisites

- **[uv](https://docs.astral.sh/uv/getting-started/installation/)**: package and dependency manager

### Setup

```sh
uv sync                               # Install dependencies
uv run poe setup                      # Install lefthook git hooks that runs the precommit and postcommit checks
source .venv/bin/activate             # Activate the virtual environment
```

Here is a summary of what the lefthook git hooks does:
- **Pre-commit**: lint, format, typecheck.

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

### Reproducing the Work

Run the following `poe` tasks in order from the project root:

```sh
uv run poe setup        # install lefthook git hooks (precommit/postcommit checks)
uv run poe preprocess   # generate temporal evaluation splits to data/splits/
uv run poe eda          # dataset audit -> outputs/eda/
uv run poe tune         # RQ1-RQ3: baseline grid + decomposition + RQ2 + RQ3
uv run poe stratify     # RQ4: stratified profile-coherent LightGCN
```

> **Note**: the SLURM batch scripts described in [GPU Cluster](#gpu-cluster) are only relevant if you have access to the SMU GPU cluster. In that case, `sbatch scripts/tune.sh` and `sbatch scripts/stratify.sh` replace the last two `poe` tasks (`tune` and `stratify`).

## GPU Cluster

The SMU `msc` partition under `studentqos` is the standard target for the grid sweep. SSH via my personal email: `samuel.sim.2024@origami.smu.edu.sg` (GlobalVPN set-up required).

### Submitting the Pipeline

```bash
sbatch scripts/tune.sh         # RQ1-RQ3: baseline grid + decomposition + Regression Studies
sbatch scripts/stratify.sh     # RQ4: stratified profile-coherent LightGCN
```

Both scripts load the cluster Python and CUDA modules, activate the venv, and invoke the matching `poe` task (`tune` or `stratify`).

Job email notifications go to the addresses listed in the `#SBATCH --mail-user` line. Live job output streams to `outputs/{user}.{jobid}.out` on the cluster filesystem.

### Useful Commands

```bash
myinfo                  # Account details, quotas, partition info
myqueue                 # Status of current jobs
myjob <jobid>           # Detailed info on a running/recent job (last 5 min)
mypastjob <days>        # Job history for the past N days (max 30)
```

## References

- Sanz-Cruzado, J., Droukas, N., and McCreadie, R. (2024). *FAR-Trans: An Investment Dataset for Financial Asset Recommendation*. IJCAI Workshop on Recommender Systems in Finance. See `papers/FAR-Trans An Investment Dataset for Financial Asset Recommendation/paper.md`.
- Sanz-Cruzado, J., McCreadie, R., Droukas, N., Macdonald, C., and Ounis, I. (2026). *Investors Are (Not) Always Right: A Comparison of Transaction-Based and Profitability-Based Metrics for Financial Asset Recommendations*. ACM Transactions on Information Systems 44(2), Article 51. See `papers/Investors Are (Not) Always Right/paper.md`.
- Ghiye, A., Barreau, B., Carlier, L., and Vazirgiannis, M. *Rolling Forward: Enhancing LightGCN with Causal Graph Convolution for Credit Bond Recommendation*. See `papers/Causal Graph Convolution for Credit Bond Recommendation/paper.md`.
- Sakurai, K., Ogawa, T., Haseyama, M., Anan, A., and Nakagawa, K. *Risk-Aware Utility Re-Ranking for Financial Asset Recommendation*. See `papers/Risk-Aware Utility Re-Ranking for Financial Asset Recommendation/paper.md`.
