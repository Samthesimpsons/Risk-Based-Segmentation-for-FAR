# Risk-Based Segmentation for Financial Asset Recommendation: Balancing Return, Accuracy, and Risk Suitability

A study on the FAR-Trans dataset that brings regulatory risk suitability inside financial asset recommendation. It evaluates models on three axes, return (ROI), ranking accuracy (nDCG), and risk suitability (measured by **Profile Coherence at k, PC@k**), summarised by their harmonic-mean **balance**. It then proposes **risk-based segmentation** of LightGCN (one sub-model per declared risk band) with two exclusive interventions: a model-side coherence margin loss and data-side customer regrouping.

## Table of Contents

1. [Paper and Findings](#paper-and-findings)
2. [FAR-Trans Context](#far-trans-context)
3. [Working with this Repository](#working-with-this-repository)
4. [GPU Cluster](#gpu-cluster)

## Paper and Findings

- **Conference writeup**: LaTeX sources live in `thesis/thesis_draft_2/` (the original first draft is retained in `thesis/thesis_draft_1/` for reference). `ijcai26.pdf` is the compiled output and `sections/` contains the per-section sources. Figures are loaded from `thesis/thesis_draft_2/figures/`.
- **Figure export**: `uv run poe figures` renders every findings figure as a single-column PDF into `thesis/thesis_draft_2/figures/`. All renderers live in `src/analysis/findings.py`.

This README is the engineering counterpart: project context, code architecture, reproduction instructions.

## FAR-Trans Context

This section summarises the FAR-Trans paper that forms the dataset and baseline foundation of this project.

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
uv run poe tune --splits-limit 2 --device cpu                                 # baseline pipeline smoke test using cpu and small subset of data
uv run poe stratify --splits-limit 2 --device cpu                             # stratified PC-LightGCN smoke test
uv run poe regroup                                                            # reassign customers to their revealed band -> data/regrouped/
uv run poe stratify-regrouped --splits-limit 2 --device cpu                   # stratified PC-LightGCN on regrouped data, smoke test
uv run poe jlab                                                               # launch Jupyter Lab for notebook exploration
uv run poe figures                                                            # export all findings figures as PDFs into thesis/thesis_draft_2/figures/
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
uv run poe tune         # baseline grid + decomposition + transaction-return and panel regressions
uv run poe stratify     # risk-based segmentation (per-band sub-models, with the coherence margin loss)
uv run poe regroup-stratify  # extension: reassign customers to their revealed band, then rerun the segmentation on data/regrouped/
uv run poe figures      # export all findings figures as single-column PDFs into thesis/thesis_draft_2/figures/
```

After the figures export, rebuild the LaTeX paper to pick up the regenerated figures:

```sh
cd thesis/thesis_draft_2 && latexmk -pdf ijcai26.tex
```

> **Note**: the SLURM batch scripts described in [GPU Cluster](#gpu-cluster) are only relevant if you have access to the SMU GPU cluster. In that case, `sbatch scripts/tune.sh`, `sbatch scripts/stratify.sh`, and `sbatch scripts/regroup.sh` replace the corresponding `poe` tasks (`tune`, `stratify`, and `regroup-stratify`).

## GPU Cluster

The SMU `msc` partition under `studentqos` is the standard target for the grid sweep. SSH via my personal email: `samuel.sim.2024@origami.smu.edu.sg` (GlobalVPN set-up required).

### Submitting the Pipeline

```bash
sbatch scripts/tune.sh         # baseline grid + decomposition + regression studies
sbatch scripts/stratify.sh     # stratified profile-coherent LightGCN
sbatch scripts/regroup.sh      # extension: regroup customers + stratified PC-LightGCN on regrouped data
```

Each script loads the cluster Python and CUDA modules, activates the venv, and invokes the matching `poe` task (`tune`, `stratify`, or `regroup-stratify`).

Job email notifications go to the addresses listed in the `#SBATCH --mail-user` line. Live job output streams to `outputs/{user}.{jobid}.out` on the cluster filesystem.

### Useful Commands

```bash
myinfo                  # Account details, quotas, partition info
myqueue                 # Status of current jobs
myjob <jobid>           # Detailed info on a running/recent job (last 5 min)
mypastjob <days>        # Job history for the past N days (max 30)
```
