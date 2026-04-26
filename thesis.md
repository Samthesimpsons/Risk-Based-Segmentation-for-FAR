# Thesis: Profile Coherence as a Diagnostic and Design Lens for Financial Asset Recommendation

> **Note.** The numerics, tables, and figures below come from prior runs and will be regenerated when the pipeline is rerun fresh on the cluster. Methodology, ablation design, and panel-regression specification are stable. Project context, code architecture, and reproduction instructions live in [README.md](README.md).

## Table of Contents

1. [Abstract](#abstract)
2. [Introduction](#1-introduction)
3. [Methodology](#2-methodology)
4. [Findings](#3-findings)
5. [Ablation Studies](#4-ablation-studies)
6. [Discussion](#5-discussion)
7. [Conclusion](#6-conclusion)

## Abstract

The Financial Asset Recommendation (FAR) literature optimises ranking quality (nDCG, Recall) and realised return (ROI) without exposing whether recommendations are profile-aligned in the regulatory sense required by MiFID II suitability. This paper introduces **Profile Coherence at k (PC@k)**, a metric that measures the share of a top-k list whose asset risk class lies within ordinal tolerance of the customer's declared MiFID II risk band, together with a band-conditional random-baseline normalisation (PC-lift@k) that makes the metric scale-invariant across customer segments. Using the FAR-Trans dataset (806 assets, 29,090 customers, 69 monthly temporal splits), we audit two FAR baselines through this lens and find that neither is risk-aware: Random Forest emits a single global ranking per split (~17 unique top-10s across all customers), and LightGCN's per-customer-band asset-band composition is nearly flat (Aggressive customers receive *less* Aggressive content than Conservative customers do). Both models score below the random baseline for Conservative customers; Random Forest scores below random for Conservative and Income, the two largest segments (60.9% of customers with a usable risk band). We propose **Profile-Coherent LightGCN (PC-LGCN)**, which adds a per-customer profile embedding and a multiplicative discordance regulariser `L_PC` to the standard BPR loss. A seven-trial ablation isolates a component-wise mechanism: `L_PC` alone is sufficient to improve PC@k by 10.1 percentage points and reverses the PC-vs-ROI Pearson correlation across splits from -0.43 to +0.64 via centre-band compression; the profile embedding alone produces no measurable change; both components together at λ=1.0 produce a genuine `(customer-band, asset-band)` diagonal where Aggressive customers receive 44.7% Aggressive content versus Conservative's 3.8% (a 12-fold ratio), eliminate all 47 chronically-discordant customers in the Aggressive band, and produce the only configuration in the study that exceeds the random baseline for all four bands. The PC-versus-quality trade-off is therefore not a single-number trade-off but a component-wise choice between metric efficiency and structural personalisation.

## 1. Introduction

Financial asset recommendation sits at the intersection of personalised ranking and regulatory obligation. The European Markets in Financial Instruments Directive (MiFID II, 2014) requires investment firms to assess suitability before recommending a financial product. Suitability means alignment between the product's risk profile and the customer's declared risk tolerance, investment horizon, and capacity to absorb losses. A recommender that maximises predicted return without conditioning on the customer's risk band can be both an excellent ranker and a regulatory liability.

The FAR-Trans dataset (Sanz-Cruzado et al., 2024) is the first open benchmark for FAR research that includes both customer profile metadata and a multi-year transaction history. It records 388,049 buy and sell transactions across 806 assets and 29,090 retail customers between 2018 and 2025, alongside MiFID II declared risk bands for each customer and asset metadata sufficient to derive each asset's risk class through a hierarchical mapping. The accompanying paper benchmarks LightGCN, Random Forest, and several other recommenders on nDCG@10, ROI@10, and Recall@10. None of those metrics measures profile alignment, and none of the published baselines explicitly conditions on the declared risk band.

This work introduces a profile-alignment lens for the FAR pipeline at three levels: a metric, an audit of existing baselines through that metric, and a method that injects profile information as both a conditioning input and a soft training constraint. The research questions we answer are:

- **RQ1**: How prevalent is profile-discordance in observed FAR-Trans buy transactions, and is it a customer-level trait or transaction-level noise?
- **RQ2**: Is profile-discordance penalised by realised returns in this market?
- **RQ3**: Do existing FAR baselines under-serve declared-band coherence, and where does the failure concentrate?
- **RQ4**: Can a minimal profile-coherent extension of LightGCN improve PC@k without sacrificing ranking quality, and through which structural mechanism?

Our contributions are: (i) a new metric, PC@k, with a band-conditional random-baseline normalisation; (ii) an audit showing both FAR-Trans baselines are functionally risk-blind, in two structurally distinct ways; (iii) a profile-aware LightGCN extension (PC-LGCN) whose 2x2 component ablation cleanly attributes function to mechanism; (iv) a three-model panel regression that quantifies per-band coherence gaps with cluster-robust standard errors and reveals the within-band heterogeneity invisible to aggregate metrics.

## 2. Methodology

### 2.1 Profile Coherence at k

Let `b_u in {0, 1, 2, 3}` be customer `u`'s declared MiFID II risk band, ordinally encoded as Conservative (0), Income (1), Balanced (2), Aggressive (3). Each asset `i` is mapped to a risk band `b_i` via a hierarchical scheme that prefers explicit subcategory information (mutual-fund subcategory or bond subcategory) and falls back to volatility quartiles over a 252-day window for stocks. Pairwise discordance is the absolute ordinal distance:

```
d(u, i) = |b_u - b_i| in {0, 1, 2, 3}
```

A recommendation is profile-coherent (default tolerance) iff `d(u, i) <= 1`. PC@k for a single user is the share of the top-k list that is coherent:

```
PC@k(u) = (1/k) * |{i in top_k(u) : d(u, i) <= 1}|
```

Two boundary conventions apply. Recommendations to assets whose risk band cannot be determined (typically a few thin-history assets that fail both the subcategory mapping and the volatility-quartile fallback) are treated as discordant. Customers without a declared or predicted MiFID risk band are excluded from the per-customer PC@k contribution by setting their value to zero, so that the metric draws only on customers with a usable profile signal. The aggregate PC@k for a model on a split is the mean across the eligible-customer set, with the zero-contribution convention above. A uniformly-random recommender that samples assets without regard to customer profile achieves a band-conditional baseline that depends on the asset-universe distribution:

```
pi(b) = |{i in A : |b - b_i| <= 1}| / |A|
```

On FAR-Trans (asset bands distributed 190 / 333 / 105 / 178 across Conservative / Income / Balanced / Aggressive), this gives `pi = (0.649, 0.779, 0.764, 0.351)`. The Aggressive baseline is the lowest because its tolerated set `{Balanced, Aggressive}` covers only 35% of the asset universe, while the Income baseline at 0.78 includes the three central bands. PC-lift@k normalises PC@k by the customer's band baseline:

```
PC-lift@k(u) = PC@k(u) / pi(b_u)
```

A lift of 1.0 means the recommender is no better than uniform sampling for that customer's band. Lift values above 1.0 indicate above-random skill. We report PC@k and PC-lift@k jointly because aggregate PC@k can be inflated by a recommender that simply tilts toward the largest asset band (Income, in our case), while PC-lift@k normalises this away.

### 2.2 Profile-Coherent LightGCN

The base architecture is LightGCN (He et al., 2020), trained with Bayesian Personalised Ranking (BPR) on the bipartite customer-asset interaction graph. We extend it with two independently-toggleable components.

**Profile embedding.** A per-customer offset is added to the base user embedding before LightGCN message passing:

```
U_tilde[u] = U[u] + W_p (E_rb[b_u] + E_ct[t_u] + E_ic[c_u])
```

where `E_rb`, `E_ct`, `E_ic` are lookup tables for declared risk band, customer type (Mass / Premium / Professional), and investment capacity (LT30K / 30K_80K / 80K_300K / GT300K). `W_p` is a learned linear projection back to the LightGCN embedding dimension. The base user embedding `U` remains free; the profile offset is the only path by which declared-band information enters the message-passing computation.

**Profile coherence regulariser.** During training, for each positive interaction `(u, i+)` in the BPR batch, we compute `d(u, i+)` from precomputed per-asset bands and penalise the predicted score weighted by raw discordance:

```
L_PC = E_{(u, i+)} [ d(u, i+) * sigma(s_{u, i+}) ]
```

The total training loss is:

```
L = L_BPR + alpha * L_2 + lambda * L_PC
```

The penalty is multiplicative in `d` rather than thresholded, so a Balanced asset for an Aggressive customer (`d = 1`) is penalised half as much as an Income asset (`d = 2`). The penalty applies to positives only; negatives are sampled uniformly from the asset pool minus the user's training positives (matching vanilla LightGCN's BPR convention) and are not down-weighted by their discordance. This is a deliberate design choice: a discordance penalty on negatives would also push the model toward assigning high scores to coherent negatives, which would conflict with the BPR ordering signal. The two components are independently toggleable, which makes a 2x2 ablation possible without changing other hyperparameters.

### 2.3 Evaluation Protocol

Following FAR-Trans, we use 69 monthly temporal splits between August 2019 and April 2025. For each split, models are trained on interactions whose timestamp falls before the split's `time_point` and evaluated on test interactions between `time_point` and `test_end` (typically a 5-month window). The eligible asset universe at `time_point` is the FAR-Trans-provided list of assets that have a listing date prior to `time_point`, computed point-in-time without reference to later metadata. Per-split metrics are nDCG@10, ROI@10 (geometric 30-day-rescaled forward return averaged over the top-10), Recall@10, PC@10, and PC-lift@10.

A single "trial" in this paper corresponds to one fully specified hyperparameter configuration evaluated as 69 fresh train-evaluate cycles, one per temporal split, each cycle producing its own model fit and its own per-split metric vector. A trial's reported mean is the simple average of the 69 per-split values; the reported standard deviation is the across-split standard deviation. The asset risk classification used by the PC@k metric (and by the `L_PC` regulariser at training time) is computed once on the full FAR-Trans price history rather than per split, which we discuss as a methodological caveat in Section 5.6.

The FAR-Trans evaluation protocol has a known overlap between the validation and evaluation windows for some splits; this is documented in the [Validation/Evaluation Window Overlap](README.md#validationevaluation-window-overlap-known-caveat) section of the project README. The overlap affects all models symmetrically and so is not expected to bias relative comparisons between models, but absolute per-split metric levels should be read with this caveat in mind.

For the headline coherence-gap analysis we estimate a panel OLS:

```
coherent_share_{u, s, m} = beta_0 + beta_b * C(b_u) + beta_m * C(m)
                         + beta_bm * C(b_u) x C(m) + beta_s * C(s) + epsilon
```

where `u` indexes customers, `s` indexes splits, `m` indexes models, and the dependent variable is the share of customer `u`'s top-10 in split `s` that is coherent under model `m`. Standard errors are clustered on `customer_id` to absorb arbitrary within-customer correlation across splits. The interaction terms `beta_bm` are the headline coefficients: each one quantifies how a given model shifts a given band's coherence relative to the reference cell (Conservative band on alphabetically-first model). Predictions are reported at split index 36, the empirical median of the unbalanced panel (since the eligible-customer set varies by split), which serves as a representative middle-of-window time point.

**Statistical inference.** Where this paper compares two trials' aggregate metrics, the comparison is implicitly a paired-per-split contrast over 69 splits. Per-split metric standard deviations are on the order of 0.01 for PC@10 and 0.07 for nDCG@10 (LightGCN-best trial values), which gives standard errors of approximately 0.001 and 0.008 respectively for the trial mean across splits, and approximately 0.0017 and 0.012 for the standard error of a paired difference between two trials. Differences cited in the body text exceed these bands by an order of magnitude or more unless otherwise noted. Pearson and Spearman correlations across the 69 monthly splits are subject to temporal autocorrelation; we do not apply HAC adjustment, so the reported values should be read as point estimates rather than confidence-bound-bearing inferential statements. The qualitative direction of the trial 00001 sign reversal (Pearson +0.64) is robust to plausible reductions in the effective sample size, which we discuss in Section 4.5.

## 3. Findings

### 3.1 Profile-discordance in FAR-Trans is prevalent and structural

The dataset audit (full detail in [Dataset Audit Findings](README.md#dataset-audit-findings)) shows that 18.6% of all observed buy transactions are profile-discordant under the default tolerance, and the discordance is concentrated on specific customers rather than spread uniformly across the population. The customer-level self-discordance distribution is sharply bimodal: a majority of customers are perfectly profile-coherent across all of their transactions, while a non-trivial tail of customers (~14%) is discordant on more than half of their buys. Per-customer self-discordance correlates more strongly with `customerType` and declared band than with calendar year, and it is stable across the 2019-2025 window including the 2020 COVID drawdown and the 2022 yield-spike regime. The audit's most relevant finding for this paper is that Conservative and Aggressive customers (the two declared bands at the ordinal extremes) reach toward the centre in their transaction history: a Conservative customer's discordant buys are disproportionately Income, and an Aggressive customer's discordant buys are disproportionately Balanced. This is the empirical pattern that the model-side audit will reproduce inside vanilla LightGCN.

### 3.2 Vanilla LightGCN and Random Forest are risk-blind in structurally different ways

The two FAR-Trans baselines, optimised with full grid search over their canonical hyperparameter spaces, achieve the following at their primary-metric optima (LightGCN by nDCG@10; Random Forest by ROI@10):

| Model | nDCG@10 | ROI@10 (%/mo) | Recall@10 | PC@10 |
|---|---:|---:|---:|---:|
| LightGCN | 0.329 | -0.60 | 0.492 | 0.787 |
| Random Forest | 0.020 | +1.29 | 0.037 | 0.667 |

These aggregate numbers conceal qualitatively different failure modes. Three diagnostics expose the structure.

**Personalisation diversity.** In any given split, Random Forest produces only ~17 unique top-10 multisets across all evaluated customers (15 in split 0 across 625 customers); LightGCN produces ~148. Random Forest's `recommend_for_user` walks a single per-split global ranking and only filters out already-held assets, so two customers with similar holdings receive identical top-10s. LightGCN, in contrast, scores assets per customer via the user-asset embedding inner product.

**Per-customer-band asset-band composition.** Conditional on declared customer band, the share of recommendations falling into each asset band reveals whether the model differentiates on the risk axis. For LightGCN:

| Customer band | b0 | b1 | b2 | b3 |
|---|---:|---:|---:|---:|
| Conservative | 0.118 | 0.375 | 0.393 | 0.114 |
| Income | 0.116 | 0.379 | 0.388 | 0.118 |
| Balanced | 0.121 | 0.400 | 0.370 | 0.110 |
| Aggressive | 0.126 | 0.408 | 0.359 | **0.106** |

Aggressive customers receive *less* Aggressive (b3) content (10.6%) than Conservative customers do (11.4%). For Random Forest the table is degenerate: every row is essentially `(0.211, 0.126, 0.201, 0.463)` to three decimals, because the global ranking is the same for every customer. Neither model is risk-aware.

**Lift over band-conditional random baseline.**

| Customer band | Random PC | LightGCN | LGCN lift | RF | RF lift |
|---|---:|---:|---:|---:|---:|
| Conservative | 0.649 | 0.493 | **0.76x** | 0.337 | **0.52x** |
| Income | 0.779 | 0.882 | 1.13x | 0.537 | **0.69x** |
| Balanced | 0.764 | 0.879 | 1.15x | 0.790 | 1.03x |
| Aggressive | 0.351 | 0.466 | 1.33x | 0.664 | 1.89x |

LightGCN is *below random* for Conservative customers; Random Forest is below random for Conservative and Income, the two largest customer segments (combined: 17,510 of 28,770 customers with a usable risk band, 60.9%). Random Forest's "advantage" on Aggressive is mechanical: its global ranking is 46.5% band-3 (vs the 22.1% asset-universe share), so the 1.89x lift is a base-rate property of the ranking, not personalisation.

Two reporting conventions are worth noting briefly. The per-band PC values in this subsection are pooled means restricted to customers with a usable risk band, while the headline aggregate PC@10 of 0.787 follows the per-customer metric defined in Section 2.3 (averaged over each split's eligible customer set, including customers without a usable band who contribute zero). The customer-band counts in this paper include both declared and FAR-Trans-predicted bands (28,770 customers total); restricting to strictly declared bands gives 21,629 customers, and the per-band lifts and panel regression coefficients are stable across the two definitions.

These findings answer RQ3 affirmatively: existing FAR baselines under-serve declared-band coherence, and the failure concentrates on the two largest customer segments and the two ordinal extremes.

### 3.3 The pull-to-centre is small for LightGCN; the universe-tilt is large for Random Forest

The mean ordinal asset band of recommendations, averaged over 69 splits, exposes the magnitude of each model's bias relative to the asset universe (mean ordinal band 1.494):

| | Universe | LightGCN | RF |
|---|---:|---:|---:|
| Mean recommended band | 1.494 | 1.478 | **1.924** |
| Std across splits | 0.034 | 0.054 | 0.297 |

LightGCN sits 0.02 bands below the universe mean; Random Forest sits 0.43 bands above it, an order of magnitude larger tilt. RF's headline ROI of +1.29%/month is therefore partly a regime artefact: in a backtest window where high-band assets drifted up, overweighting band-3 by 24.4 percentage points relative to availability (46.5% in the top-10 vs 22.1% in the universe) captures market beta rather than model skill. In a sustained equity-bear regime the sign of the LightGCN-versus-RF ROI comparison would be expected to invert. This is the universe-overweight contingency that any FAR-Trans ROI claim must disclose.

### 3.4 The Aggressive coherent slice is the worst-performing realised-return cell under LightGCN

Decomposing realised monthly return (%/mo) by `(declared band, model, coherence)` for all four model configurations evaluated in this paper:

| Band | LGCN coh | LGCN disc | PC-LGCN 00001 coh | PC-LGCN 00001 disc | PC-LGCN 00005 coh | PC-LGCN 00005 disc | RF coh | RF disc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Conservative | -0.04 | -1.58 | -0.19 | -0.74 | +0.17 | -0.24 | +0.55 | +1.54 |
| Income | -0.47 | -3.04 | -0.31 | -2.35 | -0.35 | +0.06 | +1.31 | +1.28 |
| Balanced | -1.01 | +0.40 | -0.42 | -0.12 | +0.07 | -0.53 | +1.63 | -0.34 |
| Aggressive | **-2.02** | -0.01 | **-0.58** | -0.43 | -1.52 | +0.49 | +1.50 | +0.53 |

The LightGCN Aggressive coherent slice loses 2.02% per month, the worst cell in the matrix; the same band-3 asset universe earns +1.50% per month under Random Forest, so the loss is item-selection within the coherent slice rather than a regime property of band-3 assets. PC-LGCN trial 00001 reduces this cell to -0.58% per month, closing 71% of the gap to Random Forest, while trial 00005's b3-substitution mechanism produces -1.52% per month (closing only 25%); we examine the substitution mechanics in Section 4.3.

### 3.5 PC-LGCN closes the per-band gap

We estimate the panel regression of Section 2.3 with three models: LightGCN, Random Forest, and PC-LGCN. Two specifications pin PC-LGCN to two distinct trials from the seven-trial sweep (justified in Section 4): trial 00001 (`L_PC` alone, λ=0.5) and trial 00005 (full method, λ=1.0). Predicted PC@10 reported at split index 36 (the empirical median of the unbalanced panel, since the eligible-customer set varies by split):

| Band | LightGCN | PC-LGCN 00001 | PC-LGCN 00005 | RF |
|---|---:|---:|---:|---:|
| Conservative | 0.518 | 0.572 | **0.879** | 0.362 |
| Income | 0.907 | 1.010* | 1.013* | 0.562 |
| Balanced | 0.904 | 1.001* | 1.010* | 0.815 |
| Aggressive | 0.491 | 0.604 | **0.762** | 0.689 |

(*Linear extrapolation past the [0,1] ceiling; the underlying observed PC is at saturation.) All cell shifts are statistically significant at p < 10^-9 with cluster-robust standard errors on `customer_id`. Trial 00005 lifts Conservative PC@10 by +36 percentage points and Aggressive by +27 percentage points, the two bands the audit identified as below-random and as the LightGCN pull-to-centre target.

Per-band lifts over the random baseline `pi = (0.649, 0.779, 0.764, 0.351)`, computed from the panel-regression-predicted PCs above:

| Band | LightGCN | PC-LGCN 00001 | PC-LGCN 00005 | RF |
|---|---:|---:|---:|---:|
| Conservative | 0.80x | 0.88x | **1.35x** | 0.56x |
| Income | 1.16x | 1.30x* | 1.30x* | 0.72x |
| Balanced | 1.18x | 1.31x* | 1.32x* | 1.07x |
| Aggressive | 1.40x | 1.72x | **2.17x** | 1.96x |

(Trial-aggregate lifts computed without panel-regression smoothing differ slightly: e.g., trial 00005 aggregate lift on Aggressive is 2.10x instead of the predicted-PC value of 2.17x. We report the panel-derived lifts here for consistency with the predicted-PC table above.) Trial 00005 is the only configuration that scores above the random baseline for all four bands simultaneously; vanilla LightGCN, trial 00001, and Random Forest each fail at least one band.

## 4. Ablation Studies

The seven-trial sweep at the winning LightGCN backbone (`embedding_dimension=128, number_of_layers=2, learning_rate=1e-3`) consists of a 2x2 component ablation at λ=0.5 and a three-point λ sensitivity sweep on the both-flags-on cell.

### 4.1 Component ablation (2x2 at λ=0.5)

| Trial | profile_embedding | L_PC | nDCG@10 | ROI@10 (%/mo) | Recall@10 | PC@10 |
|---|---|---|---:|---:|---:|---:|
| 00000 | off | off | 0.329 | -0.59 | 0.492 | 0.787 |
| 00001 | off | **on** | 0.301 | **-0.18** | 0.434 | **0.888** |
| 00002 | **on** | off | 0.310 | -0.46 | 0.473 | 0.780 |
| 00003 | **on** | **on** | 0.241 | -0.24 | 0.380 | **0.915** |

Two findings from the 2x2:

1. **Profile embedding alone (00002) does nothing.** PC@10 is unchanged from the off-cell (0.780 vs 0.787). This is the necessary but not sufficient condition: the embedding head adds a learnable per-customer band signal but, without `L_PC` to make it cost-effective during training, the gradient from BPR alone provides no incentive to use it.
2. **`L_PC` alone (00001) achieves 89% of the PC gain at half the nDCG cost of the full method.** The metric improvement is `L_PC`-driven, not embedding-driven.

Each cell in the 2x2 is a single training run with a deterministic seed and 69 train-evaluate cycles (one per temporal split). The per-trial metric is the mean across these 69 splits; the corresponding standard deviation is on the order of 0.01 for PC@10 and 0.07 for nDCG@10, so the 0.7 pp profile-embedding-alone effect (00002 vs 00000) is statistically indistinguishable from training noise without multi-seed repetitions, while the 10.1 pp `L_PC`-alone effect (00001 vs 00000) and the 12.8 pp full-method effect (00003 vs 00000) are well outside any plausible seed-variance band on a paired-per-split test. We do not run multi-seed repetitions in this study, so the "embedding alone does nothing" claim is best interpreted as "embedding alone has no detectable effect at this seed and one λ setting"; multi-seed repetitions or a larger embedding-only λ search would be required to falsify the alternative hypothesis that the embedding has a small-but-real effect that is masked by single-run variance.

### 4.2 Lambda sensitivity (both flags on)

| λ | nDCG@10 | ROI@10 (%/mo) | Recall@10 | PC@10 |
|---|---:|---:|---:|---:|
| 0.1 | 0.311 | -0.46 | 0.473 | 0.787 |
| 0.5 | 0.241 | -0.24 | 0.380 | 0.915 |
| 1.0 | 0.184 | -0.13 | 0.287 | **0.934** |
| 2.0 | 0.175 | -0.15 | 0.264 | 0.936 |

λ=0.1 is too small to bind (PC unchanged). λ=0.5 is the elbow. λ=1.0 saturates PC; λ=2.0 is past the elbow and continues to bleed nDCG without a corresponding PC gain. λ=1.0 is the operating point of the full method.

### 4.3 Mechanistic decomposition: trials 00001 and 00005 are structurally different

The two trials with the cleanest claims, 00001 (`L_PC` alone, λ=0.5) and 00005 (full method, λ=1.0), do not differ along a single axis. They are structurally distinct mechanisms.

**Trial 00001's mechanism: centre-band compression.** Conditioning the top-10 asset-band composition on declared customer band:

| Customer band | b0 | b1 | b2 | b3 |
|---|---:|---:|---:|---:|
| Conservative | 0.047 | 0.499 | 0.436 | 0.018 |
| Income | 0.027 | 0.470 | 0.489 | 0.015 |
| Balanced | 0.024 | 0.425 | 0.533 | 0.018 |
| Aggressive | 0.026 | 0.396 | 0.553 | **0.025** |

The b3 column is flat at 1.5-2.5% across all customer bands (Aggressive customers receive 0.025 band-3 content vs Conservative's 0.018, a 1.4-fold ratio). Trial 00001 wins PC@k by collapsing the top-10 onto the b1-b2 centre even more aggressively than vanilla LightGCN; coherence rises because the band-edge tolerance window (`d <= 1`) absorbs the central recommendations, not because the model has learnt to differentiate Aggressive from Conservative customers. Notably, the Conservative customer's coherence improvement under trial 00001 comes from *replacing* b0 (Conservative) picks with b1 (Income) picks: their b0 share falls from 0.118 in vanilla LightGCN to 0.047, while their b1 share rises from 0.375 to 0.499, the Income content absorbing what would otherwise have been Conservative recommendations. The discordance distribution overall shifts as follows: `d = 0` (exact match) rises from 0.314 in vanilla LightGCN to 0.399 (+8.5 pp), `d = 1` (edge match) rises from 0.475 to 0.489 (+1.4 pp), and `d >= 2` (discordant) falls from 0.211 to 0.111 (-10 pp). Approximately 85% of the 10-pp reduction in discordant mass goes to exact-match and 15% to edge-match, but per the contingency table this exact-match increase is concentrated in the b1-b2 cells of the centre, not in the b0 or b3 corners.

**Trial 00005's mechanism: risk-axis diagonalisation.** Same conditioning:

| Customer band | b0 | b1 | b2 | b3 |
|---|---:|---:|---:|---:|
| Conservative | **0.593** | 0.261 | 0.108 | 0.038 |
| Income | 0.050 | **0.881** | 0.057 | 0.012 |
| Balanced | 0.015 | 0.163 | **0.656** | 0.166 |
| Aggressive | 0.090 | 0.173 | 0.290 | **0.447** |

Trial 00005 is the only configuration in the study that produces a clean diagonal. Aggressive customers receive 44.7% Aggressive content vs Conservative's 3.8%, a 12-fold ratio. The discordance distribution shifts mass into `d = 0` (up from 0.314 in vanilla LightGCN to 0.689). Personalisation diversity is preserved (587 unique top-10s in split 0, slightly more than vanilla's 555 and many times more than RF's 15), so the regulariser is not collapsing toward popularity.

The two trials' Aggressive-band coherent ROI tells the trade-off:

| Configuration | Aggressive coherent ROI (%/mo) |
|---|---:|
| LightGCN | -2.02 |
| PC-LGCN 00001 (L_PC alone) | **-0.58** |
| PC-LGCN 00005 (full method) | -1.52 |
| Random Forest | +1.50 |

Trial 00001 closes 71% of the gap to RF on this cell while losing only 2.85 pp aggregate nDCG; trial 00005 only closes 25% of the gap. The mechanism is composition-driven. Within the Aggressive coherent slice, trial 00001 holds a 4.3% band-3 share (`0.025 / (0.553 + 0.025)`) and a 95.7% band-2 share, while trial 00005 holds 60.7% band-3 (`0.447 / (0.290 + 0.447)`) and 39.3% band-2. Treating the trial 00001 slice as approximately a pure-b2 portfolio earning -0.58%/month and the trial 00005 slice as a 60.7/39.3 b3/b2 mix earning -1.52%/month, the implied b3-only mean return within the Aggressive customer subset is approximately -2.13%/month and the implied b2-only mean return is approximately -0.58%/month, a band-2-over-band-3 spread of roughly 1.55 pp/month during this backtest period. Trial 00001 satisfies coherence by edge-tolerance substitution into b2; trial 00005 satisfies it by exact-match substitution into b3. On this dataset and over this period, edge-tolerance substitution is the more profitable composition.

A complementary view of the substitution mechanism comes from comparing trial 00001's top-10 to vanilla LightGCN's top-10 for Aggressive customers (Jaccard = 0.43, mean intersection = 5.77 / 10 assets per customer-split). On the 4.23 slots that change per pair (107,215 swaps in total): 26.9% of the removed slots were band-0, 40.6% band-1, 10.4% band-2, 22.1% band-3; 3.2% of the added slots are band-0, 37.7% band-1, 56.1% band-2, 3.0% band-3. The dominant move is replacing extreme-band picks (b0 and b3) with band-2 picks. Realised return on the slots vanilla LightGCN held was -0.56% per month; on the slots trial 00001 holds it is +0.46% per month, a +1.02 pp/slot swing.

### 4.4 Stuck-tail elimination

Vanilla LightGCN had 47 customers (3.8% of 1,222 declared Aggressive) at `mean PC = 0` across nearly every split, an audit finding flagged as a deployment liability. The regulariser closes this tail:

| Configuration | Customers stuck at PC = 0 | Rescue rate | Mean PC of rescued |
|---|---:|---:|---:|
| LightGCN | 47 / 47 stuck | 0% | n/a |
| PC-LGCN 00001 | 10 / 47 stuck | 79% | 0.207 |
| PC-LGCN 00005 | **0 / 47 stuck** | **100%** | 0.549 |

Within-band standard deviation of per-customer mean PC, on the Aggressive band: vanilla LightGCN 0.21, trial 00001 0.20, trial 00005 0.17, Random Forest 0.06. The full method lifts the band mean while compressing the tail without collapsing toward Random Forest's near-degenerate distribution. The wealth-PC anti-correlation observed in vanilla LightGCN (CAP_GT300K mean 0.36 vs CAP_LT30K mean 0.50, spread 0.14) compresses to 0.12 under trial 00005, still present but smaller.

### 4.5 PC-versus-ROI relationship

Pearson and Spearman correlation between PC@10 and ROI@10 across the 69 monthly splits:

| Model | Pearson | Spearman |
|---|---:|---:|
| LightGCN | -0.432 | -0.346 |
| **PC-LGCN 00001** | **+0.637** | **+0.619** |
| PC-LGCN 00005 | -0.259 | -0.127 |
| RF | -0.072 | -0.119 |

Under `L_PC`-only regularisation the trade-off does not soften: it reverses sign. Splits on which trial 00001 is more profile-coherent are also splits on which it earns higher ROI, and Spearman tracks Pearson closely for all four models so the relationship is not driven by a small number of leverage points. The sign reversal is the strongest single-number RQ2 result this paper produces. Trial 00005 retains a weakly negative correlation, consistent with its over-regularised b3 substitution dragging realised return on coherence-heavy splits.

The 69 monthly splits are temporally adjacent and the per-split metrics inherit the autocorrelation of the underlying market series, so the effective sample size is smaller than 69. Applying a Fisher z-transform with the naive sample size gives an approximate 95% confidence interval on the trial 00001 Pearson estimate of `[+0.47, +0.76]`. Treating the autocorrelation conservatively (e.g., assuming an effective sample size on the order of 35 monthly observations) widens this to roughly `[+0.39, +0.80]`. The sign of the reversal and the magnitude of the qualitative shift relative to LightGCN's `-0.43` are robust across either basis; the precise endpoint of the interval should be read as approximate.

## 5. Discussion

### 5.1 The component-wise mechanism is the load-bearing finding

The thesis can credibly defend a result that is more interesting than a single-cell win. The 2x2 ablation cleanly attributes function to mechanism: `L_PC` alone is sufficient for metric improvement and for reversing the PC-ROI trade-off, but it achieves both via centre-band compression rather than risk-axis personalisation. The profile embedding is necessary for risk-axis personalisation, but it is inert without `L_PC` to bind it during training. The full method at λ=1.0 produces the diagonal, which is what RQ4 asked for, but at the cost of nDCG drop and a rebound in the Aggressive coherent ROI cell. We report both trials as the headline rather than picking one, because the difference between them is the cleanest empirical answer the experimental design can produce to the question "what does each component do?".

### 5.2 The PC-versus-ROI trade-off is not a fundamental property

A strict reading of FAR-Trans's earlier results would suggest that profile coherence costs ROI: vanilla LightGCN has a -0.43 Pearson correlation between PC@10 and ROI@10 across splits. The trial 00001 result (+0.64 Pearson) demonstrates this is not a fundamental property of the FAR-Trans market. The same recommender architecture, with the same backbone hyperparameters, can be regularised into a regime where PC and ROI co-move. The mechanism is replacing extreme-band picks (b0 for non-Conservative customers; b3 for non-Aggressive) with band-2 (Balanced) picks, which sit one band off the customer's declaration and inside the coherence window. Replacement returns +1.0 pp/swap on average in the test window: the discarded slots were predominantly band-0 (Conservative) and band-3 (Aggressive) assets, which on FAR-Trans during this backtest period earned lower mean monthly returns within the Aggressive customer subset than the band-2 (Balanced) substitutes that took their place (the slice arithmetic in Section 4.3 implies an approximate b2-versus-b3 return spread of 1.5 pp/mo on this subset).

### 5.3 Random Forest's headline ROI advantage is recontextualised

Random Forest's +1.29%/month aggregate ROI is genuinely above vanilla LightGCN's -0.60%/month, but the audit framework establishes that RF's universe-overweight tilt of +0.43 bands relative to the asset universe (driven primarily by a 24.4 percentage-point overweight of band-3) is the principal mechanism by which RF captures market beta during this rising-equity backtest. We cannot decompose the 1.89 pp/mo headline ROI gap into a precise "regime" share and "skill" share without an independent benchmark in a different macro regime, but the magnitude of the universe overweight makes a substantial regime contribution structurally unavoidable on this data. Profile-Coherent LightGCN at trial 00005 surpasses Random Forest on coherence in three of four bands (Conservative by 52 pp, Income and Balanced past saturation), trails by 7 pp on Aggressive, and is the only model in the study that scores above random for all four bands. The PC@k axis therefore admits a clear ranking; the ROI@k axis remains environment-dependent.

### 5.4 The stuck-tail outcome is deployment-relevant

The 47 chronically-discordant Aggressive customers under vanilla LightGCN are not abstract. They are observable customer IDs whose top-10 lists across nearly every monthly split contain zero coherent recommendations. The full method eliminates this entire population (47 of 47 rescued, mean PC of the rescued subset = 0.55). For a deployed FAR system whose suitability claims must be defensible at the per-customer level under MiFID II, this is the metric that auditors will look at. Aggregate PC@10 hides the tail; the within-band standard deviation and the count of stuck customers expose it.

### 5.5 Methodological contribution: PC-lift@k normalisation

The band-conditional random baseline `pi(b)` is non-uniform on FAR-Trans because the asset universe is skewed (Income is 41% of the catalogue; Balanced is 13%; Aggressive's coherent set is the smallest at 35% of total assets). Reporting raw PC@k allows a centre-tilting recommender to inflate the metric on Income and Balanced customers without delivering structural alignment. PC-lift@k makes this inflation visible: trial 00001's panel-regression-implied lift profile is `(0.88, 1.30, 1.31, 1.72)` across Conservative / Income / Balanced / Aggressive, with the Conservative cell remaining below the random baseline despite a 10.1 pp aggregate PC@k gain over vanilla LightGCN; trial 00005's profile is `(1.35, 1.30, 1.32, 2.17)`, the only configuration that exceeds the random baseline on every band. The aggregate-PC-derived lift (computed without panel-regression smoothing) gives slightly different numerics (e.g., trial 00005 Aggressive lift of 2.10x vs the panel-derived 2.17x) but the same qualitative conclusion. We recommend reporting both raw PC@k and PC-lift@k jointly in any future PC@k publication, with the basis (per-trial aggregate or panel-regression-predicted) stated explicitly to prevent the kind of conflation we noted in our own intermediate drafts.

### 5.6 Limitations

The findings in this paper inherit FAR-Trans's specifics. The asset universe is small (806 assets) and biased toward European listings. The customer base is retail rather than institutional. The temporal window includes the COVID drawdown, the 2022 yield-spike regime, and the 2023-2025 equity recovery, which is a representative but not exhaustive macro spread.

The hierarchical risk-class mapping over assets relies on subcategory metadata that is denser for mutual funds than for stocks, where a volatility-quartile fallback is used; the ablation in [Dataset Audit Findings](README.md#dataset-audit-findings) shows the two mappings agree within 1-2 percentage points of cross-band assignment, so the substantive conclusions are robust to this choice.

**Risk-classification lookahead.** The asset risk classification used throughout this paper is computed once on the full FAR-Trans price history (trailing 252-day annualised volatility, with the most recent rolling-window value retained per asset; cross-section quartile cutoffs likewise computed on the pooled cross-section). For PC@k as a metric this is interpretable: PC@k measures alignment to the canonical, stable, end-of-dataset risk classification, which matches how a regulator-aligned suitability check would treat a "currently Aggressive" stock. For PC-LGCN's training, however, the `L_PC` regulariser uses these end-of-dataset bands when training on early-window splits, which is genuine lookahead. Audit Finding 4 (discordance is stable across regimes) provides indirect evidence that the practical magnitude of this lookahead is small (per-asset risk bands are mostly time-stationary on FAR-Trans), but a fully time-honest sensitivity check that recomputes asset bands per split using only `time_point`-prior price history is not part of the experiments reported here.

**Single-seed component ablation.** Each of the seven PC-LGCN trials is a single training run with a deterministic LightGCN backbone seed. Across-seed variance is therefore not estimated, and the "profile embedding alone has no measurable effect" claim is supported by a single-run 0.7 pp difference that is within the plausible band of seed-induced training noise. The 10.1 pp `L_PC`-alone effect and the 14.6 pp full-method effect are robust to any plausible seed variance under paired-per-split tests, but a defensible publication of the necessity result (the embedding is necessary for diagonalisation) would benefit from a multi-seed repetition.

**Backbone confounding.** The PC-LGCN sweep is conducted at the LightGCN backbone selected by the nDCG@10 grid search (`embedding_dimension=128, number_of_layers=2, learning_rate=1e-3`). PC-LGCN may exhibit a different backbone optimum: the regulariser changes the loss landscape, and a smaller embedding dimension or shallower graph could yield better PC-versus-quality trade-offs. We do not test alternative backbones for PC-LGCN.

**Validation/evaluation window overlap.** FAR-Trans's temporal-split convention has overlapping `time_point` windows between training and validation buckets. We follow the FAR-Trans paper's protocol; the specific caveat is documented in the [Validation/Evaluation Window Overlap](README.md#validationevaluation-window-overlap-known-caveat) section of the project README. The substantive comparisons in this paper (LightGCN vs PC-LGCN ablation, and per-band coherence gaps) are within-protocol so the caveat affects all models symmetrically.

**External validity.** The trade-off between metric-efficient (trial 00001) and structurally-correct (trial 00005) configurations may close with richer asset-side input signal (e.g., asset content embeddings or longer interaction history), or it may not. We do not test this here. Generalisation beyond FAR-Trans should also be expected to be sensitive to the asset-universe band distribution: a portfolio with a different `pi(b)` profile will have different lift dynamics under the same model, by construction.

## 6. Conclusion

This paper introduced Profile Coherence at k as a regulator-aligned *risk-band* evaluation axis for Financial Asset Recommendation, with a band-conditional normalisation (PC-lift@k) that makes the metric scale-invariant across customer segments. PC@k captures the risk-band dimension of MiFID II suitability and does not address the other suitability axes (investment horizon, capacity, knowledge and experience, sustainability preferences); a comprehensive suitability metric for FAR systems would compose multiple alignment terms of which PC@k is one. Applied to the FAR-Trans dataset, the metric exposes that the two canonical FAR baselines, optimised over their full hyperparameter grids, are functionally risk-blind in distinct ways: Random Forest emits a single global ranking that achieves above-random coherence only on the smallest customer segment, while LightGCN's per-customer ranking does not differentiate on the risk axis at all and scores below the random baseline for the entire Conservative segment.

We proposed Profile-Coherent LightGCN, a minimal architectural extension that adds a per-customer profile embedding and a multiplicative discordance regulariser to the standard BPR loss. A seven-trial component ablation reveals that the two extensions play structurally distinct roles: the regulariser alone is sufficient to lift PC@k by 10.1 percentage points and to reverse the PC-versus-ROI trade-off across splits (Pearson -0.43 to +0.64), but it does so by compressing recommendations onto the central risk bands rather than by learning per-customer band structure. The profile embedding alone has no measurable effect. Both components together at λ=1.0 produce the only configuration in the study with a genuine `(customer-band, asset-band)` diagonal: Aggressive customers receive 12 times more Aggressive content than Conservative customers, the chronically-discordant tail is eliminated, and the model exceeds the random baseline for all four customer bands.

The headline outcome for the field is that profile coherence and ranking quality are not in fundamental tension on FAR-Trans. They appear to trade off under vanilla LightGCN, but the trade-off can be made favourable (Pearson +0.64) under a single-component regulariser, and structural risk-axis personalisation is achievable as a separate research goal at a quantifiable nDCG cost. The choice between metric-efficient and structurally-correct configurations is therefore not a free parameter to be hidden behind a tuning sweep but an explicit modelling decision a deployer should make with their regulator's expectations in mind.
