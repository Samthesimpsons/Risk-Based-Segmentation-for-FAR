# SMU Capstone: Temporal User Modelling for Financial Asset Recommendation

A sequential Transformer-based recommender for the FAR-Trans dataset that models temporal dynamics of investor behaviour and combines predicted user interest with asset profitability.

## Table of Contents

1. [Paper Summary: FAR-Trans](#paper-summary-far-trans)
2. [Problem Statement](#problem-statement)
3. [Novelty and Research Contributions](#novelty-and-research-contributions)
4. [Proposed Approach](#proposed-approach)
5. [Source Code Architecture](#source-code-architecture)
6. [Working with this Repository](#working-with-this-repository)
7. [GPU Cluster](#gpu-cluster)
8. [References](#references)


## Paper Summary: FAR-Trans

This section summarises the FAR-Trans paper that forms the foundation of this project. It is intended as my self-contained reference of the entire paper.

I have also taken the liberty of converting the research paper into `markdown` format from `TeX` so that large language modelling tools can digest its context more easily compared to `pdf` format. The markdown version serves as good grounding context for asking questions about the paper.

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

Most existing FAR models are developed over **proprietary or simulated datasets**, making fair comparison across methods impossible.

The only prior public dataset (ObjectWay, Musto et al. 2014) has 1,172 users but **lacks pricing data and asset identifiers**, which prevents price-based approaches from being tested.

**FAR-Trans fills this gap**: the first public dataset for FAR that contains both real asset pricing information and real retail investor transactions. The paper also provides a **benchmark comparison of 11 FAR algorithms** as baselines for future research.

### Dataset

[FAR-Trans](https://doi.org/10.5525/gla.researchdata.1658) dataset (Sanz-Cruzado et al., 2024): the **first public dataset** for financial asset recommendation containing both pricing information and retail investor transactions. It was collected from a large European financial institution (the National Bank of Greece) and covers January 2018 to November 2022.

#### What the Dataset Contains

| Component | Description |
|---|---|
| **Price time series** | Daily closing prices for 806 financial assets (stocks, bonds, mutual funds) across 38 markets. 703,303 price data points total. |
| **Investment transactions** | 388,049 buy/sell records from 29,090 retail investors. Each transaction includes: customer ID, asset ID, date, buy/sell flag, number of shares, total value, and channel used. |
| **Asset metadata** | For each of the 806 assets: asset type (stock, bond, mutual fund), sub-type (e.g., government vs. corporate bond), name, market, sector, and industry. |
| **Customer profiles** | For each of the 29,090 customers: customer segment, investment risk profile, and investment capacity. All anonymised. |

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

#### Customer Segments

Customers are classified by the bank into five segments based on their managed assets:

| Segment | Description | Count |
|---|---|---|
| **Mass** | < €60,000 in managed assets (investments, deposits, insurance) | 18,610 |
| **Premium** | > €60,000 in managed assets | 8,906 |
| **Professional** | Sole proprietorship: individual exercising business activity | 1,327 |
| **Legal Entity** | Legal entities with bank services | 39 |
| **Inactive** | Customers without available segment data | 208 |

#### Investment Risk Profiles

Each customer is assigned a risk profile based on a 25-question MiFID II regulatory questionnaire (or estimated via linear regression for customers who haven't completed it):

| Risk Profile | Description |
|---|---|
| **Conservative** | Prioritises capital protection. Portfolios: short-term placements, fixed-income securities. |
| **Income** | Aims for fixed income from bond coupons, dividends, short-term placements. Very low risk. |
| **Balanced** | Accepts moderate fluctuations. Mix of bonds, stocks. Medium-term capital gains. |
| **Aggressive** | Targets significant long-term gains. Accepts high risk. |

Most customers have intermediate risk profiles (Income or Balanced).

#### Investment Capacity

Customers are also categorised by how much they can invest:

| Capacity | Range |
|---|---|
| Low | < €30,000 |
| Medium-Low | €30,000 - €80,000 |
| Medium-High | €80,000 - €300,000 |
| High | > €300,000 |

The majority of investors have low investment capacity (< €30k), consistent with the Mass customer segment being the largest.

#### Key Observations About the Data

- **Long-tail distribution**: Over 50% of users have 3 or fewer transactions across the entire 2018-2022 period. Only ~650 customers have more than 100 transactions. This is similar to other recommender datasets like MovieLens.
- **Concentrated asset popularity**: The top 12 assets account for more than 50% of all transactions, yet 75% of traded assets have more than 20 interactions.
- **Activity spike during COVID**: The highest transaction volume occurred in March 2020, likely driven by the market crash.
- **Not all assets are traded**: Only 321 of 806 assets with pricing data have associated transactions.

**Data Loading:** The FAR-Trans dataset is loaded and explored in [`notebooks/data_loading.ipynb`](notebooks/data_loading.ipynb). This notebook handles download, extraction, and initial data inspection.

#### Cleaning and Pre-processing

**Note:** All cleaning and pre-processing steps listed below have already been applied by the paper's authors to the FAR-Trans dataset. The data is production-ready; no additional cleaning is required. See the [original dataset documentation](https://researchdata.gla.ac.uk/1658/).

**Price Cleaning**

| Step | What they did |
|---|---|
| **Remove duplicates** | Ensure at most one price per asset per date |
| **Handle multiple values** | Drop zeros; otherwise keep the value closest to the 5-day trailing price |
| **Remove untradeable assets** | Drop assets with closing price = 0 at any point (capital exchange required) |
| **Remove gapped assets** | Drop assets with time gaps > 10 days (too inaccurate to estimate) |
| **Outlier handling** | If price jumps 10x or drops 90% in one day and reverts the next, replace with 5-day moving average |
| **Currency normalisation** | Convert all prices to euros |
| **Stock split adjustment** | Use Yahoo Finance to identify splits; divide pre-split prices by the split ratio |
| **Fill remaining gaps** | 5-day moving average |

**Transaction Cleaning**

| Step | What they did |
|---|---|
| **Remove blank customers** | Drop transactions with no associated customer ID |
| **Stock split adjustment** | Multiply share counts by the split ratio |
| **Fractional shares (reverse splits)** | Add a sell transaction for the fractional portion (company pays cash for these) |
| **Backfill missing buys** | If a customer sells shares they never bought in the dataset (acquired pre-2018), add a synthetic buy at the earliest available date (usually Jan 2, 2018) |
| **Align to pricing dates** | If a transaction date has no pricing data, move to the nearest date that does |
| **Handle post-series holdings** | If a customer still holds shares after the pricing time series ends, add a sell transaction |
| **Estimate transaction values** | Number of shares x closing price on the transaction date |

### Existing FAR Methods (Related Work)

#### 1st Category: Price-based

- **Not personalised**: same predictions for all customers
- Regression models (Random Forest, SVM) on technical indicators to estimate asset profitability
- More complex models explore time series similarities between assets
- Some incorporate external data: news, social media sentiment
- Stock ranking selection: pick assets maximising a utility function (e.g. combined predicted returns)

#### 2nd Category: Transaction-based

- Uses past investment transactions as the core data source
- Assumes investors follow patterns (individually or as groups)
- **Collaborative filtering**: matrix factorisation, convolutional networks, LightGCN
- **Customer clustering** (e.g. cluster based on risk aversion)
- **Content-based**: add asset info like market sector or enterprise life cycle
- **Apriori Association rule mining**: To be honest, I have not studied this field yet.

#### 3rd Category: Hybrid

- Combines multiple information sources (price + transactions + customer data)
- Collaborative filtering + multi-criteria decision analysis
- Gradient boosting reranking of collaborative filtering outputs using portfolio optimisation

### Task Definition

At time *t*, let `I_u(t)` = set of assets customer *u* has bought before *t*.

A FAR system generates a ranking `R_u ⊂ I \ I_u(t)`: a ranked list of assets the user has **NOT** previously interacted with and ordered by predicted suitability.

The paper poses **two research questions**:

| RQ | Question | Evaluation Metric |
|---|---|---|
| **RQ1** | Which algorithms are best at identifying **profitable** assets? | ROI@k |
| **RQ2** | Which algorithms are best at identifying **future customer investments**? | nDCG@k |

### Evaluation Metrics

#### Dataset Split

The paper does not describe its evaluation schedule explicitly, but the authors' code (`run_experiments.py`) runs two separate date ranges: Aug 2019 to Feb 2021 (28 splits) and Sep 2020 to May 2022 (31 splits), totalling 61 evaluation points. At each time point *t*:
- **Training set**: All pricing data and transactions before *t*.
- **Test set**: All transactions in the 6-month window (t, t + 6 months).
- **Deduplication**: (user, asset) pairs appearing in both sets are kept only in training.
- **Filtering**: Only users active in both sets and assets with pricing data spanning the full test window are retained.

Because consecutive test windows overlap heavily by design, the sliding-window approach ensures the averaged metrics are not dominated by any single market condition.

All metrics are **averaged across all evaluation time points**, providing a robust estimate of model performance across different market conditions.

#### nDCG@10 : Normalised Discounted Cumulative Gain at 10

**What it measures**: How well the model predicts which assets a user will actually acquire in the future test period. This is a standard information retrieval metric that evaluates the **relevance** of a ranked recommendation list.

**How it is computed**:

1. **Relevance**: An asset *i* is relevant to user *u* at time *t* if and only if user *u* acquires asset *i* during the test window (t, t + 6 months). Relevance is binary: 1 if acquired, 0 otherwise.

2. **Discounted Cumulative Gain (DCG@k)**: For a ranked list of *k* recommendations, DCG rewards relevant items appearing at higher ranks with a logarithmic discount:

```
DCG@k = Σ (rel_i / log₂(i + 1))   for i = 1 to k
```

Where `rel_i` is the relevance (0 or 1) of the item at rank *i*. The denominator `log₂(i + 1)` penalises relevant items appearing at lower positions: a relevant item at rank 1 gets full credit (divided by log₂(2) = 1), while one at rank 10 gets less (divided by log₂(11) ≈ 3.46).

3. **Ideal DCG (IDCG@k)**: The maximum possible DCG, achieved by a perfect ranking that places all relevant items at the top.

4. **nDCG@k = DCG@k / IDCG@k**: Normalised to [0, 1]. A score of 1.0 means the model produced a perfect ranking.

nDCG@10 tells me whether the recommender is surfacing assets the investor actually wants. In the original paper, the best nDCG@10 was 0.3404 (LightGCN), meaning collaborative filtering models are best at predicting what users will buy.

#### ROI@10 : Return on Investment at 10

**What it measures**: The average monthly profitability of an equally weighted portfolio constructed from the top 10 recommended assets. This measures the **financial quality** of the recommendations.

**How it is computed**:

1. At each time point *t*, the model produces a top-10 ranked list of assets for each user.
2. For each recommended asset *i*, compute its return over the 6-month test window:

```
Return(i) = (Price at t + 6 months - Price at t) / Price at t
```

3. Convert to a geometric monthly return (compounding-aware):

```
Monthly_Return(i) = (1 + Return(i))^(30 / days_in_period) - 1
```

Where `days_in_period` is the calendar days between the recommendation date and the test-end date.

4. ROI@10 is the average monthly return across all 10 recommended assets (equally weighted portfolio):

```
ROI@10 = (1/10) * Σ Monthly_Return(asset_i)   for i = 1 to 10
```

5. This is then averaged across all users and all 61 time points.

**Benchmark values from the paper**:
- **Market average ROI**: 0.0079 (0.79% per month), the return if you bought every asset equally.
- **Customer average ROI**: 0.0018 (0.18% per month), the actual average return investors achieved.
- **Best model ROI@10**: 0.0259 (Random Forest), 2.59% per month, significantly beating the market.

ROI@10 tells me whether the recommender is surfacing assets that will make money.

### Baseline Models from Paper

#### Price-Based Models (Profitability Prediction)

These models ignore user preferences entirely. They predict which assets will have the highest future ROI based on price history, then recommend those assets to *all* users identically (non-personalised).

**Features used**: Technical indicators computed from historical closing prices:

| Indicator | What it measures |
|---|---|
| **ROI** | Return on investment over a trailing window: how much the asset gained/lost recently |
| **Volatility** | Standard deviation of daily returns: how much the price fluctuates |
| **MACD** | Moving Average Convergence/Divergence: a momentum indicator based on the difference between short-term and long-term exponential moving averages |
| **Momentum** | Raw price change over N days: whether the asset is trending up or down |
| **Rate of Change** | Percentage price change over N days: similar to momentum but normalised |
| **RSI** | Relative Strength Index: oscillator (0-100) measuring whether an asset is overbought (>70) or oversold (<30) |
| **DCO** | Detrended Close Oscillator: removes trend to identify cycles in price |
| **ROI/Volatility** | Risk-adjusted return: how much return per unit of risk (similar to Sharpe ratio) |
| **Min/Max Price** | Lowest and highest closing price over trailing window |

**Models**:

| Model | How it works |
|---|---|
| **Linear Regression** | Fits a linear function from technical indicators to future 6-month ROI. Simple but interpretable. |
| **Random Forest** | Ensemble of decision trees. Each tree is trained on a random subset of features and data. Predictions are averaged. Best ROI@10 in the paper (0.0259). |
| **LightGBM** | Gradient boosted decision trees. Trees are added sequentially, each correcting errors of the previous ones. Faster and often more accurate than Random Forest. |

**Limitation**: These models recommend the same assets to every user. They cannot personalise as they have no concept of user preferences.

#### Transaction-Based Models (User Preference Prediction)

These models use only the binary user-item interaction matrix (did user *u* buy asset *i*?). They predict which assets each user is most likely to buy next.

| Model | How it works |
|---|---|
| **Random** | Recommends assets uniformly at random. Sanity-check baseline. |
| **Popularity** | Ranks assets by total purchase count. Non-personalised but surprisingly strong because the top 12 assets dominate 50%+ of transactions. |
| **Matrix Factorisation (MF)** | Decomposes the user-item matrix into two low-rank matrices (user embeddings x item embeddings). A user's score for an item is the dot product of their embeddings. Classic collaborative filtering approach. |
| **LightGCN** | Graph Convolutional Network for collaborative filtering. Builds a bipartite user-item graph, then propagates embeddings across edges to learn higher-order collaborative signals (e.g., "users who bought A and B also bought C"). **Best nDCG@10 in the paper (0.3404)**. |
| **UB kNN** | User-Based k-Nearest Neighbours. Finds users with similar purchase histories, then recommends assets those similar users bought. |
| **ARM (Apriori)** | Association Rule Mining. Discovers rules like "users who bought asset A tend to also buy asset B" and uses these rules to generate recommendations. |

**Limitation**: These models have no awareness of asset prices or returns. They optimise purely for predicting user purchases, which is why they achieve high nDCG but near-zero ROI.

#### Hybrid Models (Two-Stage Pipelines)

These attempt to combine both signals by running a transaction-based model first, then using its output as features for a profitability model:

| Model | How it works |
|---|---|
| **Hybrid-nDCG** | Takes the recommendation scores from *all* of the above models as features, then trains a LightGBM LambdaMART model to maximise nDCG. A learning-to-rank approach. |
| **Hybrid-regression** | Takes the same features but trains a LightGBM regression model to predict 6-month ROI. |

**Limitation**: These are two-stage pipelines where the first-stage models are trained independently, then their outputs are combined. The first stage never receives gradient signal from the profitability objective, so it cannot learn representations that are useful for both tasks.

#### Benchmark Results (Paper Table 2)

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

### Key Findings

1. **Price-based models are best at profitability (ROI)**: All three beat the market average. Random Forest is the best overall at ROI (0.0259). However, they fail at predicting what customers will buy, with nDCG barely above random.

2. **Transaction-based models are best at predicting customer preferences (nDCG)**: LightGCN achieves the highest nDCG (0.3404). However, they mostly fail at profitability, as most can't beat the market. The top 10 assets concentrate >50% of all transactions, giving popularity-based approaches a structural advantage in nDCG.

3. **Hybrid models don't clearly dominate**: Neither achieves the best score on either metric.

4. **The two objectives conflict**: Methods optimising one metric tend to perform poorly on the other. **No single algorithm excels at both profitability and preference prediction.** LightGCN has the best nDCG (0.3404) but nearly the worst ROI (0.0004), while Random Forest has the best ROI (0.0259) but low nDCG (0.0237). This is the central finding and the motivation for future work on joint-objective models, which is the gap I hope my approach fills.

## Problem Statement

**Can a single end-to-end model learn to jointly optimise for user interest and asset profitability while achieving strong performance on *both* metrics?**

All 11 baselines treat interactions as static; they ignore the temporal order and spacing of past investments. This project explores whether **sequential Transformer models** can capture these patterns and improve FAR performance.

## Novelty and Research Contributions

- **Extending FAR with attention-based sequential modelling**: All existing FAR baselines treat user-item interactions as static, unordered sets. I explore applying **self-attentive sequential recommendation** (SASRec; Kang & McAuley, 2018) to capture temporal patterns: recency effects, sequential dependencies, and evolving investor preferences. The **order and timing** of past investments carry important signal that static models miss.

- **Time-aware attention for irregularly spaced financial transactions**: Standard sequential models (including SASRec) encode position in the sequence but not **when** those purchases occurred in real time. This is a critical limitation for investment data because:
  - Financial transactions are **highly irregular**: a user's 5th and 6th transactions might be 2 days apart or 18 months apart, and this temporal gap carries important signal.
  - **Market regimes** change over time: purchases made during the 2020 COVID crash have a very different context than purchases during the 2021 bull market, even if they are adjacent in the sequence.
  - I extend SASRec to **TiSASRec** (Li et al., 2020), which injects both *relative* time intervals between consecutive transactions and *absolute* timestamps into the attention mechanism.

- **End-to-end joint interest-profitability optimisation**: The original paper's hybrid baselines are **two-stage pipelines** that first score relevance, then rerank by profitability. This has two key limitations:
  - The CF model is trained to maximise relevance only; it receives no gradient signal from profitability.
  - The reranking model can only reshuffle the CF model's output: it cannot surface assets the CF model missed.
  - I propose a **single, end-to-end dual-head model** that jointly optimises both objectives through shared representations. I describe the full architecture in the [Proposed Approach](#proposed-approach) section.

## Proposed Approach

### Stage 1: SASRec : Self-Attentive Sequential Recommendation

**Core idea**: Instead of treating a user's purchases as an unordered set (as CF methods do), I model them as a **chronologically ordered sequence** and use a Transformer encoder to predict the next asset.

**Architecture**:

```
Input:  [asset_1, asset_2, ..., asset_L]     (user's last L purchases, in order)
|
Asset embedding + Learnable positional embedding
|
N x Transformer blocks (causal masked self-attention + FFN)
|
Output embedding at position L -> dot product with candidate asset embeddings -> ranking scores
```

**Why causal masking?** Each position can only attend to itself and earlier positions (like GPT). This ensures that when predicting the next purchase, the model can only use information from *past* purchases, not future ones.

**Training**: Binary cross-entropy loss. For each position in the sequence, the positive example is the *actual* next item the user purchased, and negative examples are randomly sampled assets the user never bought.

### Stage 2: TiSASRec : Time-Interval-Aware Extension

**Core idea**: I inject information about *when* transactions happened into the attention mechanism, so the model knows that a purchase 3 days ago is more relevant than one 2 years ago.

**What changes from SASRec**:

1. **Relative time intervals**: For each pair of positions (i, j) in the sequence, I compute the time gap in days: `delta_{ij} = |timestamp_i - timestamp_j|`. These are bucketed via log-scale bucketing into `[0, time_bucket_count)` to handle the long tail of time gaps, then each bucket is mapped to a per-head scalar bias via a learnable embedding table.

2. **Absolute time positions**: Days since a reference date are also log-bucketed and embedded into per-head scalar biases.

3. **Modified attention computation**:

```
Attention(Q, K) = softmax( (Q·K^T + R_rel + R_abs) / sqrt(d) )
```

Where `R_rel` and `R_abs` are per-head additive bias matrices looked up from learnable embedding tables indexed by the bucketed relative and absolute time values. The standard attention score `Q·K^T` is augmented with these two bias terms that allow the model to weight attention based on temporal proximity and absolute calendar position.

**Why this matters for financial data**: An investor who made 5 trades in the last week is behaving very differently from one whose 5 trades are spread over 3 years. Standard SASRec cannot distinguish these two users because their *sequences* look identical. However, TiSASRec can, as it encodes the time gaps.

### Stage 3: Hybrid Dual-Head (Interest + Profitability)

**Core idea**: I add a second prediction head to the TiSASRec encoder that predicts asset profitability, and train both heads jointly.

**Architecture**:

```
            User transaction sequence
                  |
            TiSASRec encoder
                  |
            User embedding (h_u)
                  |
      ┌─────────────┴─────────────┐
      |                           |
Interest head                Profitability head
h_u · asset_emb^T           [h_u ; tech_indicators] -> MLP -> ROI_pred
      |                           |
score_interest              score_profit
      |                           |
      └─────────┬─────────────────┘
            |
      alpha * score_interest + (1 - alpha) * score_profit
            |
            Final ranking
```

**Training loss**: `L = L_interest + lambda * L_profit`
- `L_interest`: Binary cross-entropy on next-item prediction (same as SASRec/TiSASRec)
- `L_profit`: Mean squared error between predicted ROI and actual 6-month ROI
- `lambda`: Hyperparameter controlling the trade-off between the two losses

**Inference**: At recommendation time, each candidate asset gets two scores. The final ranking uses `alpha * score_interest + (1 - alpha) * score_profit`, where `alpha` is tunable (or could be learned per user based on their risk profile).

### Our Evaluation Schedule

Our implementation merges the paper's two separate date ranges into one continuous range (Aug 2019 to May 2022, 68 slots) to produce **69 evenly-spaced evaluation points** covering the full dataset period:
- The first time point is **t₀ = August 1, 2019** (providing ~1.5 years of training data from Jan 2018), snapped to the nearest trading day.
- Subsequent time points are spaced approximately **9 trading days** apart (t₁, t₂, ..., t₆₈), with the last point near May 23, 2022.
- At each time point *t*:
  - **Training set**: All pricing data and transactions **before** *t*.
  - **Test set**: All transactions in the 6-month window **(t, t + 6 months)**.
  - Deduplication: If the same (user, asset) pair appears in both train and test, it is kept only in training (to avoid trivial predictions).
  - Filtering: Only users with at least one interaction in *both* train and test are retained. Only assets with pricing data spanning the full test window are retained.

Because time points are ~9 trading days apart but each test window spans ~6 months (~130 trading days), consecutive test windows overlap heavily by design. This sliding-window approach ensures the averaged metrics are not dominated by any single market condition. All metrics are **averaged across all 69 time points**.

## Source Code Architecture

### Module Structure

```
src/
    config/
        settings.py             # Pydantic BaseSettings: hyperparameters for all models
        schemas.py              # Pydantic models: TemporalSplitData, SequenceData, EvaluationResult
    data/
        loading.py              # Load raw CSVs (FAR-Trans: drop zero-price assets, dedup)
        splitting.py            # 69 temporal train/test splits with cumulative construction
        sequences.py            # Chronological user purchase sequences, time bucketing
    features/
        technical_indicators.py # 30-column indicator set (INDICATOR_COLUMNS) with 5-day MA smoothing
    models/
        protocol.py             # Recommender protocol + MODEL_REGISTRY extensibility surface
        train.py                # Shared PyTorch training loop, seed utilities
        random_forest.py        # Price-based RF regressor (non-personalised baseline)
        light_gcn.py            # LightGCN collaborative filtering (PyG LGConv, BPR loss)
        sasrec.py               # Self-attentive sequential recommendation (Transformer encoder)
        tisasrec.py             # Time-interval-aware SASRec extension
        hybrid.py               # Dual-head model: interest + profitability
    evaluation/
        metrics.py              # nDCG@k and ROI@k computation
    pipeline/
        preprocessing.py        # Load raw data, generate splits and sequences, save to disk
        runner.py               # Train and evaluate models via MODEL_REGISTRY
        tuning.py               # Ray Tune grid search around FAR-Trans paper configurations
```

### Configuration

These two files define all the data structures and hyperparameters that every other module depends on.

**`src/config/settings.py`**: All hyperparameters as Pydantic `BaseSettings` classes. The sequential model configs (`SASRecConfig`, `TiSASRecConfig`, `HybridDualHeadConfig`) share the same core Transformer parameters and add model-specific ones. `HybridDualHeadConfig` introduces `loss_lambda` (training trade-off between interest and profit loss) and `inference_alpha` (scoring blend at recommendation time).

**`src/config/schemas.py`**: Three Pydantic models that flow through the entire system:
- `TemporalSplitData`: one train/test split (training interactions, test interactions, eligible users/assets, ID-to-index mappings). This is the central data structure that every model receives
- `SequenceData`: chronologically ordered purchase sequences per user for one split. Only the sequential models (SASRec, TiSASRec, HybridDualHead) use this
- `EvaluationResult`: nDCG and ROI for one model on one split

### Data Pipeline

1. **Loading** (`data/loading.py`): Loads 5 CSVs with date parsing. For close prices, mirrors `data/financial_asset_time_series.py:load` from the [FAR-Trans reference](https://github.com/JavierSanzCruza/far-trans): drops every asset that ever has a zero close price, dedups `(ISIN, timestamp)` keeping the last value, and sorts. Customer and asset CSVs are deduped by keeping the latest timestamp per ID. Note: for pure paper replication only `transactions.csv` and `close_prices.csv` are needed; the other files are loaded for future extensions (e.g., per-user risk conditioning) but are not consumed by any current model.

2. **Temporal Splits** (`data/splitting.py`): Generates 69 evaluation splits by snapping to actual trading days from `close_prices.csv` over a single range defined by `EVALUATION_DATE_RANGE = (2019-08-01, 2022-05-23, 68 slots, 13 future steps)`. The algorithm divides all trading days in that range into evenly spaced slots (~9 trading days apart) and pairs each slot `i` with slot `i+13` as its test-end date (13 steps spans ~6 months). This adapts `data/financial_data_continuous.py:get_dates()` from the [FAR-Trans reference](https://github.com/JavierSanzCruza/far-trans) to a single continuous range. The cold-start positives filter (test items must have appeared in training) is applied unconditionally, mirroring `data/financial_interaction_data.py:106-107` in the same repo. Per-split, three filters run: per-user test-in-train dedup, global train-item filter, and asset eligibility (must have a close price on exactly the recommendation date AND the future date). The core loop in `generate_all_splits` cumulatively builds training interactions via `_add_delta_transactions` (mutates in place), then `copy.deepcopy` snapshots the state for each split.

    Sample `TemporalSplitData` (one of the 69 splits):
    ```python
    TemporalSplitData(
        split_index=0,
        time_point=date(2019, 8, 1),
        test_end=date(2020, 2, 1),
        training_interactions={
            "CUST_001": {"IE00B4L5Y983", "US0378331005"},
            "CUST_002": {"DE0005933931"},
        },
        test_interactions={
            "CUST_001": {"LU0290358497"},        # assets bought in the 6-month test window
            "CUST_002": {"US0378331005"},         # excluding assets already in training
        },
        eligible_customer_ids=["CUST_001", "CUST_002"],
        eligible_asset_ids=["IE00B4L5Y983", "US0378331005", "LU0290358497", "DE0005933931"],
        customer_id_to_index={"CUST_001": 0, "CUST_002": 1},
        asset_id_to_index={"IE00B4L5Y983": 0, "US0378331005": 1, "LU0290358497": 2, "DE0005933931": 3},
    )
    ```

3. **Sequences** (`data/sequences.py`): Builds chronologically ordered purchase sequences per user including repeat purchases, plus time bucketing utilities for the TiSASRec / HybridDualHead extensions. Not used by the RF or LightGCN baselines. Five utilities: `build_user_sequences` (chronological `(asset_id, date)` pairs), `truncate_sequences` (keep last N items), `compute_relative_time_intervals` (days between consecutive purchases), `compute_absolute_positions` (days since a reference date), and `bucket_time_values` (log-scale bucketing to handle the long tail of time gaps).

    Sample `SequenceData` (one split's worth of user sequences):
    ```python
    SequenceData(
        split_index=0,
        time_point=date(2019, 8, 1),
        user_sequences={
            "CUST_001": [
                ("IE00B4L5Y983", date(2018, 3, 15)),
                ("US0378331005", date(2018, 9, 22)),
                ("IE00B4L5Y983", date(2019, 1, 10)),   # repeat purchases preserved
            ],
            "CUST_002": [
                ("DE0005933931", date(2019, 5, 3)),
            ],
        },
    )
    ```

4. **Technical Indicators** (`features/technical_indicators.py`): Computes the `full_short` indicator set the FAR-Trans paper uses for its RF baseline. `INDICATOR_COLUMNS` defines the canonical 30 columns across three rolling horizons (21d, 63d, 126d): ROI, volatility, average price, Sharpe ratio, momentum, rate of change, EMA; plus single-horizon MACD (12/26), RSI-14, DCO-22, and rolling min/max. `DEFAULT_PERIODS` is `(21, 63, 126, 189)`, meaning the DataFrame also computes 189d variants of the period-based indicators beyond the 30 canonical columns. After computing all indicators, a 5-day moving-average smoothing pass is applied to every numeric column, followed by per-asset `dropna()`. This matches `algorithms/kpi_gen/indicators.py` + `algorithms/kpi_gen/ma_kpi_generator.py` from the [FAR-Trans reference](https://github.com/JavierSanzCruza/far-trans) formula-by-formula, and `recommendation.py:85-94` column-by-column in identical order for the 30 canonical columns. `compute_all_indicators` is a helper that zero-fills missing-asset rows at lookup time; it is only used by HybridDualHead while the RF baseline queries the indicator DataFrame directly.

    | Indicator family | Horizons | Formula |
    |---|---|---|
    | `avg_price` | DEFAULT_PERIODS | `close.rolling(h).mean()` |
    | `past_profitability` | `(1,) + DEFAULT_PERIODS` | `(close - close.shift(h)) / close.shift(h)` |
    | `volatility` | DEFAULT_PERIODS | `std(daily_return, h) * sqrt(252)` |
    | `sharpe` | DEFAULT_PERIODS | `past_profitability_h / volatility_h` (Inf/NaN -> 0) |
    | `m` (momentum) | DEFAULT_PERIODS | `close.diff(h)` |
    | `roc` (rate of change) | DEFAULT_PERIODS | `m_h / close.shift(h)` |
    | `min`, `max` | DEFAULT_PERIODS | `close.rolling(h).{min,max}()` |
    | `exp_mean` | DEFAULT_PERIODS | `close.ewm(span=h).mean()` |
    | `MACD` | single | `EMA(close, 12) - EMA(close, 26)` (adjust=False) |
    | `rsi_14` | single | Wilder's RSI via EMA(span=14, adjust=False) |
    | `dco_22` | single | `close.shift(12) - close.rolling(22).mean()` |

### Models

All five models implement the `Recommender` protocol and are discovered via `MODEL_REGISTRY` in `src/models/protocol.py`:

```python
class Recommender(Protocol):
    @property
    def name(self) -> str: ...
    def train_on_split(self, split: TemporalSplitData, **kwargs: object) -> None: ...
    def recommend_for_user(self, user_id: str, excluded_assets: set[str], k: int = 10) -> list[str]: ...
```

`MODEL_REGISTRY` is the single source of truth for model metadata. Each entry declares its config class, data dependencies (`needs_indicators`, `needs_sequences`, `needs_close_prices`), and factory. Adding a new model is one registry entry plus the `Recommender` implementation:

```python
MODEL_REGISTRY["my_new_model"] = ModelEntry(
    model_name="my_new_model",
    config_class=MyNewModelConfig,
    needs_indicators=True,
    needs_sequences=False,
    needs_close_prices=False,
)
```

| Model | Category | Data Source | Key Mechanism |
|---|---|---|---|
| **Random Forest** | Price-based | Technical indicators | sklearn regressor predicts forward 126-trading-day ROI; ranks all assets identically for every user |
| **LightGCN** | Transaction-based | User-asset bipartite graph | PyTorch Geometric `LGConv` (canonical symmetric `D^{-1/2}AD^{-1/2}`, no self-loops); BPR loss with L2 on initial embeddings; dot-product scoring |
| **SASRec** | Sequential | Chronological purchase sequences | Transformer encoder with causal masking; BCE loss on next-item prediction |
| **TiSASRec** | Sequential + temporal | Sequences + timestamps | Extends SASRec with relative/absolute time-interval embeddings in attention |
| **HybridDualHead** | Sequential + price | Sequences + timestamps + indicators | Extends TiSASRec with MLP profitability head; combined BCE + MSE loss; alpha-weighted scoring |

**`src/models/train.py`**: Two shared utilities: `set_random_seeds` (reproducibility across numpy, torch, Python random) and `train_pytorch_model` (generic training loop: model, dataset, loss function, optimizer, epochs). Used by SASRec, TiSASRec, and HybridDualHead. LightGCN has its own training loop because it needs the edge index.

**Note on LightGCN fidelity.** The [FAR-Trans paper](https://github.com/JavierSanzCruza/far-trans) runs LightGCN through the [Beta-RecSys library](https://github.com/beta-team/community/blob/master/beta_recsys/README.md) (`beta_rec/models/lightgcn.py` + `beta_rec/data/base_data.py:337-360`), which applies NGCF-style asymmetric `D^{-1}(A + I)` normalization with added self-loops. That is not the LightGCN of He et al. 2020 (Eq. 3), which specifies symmetric `D^{-1/2}AD^{-1/2}` without self-loops. Our implementation uses PyTorch Geometric's `LGConv`, which is the canonical paper formulation; Beta-RecSys's implementation is a bug. Our reported LightGCN numbers are therefore the correct baseline and may differ from the paper's 0.3404 in either direction. We also deliberately do not track a "best validation-epoch checkpoint": Beta-RecSys selects the best epoch on a validation set that FAR-Trans sets equal to the test set, which is leaky. We use last-epoch weights for honest inference.

**Sequential model details.** The three sequential models form an inheritance chain. Each extends the previous one:

**`src/models/sasrec.py`** (base sequential Transformer):
- `SASRecModel` (nn.Module): Transformer encoder with causal masking. `_compute_attention_scores(Q, K)` is the override point (plain `Q * K^T`). `predict` runs the forward pass, takes the last position's hidden state, and dot-products it with candidate embeddings
- `TransformerBlock` (nn.Module): multi-head self-attention with an injectable `attention_score_fn`, feed-forward network with GELU, two residual connections with LayerNorm
- `SASRecDataset`: converts asset sequences to indices (offset by +1 because 0 is padding), next-item prediction targets, random negative sampling, left-padding
- `SASRecRecommender`: `train_on_split` receives user sequences via kwargs, truncates, builds dataset, trains with BCE loss. `recommend_for_user` left-pads the user's sequence, runs the model, returns top-k

**`src/models/tisasrec.py`** (extends SASRec with time-interval awareness):
- `TiSASRecModel(SASRecModel)`: adds `relative_time_embedding` and `absolute_time_embedding` tables. Overrides `_compute_attention_scores` to add per-head time-interval biases to the base `Q * K^T` scores
- `TiSASRecDataset(SASRecDataset)`: extends the base dataset to also return relative and absolute time matrices (seq_len x seq_len), both left-padded with zeros
- `TiSASRecRecommender(SASRecRecommender)`: overrides `_build_model`, `_build_dataset`, `train_on_split`, `recommend_for_user`, and adds `_build_inference_time_matrices`

**`src/models/hybrid.py`** (dual-head: interest + profitability):
- `HybridDualHeadModel(TiSASRecModel)`: adds `profitability_head` (MLP taking `[user_hidden_state ; technical_indicators]` and outputting a scalar ROI prediction)
- `HybridDualHeadRecommender(TiSASRecRecommender)`: overrides `train_on_split` (combined loss: `interest_loss + lambda * profit_loss` where interest is BCE and profit is MSE), `_build_indicator_and_roi_tensors` (pre-computes indicator features and ROI targets), and `recommend_for_user` (min-max normalizes interest and profitability scores, then blends: `alpha * interest + (1-alpha) * profit`)

```
nn.Module hierarchy:
    SASRecModel
        |--- _compute_attention_scores (Q*K^T)
        |--- TransformerBlock (multi-head attention + FFN)
        |
        TiSASRecModel(SASRecModel)
            |--- overrides _compute_attention_scores (adds time biases)
            |--- adds relative_time_embedding, absolute_time_embedding
            |
            HybridDualHeadModel(TiSASRecModel)
                |--- adds profitability_head (MLP)
                |--- adds predict_profitability method

Recommender hierarchy:
    SASRecRecommender
        |--- _build_model       -> SASRecModel
        |--- _build_dataset     -> SASRecDataset
        |--- train_on_split     (BCE loss)
        |--- recommend_for_user (dot product scoring)
        |
        TiSASRecRecommender(SASRecRecommender)
            |--- _build_model   -> TiSASRecModel
            |--- _build_dataset -> TiSASRecDataset
            |--- train_on_split (BCE loss + time matrices)
            |--- recommend_for_user (dot product + time matrices)
            |
            HybridDualHeadRecommender(TiSASRecRecommender)
                |--- _build_model   -> HybridDualHeadModel
                |--- train_on_split (BCE + lambda * MSE loss)
                |--- recommend_for_user (alpha * interest + (1-alpha) * profit)
```

### Evaluation

- **nDCG@k**: binary relevance (1 if the user acquires the asset in the 6-month test window, 0 otherwise); IDCG is capped at `min(k, num_relevant)`; users with no relevant items contribute 0. Matches `metrics/pure_ndcg.py` from the [FAR-Trans reference](https://github.com/JavierSanzCruza/far-trans) (base-invariant: ours uses `log2`, theirs uses natural log, the ratio is identical).
- **ROI@k**: geometric monthly return `(1 + total_return)^(30/days) - 1` per recommended asset, averaged across the top-k list. Missing-price recommendations are imputed as 0 return (not skipped), matching `metrics/kpi_monthly_evaluation_metric.py` and `metrics/kpi_evaluation_metric.py:30-32` in the same repo. Calendar-day horizon between recommendation date and test-end date.
- **`build_price_lookup`**: finds the closest available price on or before the time point and test end for each asset. Since all split dates are snapped to actual trading days, the `<=` fallback collapses to exact-date lookup.
- **`evaluate_model_on_split`**: iterates `split.eligible_customer_ids` (users in both train and test), averages both metrics across users.
- Both metrics are averaged across all eligible users and then across all 69 temporal splits.

### Hyperparameter Tuning

The tuning pipeline (`pipeline/tuning.py`) uses Ray Tune's native grid search via `tune.grid_search(...)`. Each model declares a `ModelTuningSpec` with a small grid centered on each model's reference configuration so that the reference hyperparameters are always one of the trial points. Evaluation is on 3 validation splits at fixed dates (2019-04-01, 2019-10-01, 2020-01-31), snapped to the nearest trading day. Only the first validation date (2019-04-01) precedes the first evaluation split (2019-08-01); the other two fall within the evaluation period. Because models are retrained from scratch for every evaluation split, the validation step selects hyperparameters but does not leak trained weights into Table 2. Best configs are saved to a timestamped JSON directory under `outputs/configs/` and loaded by the runner via `--config`.

#### Trial Concurrency and Resource Allocation

Ray Tune assigns resources per trial based on each model's compute profile, not a single blanket rule:

- **Random Forest** always claims `{"cpu": 1}` regardless of `--device`, because sklearn has no GPU backend. With `max_concurrent_trials=3`, all 3 RF trials run in parallel on the CPU pool.
- **LightGCN, SASRec, TiSASRec, HybridDualHead** claim `{"gpu": 1, "cpu": 1}` when `--device cuda` is passed. `max_concurrent_trials=1` for all four because a single physical GPU can only satisfy one `{"gpu": 1}` request at a time; setting it higher has no effect on a single-GPU node.

The single-GPU cluster allocation (1 GPU, 4 CPUs, 16 GB RAM on the SMU `msc` partition) is the reason for this split. If you run on a multi-GPU node you can raise `max_concurrent_trials` for the Transformer models to match the GPU count, or experiment with fractional `{"gpu": 0.5}` allocation to share a GPU between two small SASRec/TiSASRec trials (expected gain is modest at this model size because PyTorch serialises kernels on the default CUDA stream).

#### Grid Search Rationale

**Random Forest** (3 trials): The FAR-Trans reference code hardcodes `n_estimators=20` with no documented tuning. The grid `[20, 50, 75]` keeps 20 as the paper anchor and searches moderately larger ensembles.

**LightGCN** (32 trials): Grid is centered on the FAR-Trans reference config (`emb_dim=64`, `lr=0.01`, `keep_prob=0.6`, `weight_decay=1e-5`, `epochs=50`) and searches one alternative value for embedding size, layers, learning rate, weight decay, and keep probability.

**SASRec** (8 trials): Grid is informed by the original SASRec paper (Kang & McAuley, ICDM 2018) whose reference implementation uses `emb_dim=50`, `num_heads=1`, `maxlen=50`, `dropout=0.5`, `lr=0.001`. Key choices:
- `embedding_dimension=[64]`: fixed at 64 (the nearest power-of-2 to the paper's 50 that stays divisible by both 1 and 2 heads)
- `number_of_attention_heads=[1, 2]`: the original paper uses 1 head; our default of 2 is searched against the paper value
- `max_sequence_length=[50]`: fixed at the paper value; FAR-Trans users are sparse (>50% have 3 or fewer transactions), so longer sequences add compute without signal
- `dropout_rate=[0.2, 0.5]`: the original SASRec paper uses 0.5; higher dropout is important for sparse financial interaction data
- `number_of_blocks=[1, 2]`: paper uses 2; 1 is included as a cheaper option for sparse data

**TiSASRec** (8 trials): Grid follows the original TiSASRec paper (Li, Wang & McAuley, WSDM 2020) whose reference implementation uses `emb_dim=50`, `num_heads=1`, `maxlen=50`, `dropout=0.2`, `time_span=256`, `lr=0.001`. Key choices:
- `number_of_attention_heads=[1, 2]`: same rationale as SASRec
- `number_of_blocks=[1, 2]`: same rationale as SASRec; paper uses 2 and 1 is included as a cheaper option for sparse data
- `dropout_rate=[0.2]`: fixed at the TiSASRec paper value (lower than SASRec because time embeddings provide additional regularisation)
- `time_bucket_count=[64, 256]`: the original paper searched `{1, 64, 256, 1024, 2048}`; 256 is the paper default and 64 is included because FAR-Trans transactions are infrequent (many users trade only a few times per year), making coarser time buckets potentially better
- `max_sequence_length=[50]`: fixed for the same reason as SASRec

**HybridDualHead** (18 trials): Transformer backbone is fixed at the TiSASRec default config (update these values after TiSASRec tuning if the best config differs). The search targets only the novel dual-head parameters:
- `profitability_hidden_dimension=[32, 64]`: searches a smaller vs. larger MLP for the profitability head
- `loss_lambda=[0.1, 0.5, 1.0]`: controls the training trade-off between interest loss (BCE) and profit loss (MSE); 0.5 is the symmetric midpoint anchor
- `inference_alpha=[0.3, 0.5, 0.7]`: controls the scoring blend at recommendation time; anchored at 0.5 (equal weight) and searched asymmetrically towards interest (0.7) and profit (0.3)

### Pipeline Orchestration

**`src/pipeline/preprocessing.py`** (Step 0): The entry point for data preparation. `run_preprocessing` loads all raw CSVs, generates 69 evaluation splits and 3 validation splits, builds user purchase sequences for each split, and saves everything to disk as JSON files. Also provides loader functions (`load_evaluation_splits`, `load_validation_splits`, `load_evaluation_sequences`, etc.) that all downstream modules use.

**`src/pipeline/runner.py`** (Step 2): The main experiment loop. `run_all_experiments` loads evaluation splits, close prices, and sequences from disk, resolves the model set via `MODEL_REGISTRY` (optionally filtered by `--models`), and for each model: constructs the recommender via `build_recommender(name, config=provided or paper-default)`, then for each split calls `model.train_on_split`, `generate_recommendations`, and `evaluate_model_on_split`. The runner is data-driven: loop bodies look up `MODEL_REGISTRY[name].needs_indicators` / `needs_sequences` to decide which auxiliary data to load, so adding a new model to the registry automatically makes it runnable without editing runner code.

```sh
uv run poe preprocess              # Step 0: generate all splits to data/splits/
uv run poe tune --models random_forest light_gcn   # Step 1: hyperparameter search
uv run poe run --config outputs/configs/.../best_hyperparameters.json  # Step 2: evaluate
uv run poe run --models random_forest light_gcn    # or skip tuning, use paper defaults
```

### Data Flow Summary

```
Raw CSVs (transactions, close_prices, customer_info, asset_info, markets)
    |
    | loading.py
    v
DataFrames
    |
    | splitting.py                          sequences.py
    v                                       v
69 TemporalSplitData                        User purchase sequences
(training_interactions,                     (chronological (asset_id, date) pairs)
 test_interactions,
 eligible users/assets,                     technical_indicators.py
 ID-to-index mappings)                      v
    |                                       Indicator features per asset
    |
    +---> RandomForestBaseline              (uses indicators + price lookup for ROI targets)
    +---> LightGCNBaseline                  (uses interaction graph from training_interactions)
    +---> SASRecRecommender                 (uses purchase sequences)
    +---> TiSASRecRecommender               (uses sequences + timestamps)
    +---> HybridDualHeadRecommender         (uses sequences + timestamps + indicators + ROI targets)
    |
    | recommend_for_user per eligible user
    v
Recommendations: dict[user_id, list[asset_id]]
    |
    | metrics.py
    v
EvaluationResult (nDCG@10, ROI@10)
```

### Preprocessed Data Layout

```
data/splits/
    metadata.json
    evaluation/                 # 69 TemporalSplitData JSON files
        split_000.json ... split_068.json
    validation/                 # 3 validation split JSON files
        split_000.json ... split_002.json
    sequences/                  # Chronological user purchase sequences per split
        evaluation/
            split_000.json ... split_068.json
        validation/
            split_000.json ... split_002.json
```

## Working with this Repository

### Prerequisites

- **[uv](https://docs.astral.sh/uv/getting-started/installation/)**: package and dependency manager

### Setup

1. Clone the repository and install all dependencies:

```sh
uv sync
```

2. Install git hooks (runs lint, format, and typecheck before every commit):

```sh
uv run poe setup
```

3. Activate the virtual environment, then install graphify:

```sh
source .venv/bin/activate
pip install graphifyy
```

4. (Optional) If using [Claude Code](https://claude.ai/claude-code), install the graphify Claude integration:

```sh
graphify claude install
```

### Git Hooks

[Lefthook](https://github.com/evilmartians/lefthook) manages all git hooks:

- **Pre-commit**: runs lint, format, and typecheck. The commit is aborted if any check fails.
- **Post-commit**: rebuilds the graphify knowledge graph for changed code files.
- **Post-checkout**: rebuilds the graphify knowledge graph when switching branches.

## GPU Cluster

### Initial Setup

1. Connect via **GlobalVPN** to reach the SMU network.
2. SSH into the cluster: `ssh samuel.sim.2024@origami.smu.edu.sg`
   - A bash alias `gpu` is configured locally to connect via `sshpass`.

### Account Details

```
Account name                    : msc
Partition                       : msc
QOS                             : studentqos
Scratch directory               : /common/scratch/users/s/samuel.sim.2024/
Home directory quota            : 120GB
Scratch file quota              : 100,000 files
```

#### Resource Limits (studentqos)

| Resource              | Limit       |
|-----------------------|-------------|
| Max running jobs      | 2           |
| Max submitted jobs    | 4           |
| CPUs per job          | 4           |
| GPUs per job          | 1           |
| RAM per job           | 32 GB       |
| Max job time          | 5 days      |

### File Transfers (SCP)

Run these from your **local** machine.

```bash
# Local file to cluster
scp /path/to/file samuel.sim.2024@origami.smu.edu.sg:~/path/to/destination

# Local folder to cluster
scp -r /path/to/folder samuel.sim.2024@origami.smu.edu.sg:~/path/to/destination

# Cluster file to local
scp samuel.sim.2024@origami.smu.edu.sg:~/path/to/file /path/to/destination

# Cluster folder to local
scp -r samuel.sim.2024@origami.smu.edu.sg:~/path/to/folder /path/to/destination
```

### Job Submission

#### Shell Script Template

The sbatch template is at [`scripts/far-tuning.sh`](scripts/far-tuning.sh), pre-configured with the `msc` account details. To create a new job script, copy it and update:

- `--job-name` : a descriptive name for the job
- `--time` : wall-clock limit (format: `DD-HH:MM:SS`)
- `--mem` : memory allocation (up to 32 GB)
- The `srun` command at the bottom with your actual script path

#### Submitting

```bash
chmod +x job.sh
sbatch job.sh
```

An output log will appear in the working directory set in the script. Email notifications are sent when the job starts, completes, or fails based on script.

### GPU Monitoring

Add `srun whichgpu` before your main workload in the batch script to log which GPU and compute node you were assigned. Then check utilization on the [Grafana dashboard](https://green.smu.edu.sg/gpustats) (requires SMU network/VPN), selecting the matching node and GPU number.

### Useful Commands

```bash
myinfo                  # Account details, quotas, and partition info
myqueue                 # Status of your current jobs
myjob <jobid>           # Detailed info on a running/recent job (last 5 min)
mypastjob <days>        # Job history for the past N days (max 30)
```

## References

- Sanz-Cruzado, J., Droukas, N., & McCreadie, R. (2024). *FAR-Trans: An Investment Dataset for Financial Asset Recommendation*. arXiv:2407.08692.
- Kang, W.-C., & McAuley, J. (2018). *Self-Attentive Sequential Recommendation*. IEEE ICDM 2018.
- Li, J., Wang, Y., & McAuley, J. (2020). *Time Interval Aware Self-Attention for Sequential Recommendation*. WSDM 2020.
