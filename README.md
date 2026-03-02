# SMU Capstone: Temporal User Modelling for Financial Asset Recommendation

A sequential Transformer-based recommender for the FAR-Trans dataset that models temporal dynamics of investor behaviour and combines predicted user interest with asset profitability.

## Table of Contents

1. [Paper Summary: FAR-Trans](#paper-summary-far-trans-sanz-cruzado-et-al-2024)
2. [Problem Statement](#problem-statement)
3. [Novelty and Research Contributions](#novelty-and-research-contributions)
4. [Proposed Approach](#proposed-approach)
5. [Source Code Architecture](#source-code-architecture)
6. [Pipeline Workflow](#pipeline-workflow)
7. [Working with this Repository](#working-with-this-repository)
8. [References](#references)


## Paper Summary: FAR-Trans

This section summarises the FAR-Trans paper that forms the foundation of this project. It is intended as my self-contained reference of the entire paper.

I have also taken the liberty of converting the research paper into `markdown` format from `TeX` so that large language modelling tools can digest its context more easily compared to `pdf` format, which serves as a good grounding context for me to ask questions on it.

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

**Data Loading:** The FAR-Trans dataset is loaded and explored in [`notebooks/data.ipynb`](notebooks/data.ipynb). This notebook handles download, extraction, and initial data inspection.

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

- **Not personalised** with same predictions for all customers
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

The dataset is split into **69 temporal evaluation points**:
- The first time point is **t₀ = August 1, 2019** (providing ~1.5 years of training data from Jan 2018), snapped to the nearest trading day.
- Subsequent time points are spaced approximately **9 trading days** apart (t₁, t₂, ..., t₆₈), with the last point near May 23, 2022.
- At each time point *t*:
- **Training set**: All pricing data and transactions **before** *t*.
- **Test set**: All transactions in the 6-month window **(t, t + 6 months)**.
- Deduplication: If the same (user, asset) pair appears in both train and test, it is kept only in training (to avoid trivial predictions).
- Filtering: Only users with at least one interaction in *both* train and test are retained. Only assets with pricing data spanning the full test window are retained.

All metrics are **averaged across all 69 time points**, providing a robust estimate of model performance across different market conditions.

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

3. Convert to a monthly average return:

```
Monthly_Return(i) = Return(i) / 6
```

4. ROI@10 is the average monthly return across all 10 recommended assets (equally weighted portfolio):

```
ROI@10 = (1/10) * Σ Monthly_Return(asset_i)   for i = 1 to 10
```

5. This is then averaged across all users and all 69 time points.

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
| **ItemKNN (UB kNN)** | User-Based k-Nearest Neighbours. Finds users with similar purchase histories, then recommends assets those similar users bought. |
| **ARM (Apriori)** | Association Rule Mining. Discovers rules like "users who bought asset A tend to also buy asset B" and uses these rules to generate recommendations. |

**Limitation**: These models have no awareness of asset prices or returns. They optimise purely for predicting user purchases, which is why they achieve high nDCG but near-zero ROI.

#### Hybrid Models (Two-Stage Pipelines)

These attempt to combine both signals by running a transaction-based model first, then using its output as features for a profitability model:

| Model | How it works |
|---|---|
| **Hybrid-nDCG** | Takes the recommendation scores from *all* of the above models as features, then trains a LightGBM LambdaMART model to maximise nDCG. A learning-to-rank approach. |
| **Hybrid-regression** | Takes the same features but trains a LightGBM regression model to predict 6-month ROI. |

**Limitation**: These are two-stages where the first-stage models are trained independently, then their outputs are combined. The first stage never receives gradient signal from the profitability objective, so it cannot learn representations that are useful for both tasks.

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
  - I propose a **single, end-to-end dual-head neural model** that jointly optimises both objectives through shared representations. I describe the full architecture in the [Proposed Approach](#proposed-approach) section.

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

I will need to think of the hyperparameters.

### Stage 2: TiSASRec : Time-Interval-Aware Extension

**Core idea**: I inject information about *when* transactions happened into the attention mechanism, so the model knows that a purchase 3 days ago is more relevant than one 2 years ago.

**What changes from SASRec**:

1. **Relative time intervals**: For each consecutive pair of transactions, I compute the time gap in days: `delta_k = timestamp_{k+1} - timestamp_k`. These are bucketed into bins (e.g., [0, 1, 2, 3, 7, 14, 30, 90, 180, 365+]) to handle long tails and embedded via a learnable lookup table.

2. **Absolute time positions**: Days since a reference date are also bucketed and embedded.

3. **Modified attention computation**:

```
Attention(Q, K) = softmax( (Q·K^T + Q·K_time_rel^T + K_time_abs·Q^T) / sqrt(d) )
```

The standard attention score `Q·K^T` is augmented with two additional terms that allow the model to weight attention based on temporal proximity and absolute calendar position.

**Why this matters for financial data**: An investor who made 5 trades in the last week is behaving very differently from one whose 5 trades are spread over 3 years. Standard SASRec cannot distinguish these two users because their *sequences* look identical. However, TiSASRec can as it encodes the time gaps.

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
        technical_indicators.py # 30-column full_short indicator set with 5-day MA smoothing
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

### Data Pipeline

1. **Loading** (`data/loading.py`): Loads 6 CSVs with date parsing. For close prices, mirrors `data/financial_asset_time_series.py:load` from the [FAR-Trans reference](https://github.com/JavierSanzCruza/far-trans): drops every asset that ever has a zero close price, dedups `(ISIN, timestamp)` keeping the last value, and sorts. Customer and asset CSVs are deduped by keeping the latest timestamp per ID. Note: for pure paper replication only `transactions.csv` and `close_prices.csv` are needed; the other files are loaded so the dual-head extension can reuse them for per-user risk conditioning.

3. **Temporal Splits** (`data/splitting.py`): Generates 69 evaluation splits by snapping to actual trading days from `close_prices.csv` over a single range defined by `EVALUATION_DATE_RANGE = (2019-08-01, 2022-05-23, 68 slots, 13 future steps)`. The algorithm divides all trading days in that range into evenly spaced slots (~9 trading days apart) and pairs each slot `i` with slot `i+13` as its test-end date (13 steps spans ~6 months). This adapts `data/financial_data_continuous.py:get_dates()` from the [FAR-Trans reference](https://github.com/JavierSanzCruza/far-trans) to a single continuous range. The cold-start positives filter (test items must have appeared in training) is applied unconditionally, mirroring `data/financial_interaction_data.py:106-107` in the same repo. Per-split, three filters run: per-user test-in-train dedup, global train-item filter, and asset eligibility (must have a close price on exactly the recommendation date AND the future date).

4. **Sequences** (`data/sequences.py`): Builds chronologically ordered purchase sequences per user including repeat purchases, plus time bucketing utilities for the TiSASRec / HybridDualHead extensions. Not used by the RF or LightGCN baselines.

5. **Technical Indicators** (`features/technical_indicators.py`): Computes the 30-column `full_short` indicator set the FAR-Trans paper uses for its RF baseline, across three rolling horizons (21d, 63d, 126d): ROI, volatility, average price, Sharpe ratio, momentum, rate of change, EMA; plus single-horizon MACD (12/26), RSI-14, DCO-22, and rolling min/max. After computing all indicators, a 5-day moving-average smoothing pass is applied to every numeric column, followed by per-asset `dropna()`. This matches `algorithms/kpi_gen/indicators.py` + `algorithms/kpi_gen/ma_kpi_generator.py` from the [FAR-Trans reference](https://github.com/JavierSanzCruza/far-trans) formula-by-formula, and `recommendation.py:85-94` column-by-column in identical order.

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

**Note on LightGCN fidelity.** The [FAR-Trans paper](https://github.com/JavierSanzCruza/far-trans) runs LightGCN through the [Beta-RecSys library](https://github.com/beta-team/community/blob/master/beta_recsys/README.md) (`beta_rec/models/lightgcn.py` + `beta_rec/data/base_data.py:337-360`), which applies NGCF-style asymmetric `D^{-1}(A + I)` normalization with added self-loops. That is not the LightGCN of He et al. 2020 (Eq. 3), which specifies symmetric `D^{-1/2}AD^{-1/2}` without self-loops. Our implementation uses PyTorch Geometric's `LGConv`, which is the canonical paper formulation; Beta-RecSys's implementation is a bug. Our reported LightGCN numbers are therefore the correct baseline and may differ from the paper's 0.3404 in either direction. We also deliberately do not track a "best validation-epoch checkpoint": Beta-RecSys selects the best epoch on a validation set that FAR-Trans sets equal to the test set, which is leaky. We use last-epoch weights for honest inference.

The sequential models form an inheritance chain:
- `SASRecModel` provides the base Transformer encoder with a `_compute_attention_scores(Q, K)` override point
- `TiSASRecModel(SASRecModel)` overrides attention to add time-interval bias terms
- `HybridDualHeadModel(TiSASRecModel)` adds the profitability MLP head

The recommender classes mirror the same hierarchy:
- `SASRecRecommender` provides `_build_model()` and `_build_dataset()` as override points
- `TiSASRecRecommender(SASRecRecommender)` overrides both to use time-aware variants
- `HybridDualHeadRecommender(TiSASRecRecommender)` overrides `train_on_split()` (combined loss) and `recommend_for_user()` (alpha-weighted scoring)

### Evaluation

- **nDCG@k**: binary relevance (1 if the user acquires the asset in the 6-month test window, 0 otherwise); IDCG is capped at `min(k, num_relevant)`; users with no relevant items contribute 0. Matches `metrics/pure_ndcg.py` from the [FAR-Trans reference](https://github.com/JavierSanzCruza/far-trans) (base-invariant: ours uses `log2`, theirs uses natural log, the ratio is identical).
- **ROI@k**: geometric monthly return `(1 + total_return)^(30/days) - 1` per recommended asset, averaged across the top-k list. Missing-price recommendations are imputed as 0 return (not skipped), matching `metrics/kpi_monthly_evaluation_metric.py` and `metrics/kpi_evaluation_metric.py:30-32` in the same repo. Calendar-day horizon between recommendation date and test-end date.
- Both metrics are averaged across all eligible users (users with interactions in both train and test) and then across all 69 temporal splits.

### Hyperparameter Tuning

The tuning pipeline (`pipeline/tuning.py`) uses Ray Tune's native grid search via `tune.grid_search(...)`. Each model declares a `ModelTuningSpec` with a small grid **centered on the FAR-Trans paper configuration** so that the paper's exact hyperparameters are always one of the trial points. Evaluation is on 3 validation splits at fixed dates (2019-04-01, 2019-10-01, 2020-01-31), snapped to the nearest trading day. All three validation dates precede the first evaluation split (2019-08-01), preventing leakage into Table 2. Best configs are saved to a timestamped JSON directory under `outputs/configs/` and loaded by the runner via `--config`.

To reproduce the paper baselines without tuning, omit `--config` entirely and the runner falls back to the defaults declared in `MODEL_REGISTRY`, which match the paper:

```sh
uv run poe run --models random_forest light_gcn
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

## Pipeline Workflow

All pipeline steps are available as [poe](https://poethepoet.natn.io/) tasks. Run `uv run poe --help` to see all available tasks and their arguments.

```sh
# Step 0: Preprocess (run once, generates all splits to disk at data/splits/)
uv run poe preprocess

# Step 1: Tune hyperparameters (reads validation splits from disk)
uv run poe tune

# Step 1a: Tune only baseline models on CPU
uv run poe tune --models random_forest light_gcn --device cpu

# Step 2: Train and evaluate models (reads evaluation splits from data/splits/)
uv run poe run

# Step 2a: Evaluate only baseline models on CPU
uv run poe run --models random_forest light_gcn --device cpu

# Step 2b: Evaluate with tuned configs
uv run poe run --config outputs/configs/20260404_143022/best_hyperparameters.json
```

All pipeline outputs are saved under `outputs/`:

```
outputs/
    configs/                        # Tuned hyperparameters (one dir per run, timestamped)
        20260404_143022/
            best_hyperparameters.json
    results/                        # Timestamped evaluation CSVs
        tuning/                     # Validation results from tuning
            random_forest/
                20260404_143022.csv
            light_gcn/
                20260404_144531.csv
        evaluation/                 # Evaluation results from runner
            random_forest/
                20260404_150112.csv
            light_gcn/
                20260404_151843.csv
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

### Development Tasks

All tasks are run via [poethepoet](https://poethepoet.natn.io/). Run the following to see all available tasks:

```sh
uv run poe --help
```

### Git Hooks

[Lefthook](https://github.com/evilmartians/lefthook) manages the pre-commit hook. After running `uv run poe setup` once, every `git commit` will automatically run lint, format, and typecheck. The commit is aborted if any check fails.

## References

- Sanz-Cruzado, J., Droukas, N., & McCreadie, R. (2024). *FAR-Trans: An Investment Dataset for Financial Asset Recommendation*. arXiv:2407.08692.
- Kang, W.-C., & McAuley, J. (2018). *Self-Attentive Sequential Recommendation*. IEEE ICDM 2018.
- Li, J., Wang, Y., & McAuley, J. (2020). *Time Interval Aware Self-Attention for Sequential Recommendation*. WSDM 2020.

