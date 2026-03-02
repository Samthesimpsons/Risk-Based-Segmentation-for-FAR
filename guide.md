# Code Reading Guide

## Project at a Glance

Five models are implemented, forming a progression:

1. **Random Forest**: price-based baseline (same recommendations for every user)
2. **LightGCN**: graph-based collaborative filtering (personalised but ignores price)
3. **SASRec**: sequential Transformer that models the *order* of purchases
4. **TiSASRec**: extends SASRec with *time-interval awareness*
5. **HybridDualHead**: extends TiSASRec with a second head for profitability prediction

The Random Forest and LightGCN implementations are verified line-for-line against the [FAR-Trans paper reference](https://github.com/JavierSanzCruza/far-trans) and, for LightGCN, the upstream [Beta-RecSys library](https://github.com/beta-team/community/blob/master/beta_recsys/README.md) it delegates to. One intentional deviation: we use the canonical LightGCN of He et al. 2020 (symmetric `D^{-1/2}AD^{-1/2}`, no self-loops) via PyTorch Geometric's `LGConv`, while the paper's reference uses Beta-RecSys's buggy `D^{-1}(A+I)` asymmetric normalization with self-loops. See `src/models/light_gcn.py` module docstring for the detailed rationale.

## Reading Order

### Phase 1: Configuration (read first, reference throughout)

These two files define all the data structures and hyperparameters that every other module depends on. Read them once, then refer back as needed.

**1. `src/config/settings.py`**

All hyperparameters as Pydantic `BaseSettings` classes. Start here to understand what knobs exist for each model. Key things to notice:
- The sequential model configs (`SASRecConfig`, `TiSASRecConfig`, `HybridDualHeadConfig`) share the same core Transformer parameters and add model-specific ones
- `HybridDualHeadConfig` introduces `loss_lambda` (training trade-off between interest and profit loss) and `inference_alpha` (scoring blend at recommendation time)

**2. `src/config/schemas.py`**

Three Pydantic models that flow through the entire system:
- `TemporalSplitData`: one train/test split (training interactions, test interactions, eligible users/assets, ID-to-index mappings). This is the central data structure that every model receives
- `SequenceData`: chronologically ordered purchase sequences per user for one split. Only the sequential models (SASRec, TiSASRec, HybridDualHead) use this
- `EvaluationResult`: nDCG and ROI for one model on one split

### Phase 2: Data Pipeline (how raw CSVs become model inputs)

Read these in order. Each module produces the input for the next.

**3. `src/data/loading.py`**

Straightforward CSV loading.

**5. `src/data/splitting.py`**

This is the most important data module. It generates the 69 temporal evaluation splits that the entire experiment runs on. Read this carefully.

The schedule follows the same algorithm as `data/financial_data_continuous.py:get_dates()` from the [FAR-Trans reference](https://github.com/JavierSanzCruza/far-trans), adapted to a single range covering our full dataset.

The constant `EVALUATION_DATE_RANGE` defines the range: `(2019-08-01, 2022-05-23, 68 splits, 13 future steps)`. The algorithm collects all actual trading days in that range from `close_prices.csv`, divides them into evenly spaced slots (approximately 9 trading days apart), and pairs each slot `i` with slot `i+13` as its test-end date (13 steps spans approximately 6 months on this trading calendar).

This produces 69 splits total (68 + 1 from the force-appended last date). Every split date is a real trading day, which is essential because the `AssetWithTestPrice` eligibility filter requires exact price presence on both endpoints.

The core logic in `generate_all_splits`:
1. Iterates through the 69 time points (trading-day snapped)
2. For each time point, **cumulatively** builds training interactions by adding only the delta since the last time point (not rebuilding from scratch each time)
3. Builds test interactions from the window ending at the paired slot 13 steps ahead
4. Applies filters in order: per-user dedup (remove test items the user already held in training), global cold-start filter (remove test items that never appeared in training anywhere), asset eligibility (must have a close price exactly on both the recommendation date AND the test-end date), user eligibility (must have at least one interaction in both train and test)
5. Builds ID-to-index mappings for converting string IDs to integer tensors

The cumulative construction pattern is important for understanding performance: `_add_delta_transactions` mutates the `cumulative_training` dict in place across loop iterations, and `copy.deepcopy` snapshots it for each split.

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

**6. `src/data/sequences.py`**

Builds the data that sequential models need. Four utilities:
- `build_user_sequences`: chronological `(asset_id, date)` pairs per user from buy transactions. Unlike interaction sets, this **preserves repeat purchases** and ordering
- `truncate_sequences`: keeps only the last N items (matching `max_sequence_length`)
- `compute_relative_time_intervals`: days between consecutive purchases (e.g., [0, 3, 45, 1, 180])
- `bucket_time_values`: log-scale bucketing to handle the long tail (a 1-day gap and a 365-day gap both get finite bucket indices)

The time bucketing converts raw day counts into discrete indices that can be used as embedding lookup keys in TiSASRec.

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

**7. `src/features/technical_indicators.py`**

Computes the 30-column `full_short` indicator set that the FAR-Trans paper uses for its RF baseline. These indicators feed the Random Forest baseline and the HybridDualHead's profitability MLP.

Eleven indicator families are computed across three rolling horizons (21d, 63d, 126d) giving 30 columns total:

| Indicator family | Horizons | Formula |
|---|---|---|
| `avg_price` | 21, 63, 126 | `close.rolling(h).mean()` |
| `past_profitability` | 21, 63, 126 | `(close - close.shift(h)) / close.shift(h)` |
| `volatility` | 21, 63, 126 | `std(daily_return, h) * sqrt(252)` |
| `sharpe` | 21, 63, 126 | `past_profitability_h / volatility_h` (Inf/NaN → 0) |
| `m` (momentum) | 21, 63, 126 | `close.diff(h)` |
| `roc` (rate of change) | 21, 63, 126 | `m_h / close.shift(h)` |
| `min`, `max` | 21, 63, 126 | `close.rolling(h).{min,max}()` |
| `exp_mean` | 21, 63, 126 | `close.ewm(span=h).mean()` |
| `MACD` | single | `EMA(close, 12) - EMA(close, 26)` (adjust=False) |
| `rsi_14` | single | Wilder's RSI via EMA(span=14, adjust=False) |
| `dco_22` | single | `close.shift(12) - close.rolling(22).mean()` |

After computing all indicators, `_smooth_numeric_columns` applies a 5-day moving-average pass to every numeric column (matching `algorithms/kpi_gen/ma_kpi_generator.py:62-65` in the [FAR-Trans reference](https://github.com/JavierSanzCruza/far-trans)), followed by per-asset `dropna()`. The 30 column names in `INDICATOR_COLUMNS` match `recommendation.py:85-94` in the same repo name-for-name in identical order.

`compute_all_indicators` is a helper that zero-fills missing-asset rows at lookup time. It is only used by the HybridDualHead extension; the RF baseline queries `_indicator_dataframe` directly.

### Phase 3: Models (the recommendation algorithms)

Read these in inheritance order. Each model extends the previous one.

**8. `src/models/protocol.py`**

Two things live here:

- The `Recommender` protocol (3 methods): `name`, `train_on_split`, `recommend_for_user`. Every model implements this interface. The pipeline code only interacts with models through this protocol.
- `MODEL_REGISTRY`: a dict of `ModelEntry` records (config class, data dependencies, factory) that makes the runner and tuner data-driven. Adding a new model is one registry entry plus the `Recommender` implementation; the runner and tuner automatically pick it up.

`build_recommender(model_name, config, ...)` resolves a registry entry and returns a constructed recommender, using the paper-default config when `config is None`.

**9. `src/models/train.py`**

Two utilities:
- `set_random_seeds`: reproducibility across numpy, torch, and Python random
- `train_pytorch_model`: a generic training loop (model, dataset, loss function, optimizer, epochs). Used by SASRec, TiSASRec, and HybridDualHead. LightGCN has its own training loop because it needs the edge index.

**10. `src/models/random_forest.py`**

The simplest model. Read this first to understand the Recommender pattern:
- `train_on_split`: computes technical indicators for all eligible assets, computes actual 6-month ROI as training targets, fits a sklearn `RandomForestRegressor`, then ranks all assets by predicted ROI
- `recommend_for_user`: returns the top-k from the pre-computed ranking, skipping assets the user already owns

Key insight: this model is **not personalised**. Every user gets the same ranking (minus their excluded assets).

**11. `src/models/light_gcn.py`**

Graph collaborative filtering. Two classes:
- `LightGCNModel` (nn.Module): user and asset embeddings (Xavier uniform init), `LGConv(normalize=False)` layers from PyTorch Geometric, BPR (Bayesian Personalised Ranking) loss
- `LightGCNBaseline` (Recommender): builds the bipartite user-asset graph from all training interactions, pre-computes symmetric `D^{-1/2}AD^{-1/2}` normalization via `gcn_norm(..., add_self_loops=False)`, trains with BPR loss and L2 regularisation on initial (0th-layer) embeddings of batch users/items (matching the original He et al. 2020 formulation), scores via dot product of user and asset embeddings

The FAR-Trans paper runs LightGCN through Beta-RecSys, whose `base_data.py:337-360` applies NGCF-style asymmetric `D^{-1}(A + I)` normalization with added self-loops.

This is different from the original LightGCN paper (He et al. 2020, Eq. 3), which specifies symmetric `D^{-1/2}AD^{-1/2}` without self-loops. We use the correct canonical form via PyG's `LGConv`. 

We also deliberately use last-epoch weights at inference. Beta-RecSys tracks a "best validation nDCG" snapshot, but FAR-Trans sets the validation set equal to the test set (`splitted_data.py:41`), making their best-epoch selection leaky model selection. The module docstring at the top of `light_gcn.py` explains this in detail.

**12. `src/models/sasrec.py`**

The base sequential Transformer. This is the largest file and contains several tightly coupled classes. Read in this order:

**`SASRecModel` (nn.Module)**: the Transformer encoder.
- Takes `input_ids` (batch of padded asset index sequences, 0 = padding)
- Adds asset embedding + positional embedding, then passes through N `TransformerBlock`s
- Uses a causal mask (upper-triangular) so each position can only attend to itself and earlier positions
- `_compute_attention_scores(Q, K)`: the override point. SASRec does plain `Q * K^T`. TiSASRec overrides this to add time-interval bias terms
- `predict`: runs the forward pass, takes the last position's hidden state, and dot-products it with candidate asset embeddings to get scores

**`TransformerBlock` (nn.Module)**: standard pre-norm Transformer block.
- Multi-head self-attention with `attention_score_fn` as an injectable function (this is how TiSASRec injects time awareness)
- Feed-forward network with GELU activation
- Two residual connections with LayerNorm

**`SASRecDataset`**: prepares training data.
- For each user, converts the asset sequence to indices (offset by +1 because 0 is padding)
- Input is positions [0, L-1], target is positions [1, L] (next-item prediction)
- Negative samples are randomly drawn from assets the user hasn't bought
- Left-pads shorter sequences with zeros

**`SASRecRecommender`**: the Recommender wrapper.
- `train_on_split`: receives user sequences via kwargs, truncates them, builds dataset, trains with BCE loss on next-item prediction using `train_pytorch_model`
- `_store_user_sequences`: caches index sequences for inference
- `recommend_for_user`: left-pads the user's sequence, runs the model, scores all candidate assets, excludes already-owned, returns top-k

**13. `src/models/tisasrec.py`**

Extends SASRec with time-interval awareness. Read this as a diff from SASRec.

**`TiSASRecModel(SASRecModel)`**: adds two embedding tables:
- `relative_time_embedding`: maps bucketed time gaps between any two positions to per-head bias values
- `absolute_time_embedding`: maps bucketed absolute timestamps to per-head bias values
- Overrides `_compute_attention_scores` to add these biases to the base `Q * K^T` scores

**`TiSASRecDataset(SASRecDataset)`**: extends the base dataset to also return:
- A relative time matrix (seq_len x seq_len): bucketed day gap between every pair of positions
- An absolute time matrix (seq_len x seq_len): bucketed absolute position of each column's timestamp
- Both matrices are left-padded with zeros to match `max_sequence_length`

**`TiSASRecRecommender(SASRecRecommender)`**: overrides:
- `_build_model`: returns `TiSASRecModel` (with time bucket count)
- `_build_dataset`: returns `TiSASRecDataset` (with dates and reference date)
- `train_on_split`: stores user dates, computes reference date, passes time matrices during training
- `recommend_for_user`: constructs time matrices for a single user at inference time
- `_build_inference_time_matrices`: builds padded relative and absolute time matrices for one user

**14. `src/models/hybrid.py`**

The project's novel contribution: a dual-head model that jointly optimizes interest prediction and profitability prediction.

**`HybridDualHeadModel(TiSASRecModel)`**: adds:
- `profitability_head`: an MLP that takes `[user_hidden_state ; technical_indicators]` and outputs a scalar ROI prediction
- `predict_profitability`: expands user hidden states and indicator features to (batch, n_candidates, dim), concatenates, and feeds through the MLP

**`HybridDualHeadRecommender(TiSASRecRecommender)`**: overrides:
- `train_on_split`: pre-computes indicator features and ROI targets for all assets, then trains with a combined loss: `interest_loss + lambda * profit_loss`. The interest loss is BCE (same as SASRec/TiSASRec), the profit loss is MSE between predicted and actual 6-month monthly ROI
- `_build_indicator_and_roi_tensors`: computes indicator features and actual ROI targets as tensors indexed by asset ID
- `recommend_for_user`: computes both interest scores (dot product with asset embeddings) and profitability scores (MLP output), min-max normalizes both to [0,1], then blends: `alpha * interest + (1-alpha) * profit`

### Phase 4: Evaluation

**15. `src/evaluation/metrics.py`**

Two metrics (both verified line-for-line against `metrics/` in the [FAR-Trans reference](https://github.com/JavierSanzCruza/far-trans)):

- `compute_ndcg_at_k`: standard nDCG with binary relevance (1 if user bought the asset in the test window, 0 otherwise). IDCG is capped at `min(k, num_relevant)`. Users with no relevant items contribute 0 to the average. We use `log2`; the paper uses natural log; the ratio is base-invariant.
- `compute_roi_at_k`: per-asset geometric monthly return `(1 + total_return)^(30/days_in_period) - 1`, averaged across the top-k list. Missing-price or zero-start-price assets are **imputed as 0 return, not skipped**, so the denominator is the actual number of items in the top-k slice. This matches `metrics/kpi_monthly_evaluation_metric.py:27` and `metrics/kpi_evaluation_metric.py:30-32` in the FAR-Trans repo.
- `build_price_lookup`: finds the closest available price on or before the time point and test end for each asset. Since all split dates are snapped to actual trading days, the `<=` fallback collapses to exact-date lookup.
- `evaluate_model_on_split`: iterates `split.eligible_customer_ids` (users in both train and test), averages both metrics across users.

### Phase 5: Pipeline (orchestrating everything)

**16. `src/pipeline/preprocessing.py`**

The entry point for data preparation (Step 0). `run_preprocessing`:
1. Loads all raw CSVs
2. Generates 69 evaluation splits and 3 validation splits
3. Builds user purchase sequences for each split
4. Saves everything to disk as JSON files

Also provides loader functions (`load_evaluation_splits`, `load_validation_splits`, `load_evaluation_sequences`, etc.) that all downstream pipeline modules use.

Run as: `uv run python -m src.pipeline.preprocessing --output data/splits`

**17. `src/pipeline/tuning.py`**

Hyperparameter optimization (Step 1). Uses Ray Tune's native `tune.grid_search(...)` to exhaustively cover small grids **centered on each model's FAR-Trans paper configuration**, so the paper's exact hyperparameters are always one of the trial points. Each model has its own `ModelTuningSpec` (`RANDOM_FOREST_TUNING`, `LIGHT_GCN_TUNING`, `SASREC_TUNING`, `TISASREC_TUNING`, `HYBRID_DUAL_HEAD_TUNING`) declaring:
- `grid`: dict of hyperparameter name → list of values
- `config_class`: Pydantic config to instantiate from a trial
- `needs_indicators`, `needs_sequences`: data dependencies for `ValidationContext`
- `primary_metric`: "roi" for price-based, "ndcg" for transaction-based

The validation dates (2019-04-01, 2019-10-01, 2020-01-31) all precede the first evaluation split (2019-08-01), preventing data leakage. Each is snapped to the nearest trading day so the `AssetWithTestPrice` eligibility rule has a non-empty candidate pool.

Best configs are saved to JSON and loaded by the runner. To skip tuning entirely and use paper defaults, omit `--config` on `poe run`:

```sh
uv run poe tune --models random_forest light_gcn
uv run poe run --config outputs/configs/best_hyperparameters.json

# or skip tuning and run paper defaults:
uv run poe run --models random_forest light_gcn
```

**18. `src/pipeline/runner.py`**

The main experiment loop (Step 2). `run_all_experiments`:
1. Loads evaluation splits, close prices, and sequences from disk
2. Resolves the model set via `MODEL_REGISTRY` (optionally filtered by `--models`)
3. For each selected model, constructs the recommender via `build_recommender(name, config=provided or paper-default)`
4. For each split: train, generate recommendations, evaluate, write CSV row
5. Prints per-model summary and a cross-model comparison table

The runner is data-driven: loop bodies look up `MODEL_REGISTRY[name].needs_indicators` / `needs_sequences` to decide which auxiliary data to load. Adding a new model to the registry automatically makes it runnable without editing runner code.

The key orchestration function is `run_experiment`: for each split, it calls `model.train_on_split`, then `generate_recommendations` (which calls `recommend_for_user` for each eligible user), then `evaluate_model_on_split`.

Run as: `uv run poe run --models random_forest light_gcn` (paper defaults) or `uv run poe run --config outputs/configs/best_hyperparameters.json` (tuned configs).


## Data Flow Summary

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

## Model Inheritance Chain

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

## Key Design Decisions

1. **Cumulative split construction**: training interactions grow incrementally across splits rather than being rebuilt from scratch, making 69 splits feasible

2. **Trading-day snapped schedule**: the 69 split dates are picked from the actual `close_prices.csv` trading calendar using the same grid-division algorithm as the FAR-Trans reference (`get_dates()`), adapted to a single range `(2019-08-01, 2022-05-23)`. Slots are spaced approximately 9 trading days apart; each split's test-end is 13 slots ahead (approximately 6 months). This is load-bearing because asset eligibility requires an exact price row on both endpoints.

3. **Model registry over hardcoded switches**: `MODEL_REGISTRY` in `src/models/protocol.py` is the single source of truth. The runner and tuner are data-driven; adding a new model is one registry entry plus the `Recommender` implementation.

4. **Protocol-based polymorphism**: all models implement `Recommender` (a `typing.Protocol`), so the runner doesn't import model classes directly.

5. **Override points in the inheritance chain**: `_compute_attention_scores`, `_build_model`, and `_build_dataset` are the primary extension points for the SASRec → TiSASRec → HybridDualHead chain, keeping the shared logic (training loop, padding, masking) in the base class.

6. **Recommender wraps nn.Module**: the Recommender classes handle sequence padding, index mapping, and top-k selection so the neural network code stays focused on forward passes and loss computation.

7. **Preprocessing to disk**: splits and sequences are serialized to JSON so that tuning and evaluation can load from disk independently without re-processing raw data each time.

8. **Canonical LightGCN, not the paper's buggy reference**: we use PyG's `LGConv` (symmetric `D^{-1/2}AD^{-1/2}`, no self-loops, per He et al. 2020) instead of reproducing Beta-RecSys's NGCF-style normalization. Paper fidelity traded for ML correctness, with a clear docstring explaining the decision.

9. **No leaky validation for LightGCN**: Beta-RecSys's `best_valid_performance` snapshot uses a validation set equal to the test set, which is label leakage. We train the full `number_of_epochs` and use last-epoch weights. Honest at the cost of a potentially lower nDCG vs the paper's reported number.

10. **Grid search centered on paper configs**: the tuning pipeline uses Ray Tune grid search with each model's paper default as one of the grid points, so tuned runs can never regress below the paper baseline.
