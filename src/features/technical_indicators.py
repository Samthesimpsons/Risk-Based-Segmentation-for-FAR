"""Technical indicator computation matching the FAR-Trans reference implementation.

Produces a daily indicator DataFrame with 30 features per (asset, date) row,
using multi-horizon rolling windows (21d, 63d, 126d) and standard formulas
for RSI (EWM), MACD, DCO, Sharpe, etc.
"""

import math

import pandas as pd

DEFAULT_PERIODS: tuple[int, ...] = (21, 63, 126, 189)

ANNUALIZATION_FACTOR: float = math.sqrt(252)
RSI_PERIOD: int = 14
DCO_PERIOD: int = 22
MACD_SHORT_SPAN: int = 12
MACD_LONG_SPAN: int = 26
DEFAULT_SMOOTHING_WINDOW: int = 5

INDICATOR_COLUMNS: list[str] = [
    "past_profitability_21d",
    "past_profitability_63d",
    "past_profitability_126d",
    "volatility_21d",
    "volatility_63d",
    "volatility_126d",
    "avg_price_21d",
    "avg_price_63d",
    "avg_price_126d",
    "sharpe_21d",
    "sharpe_63d",
    "sharpe_126d",
    "m_21d",
    "m_63d",
    "m_126d",
    "roc_21d",
    "roc_63d",
    "roc_126d",
    "MACD",
    "rsi_14",
    "dco_22",
    "min_21d",
    "min_63d",
    "min_126d",
    "max_21d",
    "max_63d",
    "max_126d",
    "exp_mean_21d",
    "exp_mean_63d",
    "exp_mean_126d",
]

TRADING_DAYS_PER_MONTH: int = 21


def build_indicator_dataframe(
    close_prices: pd.DataFrame,
    smoothing_window: int = DEFAULT_SMOOTHING_WINDOW,
    dropna: bool = True,
) -> pd.DataFrame:
    """Compute all technical indicators for every asset across all dates.

    Returns a DataFrame with columns: ISIN, timestamp, closePrice, plus all 30
    indicator columns. By default, applies the upstream 5-day moving-average
    smoothing pass to every numeric column and drops rows that remain incomplete.
    """
    asset_frames: list[pd.DataFrame] = []

    for _, asset_data in close_prices.groupby("ISIN"):
        asset_frame = asset_data.sort_values("timestamp").copy()

        asset_frame = _add_avg_price(asset_frame)
        asset_frame = _add_roi(asset_frame)
        asset_frame = _add_volatility(asset_frame)
        asset_frame = _add_macd(asset_frame)
        asset_frame = _add_momentum(asset_frame)
        asset_frame = _add_rate_of_change(asset_frame)
        asset_frame = _add_rsi(asset_frame, RSI_PERIOD)
        asset_frame = _add_detrended_close_oscillator(asset_frame, DCO_PERIOD)
        asset_frame = _add_sharpe(asset_frame)
        asset_frame = _add_min_max_exp(asset_frame)
        asset_frame = _smooth_numeric_columns(asset_frame, smoothing_window)

        if dropna:
            asset_frame = asset_frame.dropna().reset_index(drop=True)

        asset_frames.append(asset_frame)

    return pd.concat(asset_frames, ignore_index=True)


def compute_all_indicators(
    indicator_dataframe: pd.DataFrame,
    asset_ids: list[str],
    time_point: pd.Timestamp,
) -> pd.DataFrame:
    """Look up pre-computed indicators for given assets at a specific date.

    Returns a DataFrame with ISIN as index and one column per indicator.
    Assets missing from the indicator DataFrame at the requested date get zeros.
    """
    at_date = indicator_dataframe[indicator_dataframe["timestamp"] == time_point]
    at_date = at_date[at_date["ISIN"].isin(asset_ids)]

    result = pd.DataFrame(0.0, index=asset_ids, columns=INDICATOR_COLUMNS)
    result.index.name = "ISIN"

    for _, row in at_date.iterrows():
        asset_id = row["ISIN"]
        for column in INDICATOR_COLUMNS:
            value = row.get(column)
            if value is not None and pd.notna(value):
                result.at[asset_id, column] = float(value)

    return result


def _add_avg_price(
    frame: pd.DataFrame,
    periods: tuple[int, ...] = DEFAULT_PERIODS,
) -> pd.DataFrame:
    for period in periods:
        frame[f"avg_price_{period}d"] = (
            frame["closePrice"].rolling(window=period).mean()
        )
    return frame


def _add_roi(
    frame: pd.DataFrame,
    periods: tuple[int, ...] = (1, 21, 63, 126, 189),
) -> pd.DataFrame:
    """Add past-profitability columns for each requested lookback period."""
    for period in periods:
        shifted = frame["closePrice"].shift(period)
        frame[f"past_profitability_{period}d"] = (
            frame["closePrice"] - shifted
        ) / shifted
    return frame


def _add_volatility(
    frame: pd.DataFrame,
    periods: tuple[int, ...] = DEFAULT_PERIODS,
) -> pd.DataFrame:
    """Add annualised rolling-volatility columns derived from 1-day returns."""
    if "past_profitability_1d" not in frame.columns:
        shifted = frame["closePrice"].shift(1)
        frame["past_profitability_1d"] = (frame["closePrice"] - shifted) / shifted

    for period in periods:
        frame[f"volatility_{period}d"] = (
            frame["past_profitability_1d"].rolling(window=period).std()
            * ANNUALIZATION_FACTOR
        )
    return frame


def _add_sharpe(
    frame: pd.DataFrame,
    periods: tuple[int, ...] = DEFAULT_PERIODS,
) -> pd.DataFrame:
    """Add Sharpe-ratio columns (profitability divided by volatility) per horizon."""
    for period in periods:
        profitability_column = f"past_profitability_{period}d"
        volatility_column = f"volatility_{period}d"
        sharpe_column = f"sharpe_{period}d"
        frame[sharpe_column] = frame[profitability_column] / frame[volatility_column]
        frame[sharpe_column] = frame[sharpe_column].replace([math.inf, -math.inf], 0.0)
        frame[sharpe_column] = frame[sharpe_column].fillna(0.0)
    return frame


def _add_macd(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the MACD column (difference of short and long EMAs of close price)."""
    ema_short = frame["closePrice"].ewm(span=MACD_SHORT_SPAN, adjust=False).mean()
    ema_long = frame["closePrice"].ewm(span=MACD_LONG_SPAN, adjust=False).mean()
    frame["MACD"] = ema_short - ema_long
    return frame


def _add_momentum(
    frame: pd.DataFrame,
    periods: tuple[int, ...] = DEFAULT_PERIODS,
) -> pd.DataFrame:
    """Add raw momentum columns (close.diff over each lookback period)."""
    for period in periods:
        frame[f"m_{period}d"] = frame["closePrice"].diff(period)
    return frame


def _add_rate_of_change(
    frame: pd.DataFrame,
    periods: tuple[int, ...] = DEFAULT_PERIODS,
) -> pd.DataFrame:
    """Add rate-of-change columns (momentum normalised by the lagged price)."""
    for period in periods:
        momentum_column = f"m_{period}d"
        if momentum_column not in frame.columns:
            frame[momentum_column] = frame["closePrice"].diff(period)
        frame[f"roc_{period}d"] = frame[momentum_column] / frame["closePrice"].shift(
            period
        )
    return frame


def _add_rsi(frame: pd.DataFrame, period: int) -> pd.DataFrame:
    """Add Wilder's RSI column using EWM-smoothed gains and losses over `period` days."""
    price_diff = frame["closePrice"].diff()
    up = price_diff.clip(lower=0)
    down = (-price_diff).clip(lower=0)
    average_gain = up.ewm(span=period, adjust=False).mean()
    average_loss = down.ewm(span=period, adjust=False).mean()
    relative_strength = average_gain / average_loss
    column_name = f"rsi_{period}"
    frame[column_name] = 100.0 - 100.0 / (1.0 + relative_strength)
    frame[column_name] = frame[column_name].fillna(0.0)
    return frame


def _add_detrended_close_oscillator(frame: pd.DataFrame, period: int) -> pd.DataFrame:
    """Add the detrended close oscillator column for the given period."""
    mid_index = period // 2 + 1
    simple_moving_average = frame["closePrice"].rolling(window=period).mean()
    frame[f"dco_{period}"] = (
        frame["closePrice"].shift(mid_index) - simple_moving_average
    )
    return frame


def _add_min_max_exp(
    frame: pd.DataFrame,
    periods: tuple[int, ...] = DEFAULT_PERIODS,
) -> pd.DataFrame:
    """Add rolling min, rolling max, and exponential-mean columns per lookback period."""
    for period in periods:
        frame[f"min_{period}d"] = frame["closePrice"].rolling(window=period).min()
        frame[f"max_{period}d"] = frame["closePrice"].rolling(window=period).max()
        frame[f"exp_mean_{period}d"] = frame["closePrice"].ewm(span=period).mean()
    return frame


def _smooth_numeric_columns(
    frame: pd.DataFrame,
    smoothing_window: int,
) -> pd.DataFrame:
    """Apply the upstream moving-average smoothing pass to numeric columns."""
    if smoothing_window <= 1:
        return frame

    for column in frame.columns:
        if column in {"ISIN", "timestamp"}:
            continue
        frame[column] = frame[column].rolling(smoothing_window).mean()
    return frame
