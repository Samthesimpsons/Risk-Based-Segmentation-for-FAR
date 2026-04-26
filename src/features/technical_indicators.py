"""Technical indicator features computed per asset and date."""

import math
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TechnicalIndicatorParameters:
    """Hyperparameters and output columns for the technical indicator feature set."""

    lookback_one_month: int
    lookback_three_month: int
    lookback_six_month: int
    lookback_nine_month: int
    macd_short_span: int
    macd_long_span: int
    rsi_period: int
    dco_period: int
    smoothing_window: int
    annualization_factor: float
    columns: tuple[str, ...]

    @property
    def lookback_periods(self) -> tuple[int, ...]:
        """Return the four lookback windows as a tuple."""
        return (
            self.lookback_one_month,
            self.lookback_three_month,
            self.lookback_six_month,
            self.lookback_nine_month,
        )


DEFAULT_INDICATOR_PARAMETERS = TechnicalIndicatorParameters(
    lookback_one_month=21,
    lookback_three_month=63,
    lookback_six_month=126,
    lookback_nine_month=189,
    macd_short_span=12,
    macd_long_span=26,
    rsi_period=14,
    dco_period=22,
    smoothing_window=5,
    annualization_factor=math.sqrt(252),
    columns=(
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
    ),
)


def build_indicator_dataframe(
    close_prices: pd.DataFrame,
    parameters: TechnicalIndicatorParameters = DEFAULT_INDICATOR_PARAMETERS,
    dropna: bool = True,
) -> pd.DataFrame:
    """Compute all technical indicator columns for every asset across all dates."""
    asset_frames: list[pd.DataFrame] = []

    for _, asset_data in close_prices.groupby("ISIN"):
        asset_frame = asset_data.sort_values("timestamp").copy()

        asset_frame = _add_avg_price(asset_frame, parameters.lookback_periods)
        asset_frame = _add_roi(asset_frame, parameters.lookback_periods)
        asset_frame = _add_volatility(
            asset_frame, parameters.lookback_periods, parameters.annualization_factor
        )
        asset_frame = _add_macd(
            asset_frame, parameters.macd_short_span, parameters.macd_long_span
        )
        asset_frame = _add_momentum(asset_frame, parameters.lookback_periods)
        asset_frame = _add_rate_of_change(asset_frame, parameters.lookback_periods)
        asset_frame = _add_rsi(asset_frame, parameters.rsi_period)
        asset_frame = _add_detrended_close_oscillator(
            asset_frame, parameters.dco_period
        )
        asset_frame = _add_sharpe(asset_frame, parameters.lookback_periods)
        asset_frame = _add_min_max_exp(asset_frame, parameters.lookback_periods)
        asset_frame = _smooth_numeric_columns(asset_frame, parameters.smoothing_window)

        if dropna:
            asset_frame = asset_frame.dropna().reset_index(drop=True)

        asset_frames.append(asset_frame)

    return pd.concat(asset_frames, ignore_index=True)


def _add_avg_price(
    frame: pd.DataFrame,
    periods: tuple[int, ...],
) -> pd.DataFrame:
    """Add rolling average price columns for each lookback period."""
    for period in periods:
        frame[f"avg_price_{period}d"] = (
            frame["closePrice"].rolling(window=period).mean()
        )
    return frame


def _add_roi(
    frame: pd.DataFrame,
    periods: tuple[int, ...],
) -> pd.DataFrame:
    """Add past profitability columns for each lookback period."""
    for period in (1, *periods):
        shifted = frame["closePrice"].shift(period)
        frame[f"past_profitability_{period}d"] = (
            frame["closePrice"] - shifted
        ) / shifted
    return frame


def _add_volatility(
    frame: pd.DataFrame,
    periods: tuple[int, ...],
    annualization_factor: float,
) -> pd.DataFrame:
    """Add annualised rolling volatility columns for each lookback period."""
    if "past_profitability_1d" not in frame.columns:
        shifted = frame["closePrice"].shift(1)
        frame["past_profitability_1d"] = (frame["closePrice"] - shifted) / shifted

    for period in periods:
        frame[f"volatility_{period}d"] = (
            frame["past_profitability_1d"].rolling(window=period).std()
            * annualization_factor
        )
    return frame


def _add_sharpe(
    frame: pd.DataFrame,
    periods: tuple[int, ...],
) -> pd.DataFrame:
    """Add Sharpe ratio columns for each lookback period."""
    for period in periods:
        profitability_column = f"past_profitability_{period}d"
        volatility_column = f"volatility_{period}d"
        sharpe_column = f"sharpe_{period}d"
        frame[sharpe_column] = frame[profitability_column] / frame[volatility_column]
        frame[sharpe_column] = frame[sharpe_column].replace([math.inf, -math.inf], 0.0)
        frame[sharpe_column] = frame[sharpe_column].fillna(0.0)
    return frame


def _add_macd(
    frame: pd.DataFrame,
    short_span: int,
    long_span: int,
) -> pd.DataFrame:
    """Add the MACD column from short and long close-price EMAs."""
    ema_short = frame["closePrice"].ewm(span=short_span, adjust=False).mean()
    ema_long = frame["closePrice"].ewm(span=long_span, adjust=False).mean()
    frame["MACD"] = ema_short - ema_long
    return frame


def _add_momentum(
    frame: pd.DataFrame,
    periods: tuple[int, ...],
) -> pd.DataFrame:
    """Add momentum columns for each lookback period."""
    for period in periods:
        frame[f"m_{period}d"] = frame["closePrice"].diff(period)
    return frame


def _add_rate_of_change(
    frame: pd.DataFrame,
    periods: tuple[int, ...],
) -> pd.DataFrame:
    """Add rate of change columns for each lookback period."""
    for period in periods:
        momentum_column = f"m_{period}d"
        if momentum_column not in frame.columns:
            frame[momentum_column] = frame["closePrice"].diff(period)
        frame[f"roc_{period}d"] = frame[momentum_column] / frame["closePrice"].shift(
            period
        )
    return frame


def _add_rsi(frame: pd.DataFrame, period: int) -> pd.DataFrame:
    """Add the Wilder RSI column for the given period."""
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
    periods: tuple[int, ...],
) -> pd.DataFrame:
    """Add rolling min, max, and exponential mean columns for each lookback period."""
    for period in periods:
        frame[f"min_{period}d"] = frame["closePrice"].rolling(window=period).min()
        frame[f"max_{period}d"] = frame["closePrice"].rolling(window=period).max()
        frame[f"exp_mean_{period}d"] = frame["closePrice"].ewm(span=period).mean()
    return frame


def _smooth_numeric_columns(
    frame: pd.DataFrame,
    smoothing_window: int,
) -> pd.DataFrame:
    """Apply rolling-mean smoothing to all numeric columns."""
    if smoothing_window <= 1:
        return frame

    for column in frame.columns:
        if column in {"ISIN", "timestamp"}:
            continue
        frame[column] = frame[column].rolling(smoothing_window).mean()
    return frame
