"""Shared profile-coherence primitives: risk-band assignment, profile parsing, discordance."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config.schemas import CustomerProfile

CONSERVATIVE = 0
INCOME = 1
BALANCED = 2
AGGRESSIVE = 3
NUMBER_OF_RISK_BANDS = 4

RISK_BAND_NAMES: dict[int, str] = {
    CONSERVATIVE: "Conservative",
    INCOME: "Income",
    BALANCED: "Balanced",
    AGGRESSIVE: "Aggressive",
}

TRADING_DAYS_PER_YEAR = 252
TRADING_DAYS_PER_MONTH = 21
CALENDAR_DAYS_PER_MONTH = 30


@dataclass(frozen=True)
class RiskBandClassificationRules:
    """Lookup tables that map raw asset and customer fields to ordinal MiFID risk bands."""

    mtf_subcategory_to_band: dict[str, int]
    bond_subcategory_to_band: dict[str, int]
    declared_risk_level_to_band: dict[str, int]
    predicted_risk_level_to_band: dict[str, int]


def _build_classification_rules() -> RiskBandClassificationRules:
    """Construct the canonical risk-band classification rules used across this module."""
    declared_risk_level_to_band: dict[str, int] = {
        "Conservative": CONSERVATIVE,
        "Income": INCOME,
        "Balanced": BALANCED,
        "Aggressive": AGGRESSIVE,
    }
    return RiskBandClassificationRules(
        mtf_subcategory_to_band={
            "Money Market": CONSERVATIVE,
            "Bond": INCOME,
            "Bonds": INCOME,
            "Balanced": BALANCED,
            "Equity": AGGRESSIVE,
            "Large Cap": AGGRESSIVE,
        },
        bond_subcategory_to_band={
            "Government": CONSERVATIVE,
            "Corporate": INCOME,
        },
        declared_risk_level_to_band=declared_risk_level_to_band,
        predicted_risk_level_to_band={
            f"Predicted_{name}": band
            for name, band in declared_risk_level_to_band.items()
        },
    )


_RISK_BAND_RULES = _build_classification_rules()


def _classify_by_subcategory(
    asset_category: str, asset_subcategory: str | None
) -> int | None:
    """Return the risk band implied by the asset's category metadata, or None."""
    if asset_subcategory is None or pd.isna(asset_subcategory):
        return INCOME if asset_category == "Bond" else None

    if asset_category == "MTF":
        return _RISK_BAND_RULES.mtf_subcategory_to_band.get(asset_subcategory)

    if asset_category == "Bond":
        return _RISK_BAND_RULES.bond_subcategory_to_band.get(asset_subcategory, INCOME)

    return None


def compute_annualised_volatility(
    close_prices: pd.DataFrame,
    window: int = TRADING_DAYS_PER_YEAR,
) -> pd.Series:
    """Return the trailing-window annualised volatility per asset, indexed by ISIN."""
    sorted_prices = close_prices.sort_values(["ISIN", "timestamp"])
    log_returns = (
        sorted_prices.groupby("ISIN")["closePrice"]
        .apply(lambda group: np.log(group).diff())
        .reset_index(level=0, drop=True)
    )
    sorted_prices = sorted_prices.assign(log_return=log_returns)

    grouped = sorted_prices.groupby("ISIN")["log_return"]
    rolling_std = grouped.rolling(window=window, min_periods=window).std()

    annualised = rolling_std * np.sqrt(TRADING_DAYS_PER_YEAR)
    annualised.index = annualised.index.droplevel(0)
    sorted_prices = sorted_prices.assign(annualised_volatility=annualised)

    latest_per_asset = (
        sorted_prices.dropna(subset=["annualised_volatility"])
        .groupby("ISIN")["annualised_volatility"]
        .last()
    )
    return pd.Series(latest_per_asset)


def _band_from_volatility_quartile(
    isin: str,
    volatility_lookup: pd.Series,
    quartile_cutoffs: tuple[float, float, float] | None,
) -> int:
    """Assign a risk band to a single asset by its volatility quartile."""
    if quartile_cutoffs is None or isin not in volatility_lookup.index:
        return BALANCED

    q1, q2, q3 = quartile_cutoffs
    asset_volatility = float(volatility_lookup.loc[isin])
    if asset_volatility <= q1:
        return CONSERVATIVE
    if asset_volatility <= q2:
        return INCOME
    if asset_volatility <= q3:
        return BALANCED
    return AGGRESSIVE


def build_asset_risk_classes(
    asset_information: pd.DataFrame,
    close_prices: pd.DataFrame,
    volatility_window: int = TRADING_DAYS_PER_YEAR,
) -> dict[str, int]:
    """Return a mapping `asset_id -> ordinal MiFID risk band`."""
    volatility_lookup = compute_annualised_volatility(
        close_prices, window=volatility_window
    )

    stock_isins = set(
        asset_information.loc[asset_information["assetCategory"] == "Stock", "ISIN"]
    )
    stock_volatility = volatility_lookup[
        volatility_lookup.index.isin(stock_isins)
    ].dropna()

    quartile_cutoffs: tuple[float, float, float] | None = None
    if len(stock_volatility) >= 4:
        q1, q2, q3 = np.quantile(stock_volatility.to_numpy(), [0.25, 0.5, 0.75])
        quartile_cutoffs = (float(q1), float(q2), float(q3))

    risk_classes: dict[str, int] = {}
    for _, row in asset_information.iterrows():
        isin = str(row["ISIN"])
        category = str(row["assetCategory"])
        subcategory = row.get("assetSubCategory")
        subcategory_str = str(subcategory) if isinstance(subcategory, str) else None
        band_from_metadata = _classify_by_subcategory(category, subcategory_str)
        if band_from_metadata is not None:
            risk_classes[isin] = band_from_metadata
            continue
        risk_classes[isin] = _band_from_volatility_quartile(
            isin, volatility_lookup, quartile_cutoffs
        )

    return risk_classes


def _parse_risk_level(raw_value: object) -> tuple[int | None, bool]:
    """Map a raw `riskLevel` cell to (ordinal_band, is_predicted)."""
    if not isinstance(raw_value, str):
        return None, False
    if raw_value in _RISK_BAND_RULES.declared_risk_level_to_band:
        return _RISK_BAND_RULES.declared_risk_level_to_band[raw_value], False
    if raw_value in _RISK_BAND_RULES.predicted_risk_level_to_band:
        return _RISK_BAND_RULES.predicted_risk_level_to_band[raw_value], True
    return None, False


def _normalise_optional_string(raw_value: object) -> str | None:
    """Return None for missing or `Not_Available` cells, else the string value."""
    if not isinstance(raw_value, str):
        return None
    if raw_value == "Not_Available":
        return None
    return raw_value


def build_customer_profile_lookup(
    customer_information: pd.DataFrame,
) -> dict[str, CustomerProfile]:
    """Return a mapping `customer_id -> CustomerProfile`."""
    profiles: dict[str, CustomerProfile] = {}
    for _, row in customer_information.iterrows():
        customer_id = str(row["customerID"])
        risk_band, is_predicted = _parse_risk_level(row.get("riskLevel"))
        profiles[customer_id] = CustomerProfile(
            customer_id=customer_id,
            risk_band=risk_band,
            risk_band_is_predicted=is_predicted,
            customer_type=_normalise_optional_string(row.get("customerType")),
            investment_capacity=_normalise_optional_string(
                row.get("investmentCapacity")
            ),
        )
    return profiles


def compute_pairwise_discordance(
    customer_band: int | None,
    asset_band: int | None,
) -> int | None:
    """Return |customer_band - asset_band|, or None if either band is missing."""
    if customer_band is None or asset_band is None:
        return None
    return abs(customer_band - asset_band)


def is_profile_coherent(discordance: int | None) -> bool:
    """Return True when discordance is within the default tolerance (<=1)."""
    if discordance is None:
        return False
    return discordance <= 1
