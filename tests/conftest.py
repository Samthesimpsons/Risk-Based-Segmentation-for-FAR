"""Shared pytest fixtures: synthetic transactions, close prices, customers, and assets."""

import pandas as pd
import pytest


@pytest.fixture()
def sample_transactions() -> pd.DataFrame:
    """Small transaction dataset: 3 customers, 4 assets, spanning 2019-01 to 2020-06."""
    return pd.DataFrame(
        {
            "customerID": [
                "C1",
                "C1",
                "C1",
                "C2",
                "C2",
                "C2",
                "C3",
                "C3",
                "C3",
                "C1",
            ],
            "ISIN": [
                "A1",
                "A2",
                "A3",
                "A1",
                "A2",
                "A4",
                "A1",
                "A3",
                "A4",
                "A2",
            ],
            "transactionType": [
                "Buy",
                "Buy",
                "Buy",
                "Buy",
                "Sell",
                "Buy",
                "Buy",
                "Buy",
                "Buy",
                "Buy",
            ],
            "timestamp": pd.to_datetime(
                [
                    "2019-01-15",
                    "2019-03-20",
                    "2019-06-10",
                    "2019-04-01",
                    "2019-05-15",
                    "2019-09-01",
                    "2019-07-20",
                    "2019-10-05",
                    "2020-01-10",
                    "2020-06-01",
                ]
            ),
        }
    )


@pytest.fixture()
def sample_close_prices() -> pd.DataFrame:
    """Daily close prices for 4 assets from 2018-12 to 2020-12."""
    dates = pd.date_range("2018-12-01", "2020-12-31", freq="MS")
    rows: list[dict[str, object]] = []
    base_prices = {"A1": 100.0, "A2": 50.0, "A3": 200.0, "A4": 75.0}
    for timestamp in dates:
        for isin, base_price in base_prices.items():
            rows.append(
                {
                    "ISIN": isin,
                    "timestamp": timestamp,
                    "closePrice": base_price + float(timestamp.month),
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture()
def sample_limit_prices() -> pd.DataFrame:
    """Limit prices with tradeable date ranges for 4 assets."""
    return pd.DataFrame(
        {
            "ISIN": ["A1", "A2", "A3", "A4"],
            "minDate": pd.to_datetime(
                ["2018-01-01", "2018-01-01", "2018-01-01", "2019-06-01"]
            ),
            "maxDate": pd.to_datetime(
                ["2021-12-31", "2021-12-31", "2020-01-01", "2021-12-31"]
            ),
            "priceMinDate": [90.0, 40.0, 180.0, 60.0],
            "priceMaxDate": [120.0, 70.0, 230.0, 90.0],
            "profitability": [0.05, 0.03, -0.02, 0.10],
        }
    )


@pytest.fixture()
def sample_customers() -> pd.DataFrame:
    """Customer records with duplicates for dedup testing."""
    return pd.DataFrame(
        {
            "customerID": ["C1", "C1", "C2", "C3"],
            "lastQuestionnaireDate": pd.to_datetime(
                ["2019-01-01", "2019-06-01", "2019-03-01", "2019-05-01"]
            ),
            "timestamp": pd.to_datetime(
                ["2019-01-01", "2019-07-01", "2019-03-01", "2019-05-01"]
            ),
            "riskProfile": ["low", "medium", "high", "low"],
        }
    )


@pytest.fixture()
def sample_assets() -> pd.DataFrame:
    """Asset records with duplicates for dedup testing."""
    return pd.DataFrame(
        {
            "ISIN": ["A1", "A1", "A2", "A3"],
            "timestamp": pd.to_datetime(
                ["2019-01-01", "2019-08-01", "2019-02-01", "2019-04-01"]
            ),
            "assetType": ["equity", "equity_updated", "bond", "commodity"],
        }
    )
