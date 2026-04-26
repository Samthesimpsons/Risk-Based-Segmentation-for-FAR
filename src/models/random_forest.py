"""Random Forest profitability regressor baseline."""

import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.config.schemas import TemporalSplitData
from src.config.settings import RandomForestConfig
from src.features.technical_indicators import DEFAULT_INDICATOR_PARAMETERS
from src.utils.profile_coherence import TRADING_DAYS_PER_MONTH


class RandomForestBaseline:
    """Random Forest regressor for asset profitability prediction."""

    def __init__(
        self,
        random_forest_config: RandomForestConfig,
        indicator_dataframe: pd.DataFrame,
    ) -> None:
        """Store configuration and the indicator dataframe used for training."""
        self._config = random_forest_config
        self._indicator_dataframe = indicator_dataframe
        self._model: RandomForestRegressor | None = None
        self._ranked_assets: list[str] = []

    @property
    def name(self) -> str:
        """Return the display name used in evaluation results."""
        return "RandomForest"

    def train_on_split(self, split: TemporalSplitData, **kwargs: object) -> None:
        """Train the regressor on historical indicators and rank eligible assets by predicted ROI."""
        train_date = pd.Timestamp(split.time_point)
        horizon_months = self._config.prediction_horizon_months
        forward_trading_days = horizon_months * TRADING_DAYS_PER_MONTH
        delta = datetime.timedelta(days=horizon_months * 30)

        eligible_assets = set(split.eligible_asset_ids)
        indicators = self._indicator_dataframe[
            self._indicator_dataframe["ISIN"].isin(eligible_assets)
        ].copy()

        asset_frames: list[pd.DataFrame] = []
        for _, asset_data in indicators.groupby("ISIN"):
            asset_frame = asset_data.sort_values("timestamp").copy()
            asset_frame["final_price"] = asset_frame["closePrice"].shift(
                -forward_trading_days
            )
            asset_frame["target"] = (
                asset_frame["final_price"] - asset_frame["closePrice"]
            ) / asset_frame["closePrice"]
            asset_frame = asset_frame[asset_frame["closePrice"] > 0.0]
            asset_frames.append(asset_frame)

        if not asset_frames:
            self._ranked_assets = list(split.eligible_asset_ids)
            return

        training_data = pd.concat(asset_frames, ignore_index=True)
        training_data = training_data[training_data["timestamp"] < (train_date - delta)]

        feature_columns = [
            c
            for c in DEFAULT_INDICATOR_PARAMETERS.columns
            if c in training_data.columns
        ]
        keep_columns = feature_columns + ["target"]
        training_data = training_data[keep_columns].dropna()

        if training_data.empty:
            self._ranked_assets = list(split.eligible_asset_ids)
            return

        feature_matrix = training_data[feature_columns].to_numpy(dtype=np.float64)
        target_vector = training_data["target"].to_numpy(dtype=np.float64)

        self._model = RandomForestRegressor(
            n_estimators=self._config.number_of_estimators,
            max_depth=self._config.max_depth,
            random_state=self._config.random_state,
        )
        self._model.fit(feature_matrix, target_vector)

        current_indicators = self._indicator_dataframe[
            (self._indicator_dataframe["timestamp"] == train_date)
            & (self._indicator_dataframe["ISIN"].isin(eligible_assets))
        ]

        if current_indicators.empty:
            self._ranked_assets = list(split.eligible_asset_ids)
            return

        current_features = current_indicators.set_index("ISIN")[feature_columns]
        present_assets = [
            asset_id
            for asset_id in split.eligible_asset_ids
            if asset_id in current_features.index
        ]

        if not present_assets:
            self._ranked_assets = list(split.eligible_asset_ids)
            return

        all_features = current_features.loc[present_assets].to_numpy(dtype=np.float64)
        predicted_roi = self._model.predict(all_features)

        asset_roi_pairs = list(zip(present_assets, predicted_roi))
        asset_roi_pairs.sort(key=lambda pair: pair[1], reverse=True)
        self._ranked_assets = [asset_id for asset_id, _ in asset_roi_pairs]

        remaining = [
            asset_id
            for asset_id in split.eligible_asset_ids
            if asset_id not in set(self._ranked_assets)
        ]
        self._ranked_assets.extend(remaining)

    def recommend_for_user(
        self,
        user_id: str,
        excluded_assets: set[str],
        k: int = 10,
    ) -> list[str]:
        """Return top-k assets by predicted ROI, excluding held assets."""
        recommendations: list[str] = []
        for asset_id in self._ranked_assets:
            if asset_id in excluded_assets:
                continue
            recommendations.append(asset_id)
            if len(recommendations) >= k:
                break
        return recommendations
