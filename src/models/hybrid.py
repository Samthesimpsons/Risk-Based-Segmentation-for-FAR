"""Hybrid dual-head recommender combining user interest and asset profitability prediction.

Extends TiSASRec with an additional MLP profitability head trained on technical
indicators. At inference, blends interest scores and profitability scores via a
configurable alpha parameter.
"""

from datetime import date
from typing import cast

import pandas as pd
import torch
import torch.nn as nn

from src.config.schemas import TemporalSplitData
from src.config.settings import HybridDualHeadConfig
from src.data.sequences import truncate_sequences
from src.evaluation.metrics import build_price_lookup
from src.features.technical_indicators import INDICATOR_COLUMNS, compute_all_indicators
from src.models.sasrec import SASRecModel
from src.models.tisasrec import TiSASRecModel, TiSASRecRecommender
from src.models.train import train_pytorch_model

MONTHS_IN_TEST_WINDOW = 6


def _normalize_scores(scores: torch.Tensor) -> torch.Tensor:
    """Min-max normalize scores to [0, 1]."""
    minimum = scores.min()
    maximum = scores.max()
    if maximum == minimum:
        return torch.zeros_like(scores)
    return (scores - minimum) / (maximum - minimum)


class HybridDualHeadModel(TiSASRecModel):
    """TiSASRec encoder with an additional MLP head for profitability prediction."""

    def __init__(
        self,
        number_of_assets: int,
        max_sequence_length: int,
        embedding_dimension: int,
        number_of_attention_heads: int,
        number_of_blocks: int,
        dropout_rate: float,
        time_bucket_count: int,
        number_of_indicators: int,
        profitability_hidden_dimension: int,
    ) -> None:
        super().__init__(
            number_of_assets=number_of_assets,
            max_sequence_length=max_sequence_length,
            embedding_dimension=embedding_dimension,
            number_of_attention_heads=number_of_attention_heads,
            number_of_blocks=number_of_blocks,
            dropout_rate=dropout_rate,
            time_bucket_count=time_bucket_count,
        )
        self.number_of_indicators = number_of_indicators
        self.profitability_head = nn.Sequential(
            nn.Linear(
                embedding_dimension + number_of_indicators,
                profitability_hidden_dimension,
            ),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(profitability_hidden_dimension, 1),
        )

    def predict_profitability(
        self,
        last_hidden: torch.Tensor,
        indicator_features: torch.Tensor,
    ) -> torch.Tensor:
        """Predict profitability scores for candidate assets.

        Args:
            last_hidden: (batch, dim) hidden state at last sequence position.
            indicator_features: (n_candidates, n_indicators) per-asset indicators.

        Returns:
            (batch, n_candidates) profitability scores.
        """
        batch_size = last_hidden.shape[0]
        number_of_candidates = indicator_features.shape[0]

        hidden_expanded = last_hidden.unsqueeze(1).expand(-1, number_of_candidates, -1)
        indicators_expanded = indicator_features.unsqueeze(0).expand(batch_size, -1, -1)

        combined = torch.cat([hidden_expanded, indicators_expanded], dim=-1)
        return self.profitability_head(combined).squeeze(-1)


class HybridDualHeadRecommender(TiSASRecRecommender):
    """Hybrid recommender jointly optimizing user interest and asset profitability."""

    def __init__(
        self,
        config: HybridDualHeadConfig,
        close_prices: pd.DataFrame,
        indicator_dataframe: pd.DataFrame | None = None,
    ) -> None:
        super().__init__(config)
        self._hybrid_config = config
        self._close_prices = close_prices
        self._indicator_dataframe = indicator_dataframe
        self._indicator_matrix: torch.Tensor | None = None

    @property
    def name(self) -> str:
        """Return the display name used in evaluation results."""
        return "HybridDualHead"

    def _build_model(self, number_of_assets: int) -> HybridDualHeadModel:
        """Construct a HybridDualHead model with both interest and profitability heads."""
        return HybridDualHeadModel(
            number_of_assets=number_of_assets,
            max_sequence_length=self._hybrid_config.max_sequence_length,
            embedding_dimension=self._hybrid_config.embedding_dimension,
            number_of_attention_heads=self._hybrid_config.number_of_attention_heads,
            number_of_blocks=self._hybrid_config.number_of_blocks,
            dropout_rate=self._hybrid_config.dropout_rate,
            time_bucket_count=self._hybrid_config.time_bucket_count,
            number_of_indicators=len(INDICATOR_COLUMNS),
            profitability_hidden_dimension=self._hybrid_config.profitability_hidden_dimension,
        )

    def train_on_split(self, split: TemporalSplitData, **kwargs: object) -> None:
        """Train on one split with joint interest and profitability loss."""
        raw_sequences = kwargs.get("user_sequences")
        if not isinstance(raw_sequences, dict):
            raise TypeError(
                "HybridDualHead requires user_sequences kwarg"
                " (dict[str, list[tuple[str, date]]])"
            )

        device_name = kwargs.get("device", "cpu")
        device = torch.device(str(device_name))

        typed_sequences = cast(dict[str, list[tuple[str, date]]], raw_sequences)
        truncated = truncate_sequences(
            typed_sequences, self._hybrid_config.max_sequence_length + 1
        )

        all_asset_ids = sorted(
            {asset_id for pairs in typed_sequences.values() for asset_id, _ in pairs}
        )
        self._asset_id_to_index = {
            asset_id: index for index, asset_id in enumerate(all_asset_ids)
        }
        self._index_to_asset_id = {
            index: asset_id for asset_id, index in self._asset_id_to_index.items()
        }
        self._eligible_asset_indices = {
            self._asset_id_to_index[asset_id]
            for asset_id in split.eligible_asset_ids
            if asset_id in self._asset_id_to_index
        }
        number_of_assets = len(all_asset_ids)

        self._store_user_sequences(truncated)
        self._store_user_dates(truncated)

        if truncated:
            all_dates = [d for pairs in truncated.values() for _, d in pairs]
            self._reference_date = min(all_dates)

        dataset = self._build_dataset(
            truncated, self._asset_id_to_index, number_of_assets
        )

        if len(dataset) == 0:
            return

        indicator_matrix, roi_targets = self._build_indicator_and_roi_tensors(
            split, number_of_assets, device
        )
        self._indicator_matrix = indicator_matrix

        self._model = self._build_model(number_of_assets)

        optimizer = torch.optim.Adam(
            self._model.parameters(), lr=self._hybrid_config.learning_rate
        )
        loss_lambda = self._hybrid_config.loss_lambda

        def loss_function(
            model: SASRecModel,
            batch: tuple[torch.Tensor, ...],
        ) -> torch.Tensor:
            input_ids = batch[0].to(device)
            positive_ids = batch[1].to(device)
            negative_ids = batch[2].to(device)
            relative_time_matrix = batch[3].to(device)
            absolute_time_matrix = batch[4].to(device)

            sequence_output = model(
                input_ids,
                relative_time_matrix=relative_time_matrix,
                absolute_time_matrix=absolute_time_matrix,
            )

            positive_embeddings = model.asset_embedding(positive_ids)
            negative_embeddings = model.asset_embedding(negative_ids)

            positive_scores = (sequence_output * positive_embeddings).sum(dim=-1)
            negative_scores = (sequence_output * negative_embeddings).sum(dim=-1)

            mask = (input_ids != 0).float()

            positive_loss = -torch.log(torch.sigmoid(positive_scores) + 1e-8) * mask
            negative_loss = -torch.log(1 - torch.sigmoid(negative_scores) + 1e-8) * mask

            total_mask = mask.sum()
            if total_mask == 0:
                return torch.tensor(0.0, device=device)

            interest_loss = (positive_loss + negative_loss).sum() / total_mask

            hybrid_model = cast(HybridDualHeadModel, model)
            pos_indicators = indicator_matrix[positive_ids]
            profit_input = torch.cat([sequence_output, pos_indicators], dim=-1)
            predicted_roi = hybrid_model.profitability_head(profit_input).squeeze(-1)
            actual_roi = roi_targets[positive_ids]
            profit_loss = ((predicted_roi - actual_roi) ** 2 * mask).sum() / total_mask

            return interest_loss + loss_lambda * profit_loss

        train_pytorch_model(
            model=self._model,
            dataset=dataset,
            loss_function=loss_function,
            optimizer=optimizer,
            number_of_epochs=self._hybrid_config.number_of_epochs,
            batch_size=self._hybrid_config.batch_size,
            device=device,
        )

        self._model.eval()

    def _build_indicator_and_roi_tensors(
        self,
        split: TemporalSplitData,
        number_of_assets: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pre-compute indicator features and ROI targets for all assets."""
        number_of_indicators = len(INDICATOR_COLUMNS)
        time_point_timestamp = pd.Timestamp(split.time_point)

        if self._indicator_dataframe is not None:
            indicators_df = compute_all_indicators(
                self._indicator_dataframe,
                split.eligible_asset_ids,
                time_point_timestamp,
            )
        else:
            indicators_df = pd.DataFrame(
                0.0, index=split.eligible_asset_ids, columns=INDICATOR_COLUMNS
            )
            indicators_df.index.name = "ISIN"

        indicator_matrix = torch.zeros(
            number_of_assets + 1, number_of_indicators, device=device
        )
        for asset_id in split.eligible_asset_ids:
            if asset_id in self._asset_id_to_index and asset_id in indicators_df.index:
                index = self._asset_id_to_index[asset_id]
                indicator_matrix[index + 1] = torch.tensor(
                    indicators_df.loc[asset_id].to_numpy(), dtype=torch.float32
                )

        price_lookup = build_price_lookup(
            self._close_prices,
            split.time_point,
            split.test_end,
            split.eligible_asset_ids,
        )

        roi_targets = torch.zeros(number_of_assets + 1, device=device)
        for asset_id, (start_price, end_price) in price_lookup.items():
            if asset_id in self._asset_id_to_index and start_price != 0.0:
                index = self._asset_id_to_index[asset_id]
                roi = (end_price - start_price) / start_price / MONTHS_IN_TEST_WINDOW
                roi_targets[index + 1] = roi

        return indicator_matrix, roi_targets

    def recommend_for_user(
        self,
        user_id: str,
        excluded_assets: set[str],
        k: int = 10,
    ) -> list[str]:
        """Return top-k recommendations blending interest and profitability scores."""
        if self._model is None or self._indicator_matrix is None:
            return []

        if user_id not in self._user_sequences:
            return []

        sequence = self._user_sequences[user_id]
        padded_length = self._hybrid_config.max_sequence_length
        pad_count = padded_length - len(sequence)
        padded_input = [0] * pad_count + sequence

        device = next(self._model.parameters()).device
        input_ids = torch.tensor([padded_input], dtype=torch.long, device=device)

        time_kwargs = self._build_inference_time_matrices(user_id, device)

        candidate_embeddings = self._model.asset_embedding.weight[1:]
        candidate_indicators = self._indicator_matrix[1:]

        with torch.no_grad():
            hidden = self._model.forward(input_ids, **time_kwargs)
            last_hidden = hidden[:, -1, :]

            interest_scores = torch.matmul(
                last_hidden, candidate_embeddings.t()
            ).squeeze(0)

            hybrid_model = cast(HybridDualHeadModel, self._model)
            profit_scores = hybrid_model.predict_profitability(
                last_hidden, candidate_indicators
            ).squeeze(0)

        alpha = self._hybrid_config.inference_alpha
        normalized_interest = _normalize_scores(interest_scores)
        normalized_profit = _normalize_scores(profit_scores)
        final_scores = alpha * normalized_interest + (1 - alpha) * normalized_profit

        for asset_index in range(len(final_scores)):
            if asset_index not in self._eligible_asset_indices:
                final_scores[asset_index] = float("-inf")

        for asset_id in excluded_assets:
            if asset_id in self._asset_id_to_index:
                final_scores[self._asset_id_to_index[asset_id]] = float("-inf")

        top_indices = torch.topk(
            final_scores, min(k, len(final_scores)), sorted=True
        ).indices

        recommendations: list[str] = []
        for index in top_indices:
            asset_index = index.item()
            if final_scores[asset_index].item() == float("-inf"):
                break
            asset_id = self._index_to_asset_id.get(asset_index)
            if asset_id is not None:
                recommendations.append(asset_id)

        return recommendations
