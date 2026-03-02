"""Time-Interval-aware Self-Attentive Sequential Recommendation (TiSASRec) model and recommender."""

from datetime import date
from typing import cast

import torch
import torch.nn as nn

from src.config.schemas import TemporalSplitData
from src.config.settings import HybridDualHeadConfig, TiSASRecConfig
from src.data.sequences import (
    bucket_time_values,
    compute_absolute_positions,
    truncate_sequences,
)
from src.models.sasrec import SASRecDataset, SASRecModel, SASRecRecommender
from src.models.train import train_pytorch_model


class TiSASRecModel(SASRecModel):
    """Time-interval-aware self-attentive sequential recommendation model."""

    def __init__(
        self,
        number_of_assets: int,
        max_sequence_length: int,
        embedding_dimension: int,
        number_of_attention_heads: int,
        number_of_blocks: int,
        dropout_rate: float,
        time_bucket_count: int,
    ) -> None:
        super().__init__(
            number_of_assets=number_of_assets,
            max_sequence_length=max_sequence_length,
            embedding_dimension=embedding_dimension,
            number_of_attention_heads=number_of_attention_heads,
            number_of_blocks=number_of_blocks,
            dropout_rate=dropout_rate,
        )
        self.time_bucket_count = time_bucket_count

        self.relative_time_embedding = nn.Embedding(
            time_bucket_count, number_of_attention_heads
        )
        self.absolute_time_embedding = nn.Embedding(
            time_bucket_count, number_of_attention_heads
        )

    def forward(self, input_ids: torch.Tensor, **kwargs: object) -> torch.Tensor:
        """Run the TiSASRec encoder, forwarding any time-matrix kwargs to the attention layers."""
        return super().forward(input_ids, **kwargs)

    def _compute_attention_scores(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        **kwargs: object,
    ) -> torch.Tensor:
        """Compute attention scores with optional time-interval bias terms."""
        base_scores = torch.matmul(queries, keys.transpose(-2, -1))

        relative_time_matrix = kwargs.get("relative_time_matrix")
        absolute_time_matrix = kwargs.get("absolute_time_matrix")

        if relative_time_matrix is None and absolute_time_matrix is None:
            return base_scores

        if relative_time_matrix is not None:
            relative_time_matrix = cast(torch.Tensor, relative_time_matrix)
            relative_bias = self.relative_time_embedding(relative_time_matrix)
            relative_bias = relative_bias.permute(0, 3, 1, 2)
            base_scores = base_scores + relative_bias

        if absolute_time_matrix is not None:
            absolute_time_matrix = cast(torch.Tensor, absolute_time_matrix)
            absolute_bias = self.absolute_time_embedding(absolute_time_matrix)
            absolute_bias = absolute_bias.permute(0, 3, 1, 2)
            base_scores = base_scores + absolute_bias

        return base_scores

    def predict(
        self,
        input_ids: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        **kwargs: object,
    ) -> torch.Tensor:
        """Score candidates using the last position's hidden state with time awareness."""
        hidden = self.forward(input_ids, **kwargs)
        last_hidden = hidden[:, -1, :]
        return torch.matmul(last_hidden, candidate_embeddings.t())


class TiSASRecDataset(SASRecDataset):
    """Dataset that additionally returns time-interval matrices for TiSASRec."""

    def __init__(
        self,
        user_sequences: dict[str, list[tuple[str, date]]],
        asset_id_to_index: dict[str, int],
        max_sequence_length: int,
        number_of_assets: int,
        user_sequence_dates: dict[str, list[date]],
        time_bucket_count: int,
        reference_date: date,
    ) -> None:
        super().__init__(
            user_sequences=user_sequences,
            asset_id_to_index=asset_id_to_index,
            max_sequence_length=max_sequence_length,
            number_of_assets=number_of_assets,
        )
        self._time_bucket_count = time_bucket_count
        self._time_matrices: list[tuple[list[list[int]], list[list[int]]]] = []

        for user_id, asset_date_pairs in user_sequences.items():
            valid_pairs = [
                (asset_id, d)
                for asset_id, d in asset_date_pairs
                if asset_id in asset_id_to_index
            ]

            if len(valid_pairs) < 2:
                continue

            valid_pairs = valid_pairs[-max_sequence_length - 1 :]
            input_dates = [d for _, d in valid_pairs[:-1]]

            if user_id not in user_sequence_dates:
                continue

            absolute_positions = compute_absolute_positions(input_dates, reference_date)

            bucketed_absolute = bucket_time_values(
                absolute_positions, time_bucket_count
            )

            sequence_length = len(input_dates)
            relative_matrix: list[list[int]] = []
            for i in range(sequence_length):
                row: list[int] = []
                for j in range(sequence_length):
                    if i == j:
                        row.append(0)
                    else:
                        interval_days = abs((input_dates[i] - input_dates[j]).days)
                        bucketed = bucket_time_values(
                            [interval_days], time_bucket_count
                        )
                        row.append(bucketed[0])
                relative_matrix.append(row)

            absolute_matrix: list[list[int]] = []
            for i in range(sequence_length):
                row = []
                for j in range(sequence_length):
                    row.append(bucketed_absolute[j])
                absolute_matrix.append(row)

            self._time_matrices.append((relative_matrix, absolute_matrix))

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return padded (input, positive, negative, relative_time, absolute_time) tensors."""
        input_ids, positive_ids, negative_ids = super().__getitem__(index)

        padded_length = self._max_sequence_length
        relative_matrix, absolute_matrix = self._time_matrices[index]
        sequence_length = len(relative_matrix)
        pad_count = padded_length - sequence_length

        padded_relative = _pad_time_matrix(relative_matrix, pad_count, padded_length)
        padded_absolute = _pad_time_matrix(absolute_matrix, pad_count, padded_length)

        return (
            input_ids,
            positive_ids,
            negative_ids,
            torch.tensor(padded_relative, dtype=torch.long),
            torch.tensor(padded_absolute, dtype=torch.long),
        )


def _pad_time_matrix(
    matrix: list[list[int]], pad_count: int, padded_length: int
) -> list[list[int]]:
    """Left-pad a time matrix with zeros to match the padded sequence length."""
    padded: list[list[int]] = []
    for _ in range(pad_count):
        padded.append([0] * padded_length)
    for row in matrix:
        padded_row = [0] * pad_count + row
        padded.append(padded_row)
    return padded


class TiSASRecRecommender(SASRecRecommender):
    """TiSASRec sequential recommender with time-interval-aware attention."""

    def __init__(self, config: TiSASRecConfig | HybridDualHeadConfig) -> None:
        super().__init__(config)
        self._tisasrec_config = config
        self._user_dates: dict[str, list[date]] = {}
        self._reference_date: date = date(2000, 1, 1)

    @property
    def name(self) -> str:
        """Return the display name used in evaluation results."""
        return "TiSASRec"

    def _build_model(self, number_of_assets: int) -> TiSASRecModel:
        """Construct a TiSASRec model with time-interval embeddings."""
        return TiSASRecModel(
            number_of_assets=number_of_assets,
            max_sequence_length=self._tisasrec_config.max_sequence_length,
            embedding_dimension=self._tisasrec_config.embedding_dimension,
            number_of_attention_heads=self._tisasrec_config.number_of_attention_heads,
            number_of_blocks=self._tisasrec_config.number_of_blocks,
            dropout_rate=self._tisasrec_config.dropout_rate,
            time_bucket_count=self._tisasrec_config.time_bucket_count,
        )

    def _build_dataset(
        self,
        user_sequences: dict[str, list[tuple[str, date]]],
        asset_id_to_index: dict[str, int],
        number_of_assets: int,
    ) -> TiSASRecDataset:
        """Construct a training dataset with time-interval matrices."""
        user_sequence_dates: dict[str, list[date]] = {
            user_id: [d for _, d in pairs] for user_id, pairs in user_sequences.items()
        }
        return TiSASRecDataset(
            user_sequences=user_sequences,
            asset_id_to_index=asset_id_to_index,
            max_sequence_length=self._tisasrec_config.max_sequence_length,
            number_of_assets=number_of_assets,
            user_sequence_dates=user_sequence_dates,
            time_bucket_count=self._tisasrec_config.time_bucket_count,
            reference_date=self._reference_date,
        )

    def train_on_split(self, split: TemporalSplitData, **kwargs: object) -> None:
        """Train TiSASRec on one temporal split with time-aware BPR loss."""
        raw_sequences = kwargs.get("user_sequences")
        if not isinstance(raw_sequences, dict):
            raise TypeError(
                "TiSASRec requires user_sequences kwarg"
                " (dict[str, list[tuple[str, date]]])"
            )

        device_name = kwargs.get("device", "cpu")
        device = torch.device(str(device_name))

        typed_sequences = cast(dict[str, list[tuple[str, date]]], raw_sequences)
        truncated = truncate_sequences(
            typed_sequences, self._tisasrec_config.max_sequence_length + 1
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

        self._model = self._build_model(number_of_assets)

        optimizer = torch.optim.Adam(
            self._model.parameters(), lr=self._tisasrec_config.learning_rate
        )

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

            return (positive_loss + negative_loss).sum() / total_mask

        train_pytorch_model(
            model=self._model,
            dataset=dataset,
            loss_function=loss_function,
            optimizer=optimizer,
            number_of_epochs=self._tisasrec_config.number_of_epochs,
            batch_size=self._tisasrec_config.batch_size,
            device=device,
        )

        self._model.eval()

    def _store_user_dates(self, sequences: dict[str, list[tuple[str, date]]]) -> None:
        """Store per-user date sequences for inference-time matrix construction."""
        self._user_dates = {}
        for user_id, asset_date_pairs in sequences.items():
            dates = [
                d
                for asset_id, d in asset_date_pairs
                if asset_id in self._asset_id_to_index
            ]
            if dates:
                self._user_dates[user_id] = dates[
                    -self._tisasrec_config.max_sequence_length :
                ]

    def recommend_for_user(
        self,
        user_id: str,
        excluded_assets: set[str],
        k: int = 10,
    ) -> list[str]:
        """Return top-k eligible asset recommendations using time-aware scoring."""
        if self._model is None:
            return []

        if user_id not in self._user_sequences:
            return []

        sequence = self._user_sequences[user_id]
        padded_length = self._tisasrec_config.max_sequence_length
        pad_count = padded_length - len(sequence)
        padded_input = [0] * pad_count + sequence

        device = next(self._model.parameters()).device
        input_ids = torch.tensor([padded_input], dtype=torch.long, device=device)

        time_kwargs = self._build_inference_time_matrices(user_id, device)

        candidate_embeddings = self._model.asset_embedding.weight[1:]

        with torch.no_grad():
            model = cast(TiSASRecModel, self._model)
            scores = model.predict(
                input_ids, candidate_embeddings, **time_kwargs
            ).squeeze(0)

        for asset_index in range(len(scores)):
            if asset_index not in self._eligible_asset_indices:
                scores[asset_index] = float("-inf")

        for asset_id in excluded_assets:
            if asset_id in self._asset_id_to_index:
                scores[self._asset_id_to_index[asset_id]] = float("-inf")

        top_indices = torch.topk(scores, min(k, len(scores)), sorted=True).indices

        recommendations: list[str] = []
        for index in top_indices:
            asset_index = index.item()
            if scores[asset_index].item() == float("-inf"):
                break
            asset_id = self._index_to_asset_id.get(asset_index)
            if asset_id is not None:
                recommendations.append(asset_id)

        return recommendations

    def _build_inference_time_matrices(
        self, user_id: str, device: torch.device
    ) -> dict[str, torch.Tensor]:
        """Construct time matrices for a single user at inference time."""
        if user_id not in self._user_dates:
            return {}

        dates = self._user_dates[user_id]
        padded_length = self._tisasrec_config.max_sequence_length
        sequence_length = len(dates)
        pad_count = padded_length - sequence_length
        time_bucket_count = self._tisasrec_config.time_bucket_count

        bucketed_absolute = bucket_time_values(
            compute_absolute_positions(dates, self._reference_date),
            time_bucket_count,
        )

        relative_matrix: list[list[int]] = []
        for i in range(sequence_length):
            row: list[int] = []
            for j in range(sequence_length):
                if i == j:
                    row.append(0)
                else:
                    interval_days = abs((dates[i] - dates[j]).days)
                    bucketed = bucket_time_values([interval_days], time_bucket_count)
                    row.append(bucketed[0])
            relative_matrix.append(row)

        absolute_matrix: list[list[int]] = []
        for i in range(sequence_length):
            row = [bucketed_absolute[j] for j in range(sequence_length)]
            absolute_matrix.append(row)

        padded_relative = _pad_time_matrix(relative_matrix, pad_count, padded_length)
        padded_absolute = _pad_time_matrix(absolute_matrix, pad_count, padded_length)

        return {
            "relative_time_matrix": torch.tensor(
                [padded_relative], dtype=torch.long, device=device
            ),
            "absolute_time_matrix": torch.tensor(
                [padded_absolute], dtype=torch.long, device=device
            ),
        }
