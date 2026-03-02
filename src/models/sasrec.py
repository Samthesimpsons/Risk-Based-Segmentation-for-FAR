"""Self-Attentive Sequential Recommendation (SASRec) model and recommender."""

import random
from collections.abc import Callable
from datetime import date
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config.schemas import TemporalSplitData
from src.config.settings import HybridDualHeadConfig, SASRecConfig, TiSASRecConfig
from src.data.sequences import truncate_sequences
from src.models.train import train_pytorch_model

AttentionScoreFunction = Callable[..., torch.Tensor]


class SASRecModel(nn.Module):
    """Self-Attentive Sequential Recommendation transformer encoder."""

    def __init__(
        self,
        number_of_assets: int,
        max_sequence_length: int,
        embedding_dimension: int,
        number_of_attention_heads: int,
        number_of_blocks: int,
        dropout_rate: float,
    ) -> None:
        super().__init__()
        self.number_of_assets = number_of_assets
        self.max_sequence_length = max_sequence_length
        self.embedding_dimension = embedding_dimension
        self.number_of_attention_heads = number_of_attention_heads

        self.asset_embedding = nn.Embedding(
            number_of_assets + 1, embedding_dimension, padding_idx=0
        )
        self.position_embedding = nn.Embedding(max_sequence_length, embedding_dimension)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embedding_dimension, number_of_attention_heads, dropout_rate
                )
                for _ in range(number_of_blocks)
            ]
        )

        self.layer_norm = nn.LayerNorm(embedding_dimension)
        self.dropout = nn.Dropout(dropout_rate)

        self._register_causal_mask(max_sequence_length)

    def _register_causal_mask(self, size: int) -> None:
        """Register an upper-triangular causal mask as a non-learnable buffer."""
        causal_mask = torch.triu(torch.ones(size, size, dtype=torch.bool), diagonal=1)
        self.register_buffer("causal_mask", causal_mask)

    def forward(self, input_ids: torch.Tensor, **kwargs: object) -> torch.Tensor:
        """Encode a batch of padded asset-index sequences into per-position hidden states."""
        batch_size, sequence_length = input_ids.shape

        positions = torch.arange(sequence_length, device=input_ids.device)
        hidden = self.asset_embedding(input_ids) + self.position_embedding(positions)
        hidden = self.layer_norm(hidden)
        hidden = self.dropout(hidden)

        padding_mask = input_ids == 0
        causal_mask = cast(torch.Tensor, self.causal_mask)
        causal_mask = causal_mask[:sequence_length, :sequence_length]

        for block in self.blocks:
            hidden = block(
                hidden,
                causal_mask=causal_mask,
                padding_mask=padding_mask,
                attention_score_fn=self._compute_attention_scores,
                **kwargs,
            )

        return hidden

    def _compute_attention_scores(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        **kwargs: object,
    ) -> torch.Tensor:
        """Compute raw attention scores. Override point for TiSASRec."""
        return torch.matmul(queries, keys.transpose(-2, -1))

    def predict(
        self,
        input_ids: torch.Tensor,
        candidate_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Score candidates using the last position's hidden state.

        Args:
            input_ids: (batch, seq_len) asset indices.
            candidate_embeddings: (number_of_candidates, dim) asset embeddings.

        Returns:
            (batch, number_of_candidates) scores.
        """
        hidden = self.forward(input_ids)
        last_hidden = hidden[:, -1, :]
        return torch.matmul(last_hidden, candidate_embeddings.t())


class TransformerBlock(nn.Module):
    """Single transformer block with multi-head attention and FFN."""

    def __init__(
        self,
        embedding_dimension: int,
        number_of_attention_heads: int,
        dropout_rate: float,
    ) -> None:
        super().__init__()
        self.attention_layer_norm = nn.LayerNorm(embedding_dimension)
        self.ffn_layer_norm = nn.LayerNorm(embedding_dimension)

        self.query_projection = nn.Linear(embedding_dimension, embedding_dimension)
        self.key_projection = nn.Linear(embedding_dimension, embedding_dimension)
        self.value_projection = nn.Linear(embedding_dimension, embedding_dimension)
        self.output_projection = nn.Linear(embedding_dimension, embedding_dimension)

        self.ffn = nn.Sequential(
            nn.Linear(embedding_dimension, embedding_dimension * 4),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(embedding_dimension * 4, embedding_dimension),
            nn.Dropout(dropout_rate),
        )

        self.attention_dropout = nn.Dropout(dropout_rate)

        self.number_of_heads = number_of_attention_heads
        self.head_dimension = embedding_dimension // number_of_attention_heads

    def forward(
        self,
        hidden: torch.Tensor,
        causal_mask: torch.Tensor,
        padding_mask: torch.Tensor,
        attention_score_fn: AttentionScoreFunction,
        **kwargs: object,
    ) -> torch.Tensor:
        """Apply multi-head self-attention followed by a feed-forward network."""
        batch_size, sequence_length, _ = hidden.shape

        residual = hidden
        hidden = self.attention_layer_norm(hidden)

        queries = self._split_heads(self.query_projection(hidden), batch_size)
        keys = self._split_heads(self.key_projection(hidden), batch_size)
        values = self._split_heads(self.value_projection(hidden), batch_size)

        scores = attention_score_fn(queries, keys, **kwargs)
        scores = scores / (self.head_dimension**0.5)

        combined_mask = causal_mask.unsqueeze(0).unsqueeze(0)
        padding_expanded = padding_mask.unsqueeze(1).unsqueeze(2)
        combined_mask = combined_mask | padding_expanded

        scores = scores.masked_fill(combined_mask, float("-inf"))

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = attention_weights.masked_fill(combined_mask, 0.0)
        attention_weights = self.attention_dropout(attention_weights)

        context = torch.matmul(attention_weights, values)
        context = self._merge_heads(context, batch_size)
        hidden = self.output_projection(context) + residual

        residual = hidden
        hidden = self.ffn_layer_norm(hidden)
        hidden = self.ffn(hidden) + residual

        return hidden

    def _split_heads(self, tensor: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Reshape tensor from (batch, seq, dim) to (batch, heads, seq, head_dim)."""
        return tensor.view(
            batch_size, -1, self.number_of_heads, self.head_dimension
        ).transpose(1, 2)

    def _merge_heads(self, tensor: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Reshape tensor from (batch, heads, seq, head_dim) back to (batch, seq, dim)."""
        return (
            tensor.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.number_of_heads * self.head_dimension)
        )


class SASRecDataset(torch.utils.data.Dataset):
    """Dataset yielding (input_ids, positive_ids, negative_ids) for SASRec training."""

    def __init__(
        self,
        user_sequences: dict[str, list[tuple[str, date]]],
        asset_id_to_index: dict[str, int],
        max_sequence_length: int,
        number_of_assets: int,
    ) -> None:
        self._max_sequence_length = max_sequence_length
        self._number_of_assets = number_of_assets
        self._all_asset_indices = set(range(1, number_of_assets + 1))
        self._samples: list[tuple[list[int], list[int], set[int]]] = []

        for asset_date_pairs in user_sequences.values():
            index_sequence = [
                asset_id_to_index[asset_id] + 1
                for asset_id, _ in asset_date_pairs
                if asset_id in asset_id_to_index
            ]

            if len(index_sequence) < 2:
                continue

            index_sequence = index_sequence[-max_sequence_length - 1 :]
            input_indices = index_sequence[:-1]
            target_indices = index_sequence[1:]
            user_assets = set(index_sequence)

            self._samples.append((input_indices, target_indices, user_assets))

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        """Return padded (input_ids, positive_ids, negative_ids) tensors for one user."""
        input_indices, target_indices, user_assets = self._samples[index]

        padded_length = self._max_sequence_length
        pad_count = padded_length - len(input_indices)

        input_ids = [0] * pad_count + input_indices
        positive_ids = [0] * pad_count + target_indices

        negative_candidates = list(self._all_asset_indices - user_assets)
        negative_ids: list[int] = []
        for _ in range(padded_length):
            if negative_candidates:
                negative_ids.append(random.choice(negative_candidates))
            else:
                negative_ids.append(random.randint(1, self._number_of_assets))

        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(positive_ids, dtype=torch.long),
            torch.tensor(negative_ids, dtype=torch.long),
        )


class SASRecRecommender:
    """SASRec sequential recommender implementing the Recommender protocol."""

    def __init__(
        self, config: SASRecConfig | TiSASRecConfig | HybridDualHeadConfig
    ) -> None:
        self._config = config
        self._model: SASRecModel | None = None
        self._asset_id_to_index: dict[str, int] = {}
        self._index_to_asset_id: dict[int, str] = {}
        self._user_sequences: dict[str, list[int]] = {}
        self._eligible_asset_indices: set[int] = set()

    @property
    def name(self) -> str:
        """Return the display name used in evaluation results."""
        return "SASRec"

    def _build_model(self, number_of_assets: int) -> SASRecModel:
        """Construct a SASRec model with the current config."""
        return SASRecModel(
            number_of_assets=number_of_assets,
            max_sequence_length=self._config.max_sequence_length,
            embedding_dimension=self._config.embedding_dimension,
            number_of_attention_heads=self._config.number_of_attention_heads,
            number_of_blocks=self._config.number_of_blocks,
            dropout_rate=self._config.dropout_rate,
        )

    def _build_dataset(
        self,
        user_sequences: dict[str, list[tuple[str, date]]],
        asset_id_to_index: dict[str, int],
        number_of_assets: int,
    ) -> SASRecDataset:
        """Construct a training dataset from user sequences."""
        return SASRecDataset(
            user_sequences=user_sequences,
            asset_id_to_index=asset_id_to_index,
            max_sequence_length=self._config.max_sequence_length,
            number_of_assets=number_of_assets,
        )

    def train_on_split(self, split: TemporalSplitData, **kwargs: object) -> None:
        """Train the SASRec model on one temporal split using BPR-style loss."""
        raw_sequences = kwargs.get("user_sequences")
        if not isinstance(raw_sequences, dict):
            raise TypeError(
                "SASRec requires user_sequences kwarg"
                " (dict[str, list[tuple[str, date]]])"
            )

        device_name = kwargs.get("device", "cpu")
        device = torch.device(str(device_name))

        typed_sequences = cast(dict[str, list[tuple[str, date]]], raw_sequences)
        truncated = truncate_sequences(
            typed_sequences, self._config.max_sequence_length + 1
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

        dataset = self._build_dataset(
            truncated, self._asset_id_to_index, number_of_assets
        )

        if len(dataset) == 0:
            return

        self._model = self._build_model(number_of_assets)

        optimizer = torch.optim.Adam(
            self._model.parameters(), lr=self._config.learning_rate
        )

        def loss_function(
            model: SASRecModel,
            batch: tuple[torch.Tensor, ...],
        ) -> torch.Tensor:
            input_ids = batch[0].to(device)
            positive_ids = batch[1].to(device)
            negative_ids = batch[2].to(device)

            sequence_output = model(input_ids)

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
            number_of_epochs=self._config.number_of_epochs,
            batch_size=self._config.batch_size,
            device=device,
        )

        self._model.eval()

    def _store_user_sequences(
        self, sequences: dict[str, list[tuple[str, date]]]
    ) -> None:
        """Convert and cache user sequences as truncated index lists for inference."""
        self._user_sequences = {}
        for user_id, asset_date_pairs in sequences.items():
            index_sequence = [
                self._asset_id_to_index[asset_id] + 1
                for asset_id, _ in asset_date_pairs
                if asset_id in self._asset_id_to_index
            ]
            if index_sequence:
                self._user_sequences[user_id] = index_sequence[
                    -self._config.max_sequence_length :
                ]

    def recommend_for_user(
        self,
        user_id: str,
        excluded_assets: set[str],
        k: int = 10,
    ) -> list[str]:
        """Return top-k eligible asset recommendations, excluding already-acquired ones."""
        if self._model is None:
            return []

        if user_id not in self._user_sequences:
            return []

        sequence = self._user_sequences[user_id]
        padded_length = self._config.max_sequence_length
        pad_count = padded_length - len(sequence)
        padded_input = [0] * pad_count + sequence

        device = next(self._model.parameters()).device
        input_ids = torch.tensor([padded_input], dtype=torch.long, device=device)

        candidate_embeddings = self._model.asset_embedding.weight[1:]

        with torch.no_grad():
            scores = self._model.predict(input_ids, candidate_embeddings).squeeze(0)

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
