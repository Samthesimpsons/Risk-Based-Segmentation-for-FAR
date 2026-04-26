"""Profile-Coherent LightGCN model and training baseline."""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import LGConv
from torch_geometric.nn.conv.gcn_conv import gcn_norm

from src.config.schemas import CustomerProfile, TemporalSplitData
from src.config.settings import ProfileCoherentLightGCNConfig
from src.utils.profile_coherence import NUMBER_OF_RISK_BANDS


@dataclass(frozen=True)
class ProfileVocabulary:
    """Token lists, index maps, and padding indices for profile features."""

    unknown_token: str
    unknown_index: int
    customer_type_values: tuple[str, ...]
    customer_type_to_index: dict[str, int]
    investment_capacity_values: tuple[str, ...]
    investment_capacity_to_index: dict[str, int]
    risk_band_padding_index: int


def _build_profile_vocabulary() -> ProfileVocabulary:
    """Construct the default profile vocabulary used for embedding lookups."""
    unknown_token = "__unknown__"
    customer_type_values: tuple[str, ...] = (
        unknown_token,
        "Mass",
        "Premium",
        "Professional",
        "Legal Entity",
        "Inactive",
    )
    investment_capacity_values: tuple[str, ...] = (
        unknown_token,
        "CAP_LT30K",
        "CAP_30K_80K",
        "CAP_80K_300K",
        "CAP_GT300K",
        "Predicted_CAP_LT30K",
        "Predicted_CAP_30K_80K",
        "Predicted_CAP_80K_300K",
        "Predicted_CAP_GT300K",
    )
    return ProfileVocabulary(
        unknown_token=unknown_token,
        unknown_index=0,
        customer_type_values=customer_type_values,
        customer_type_to_index={
            value: index for index, value in enumerate(customer_type_values)
        },
        investment_capacity_values=investment_capacity_values,
        investment_capacity_to_index={
            value: index for index, value in enumerate(investment_capacity_values)
        },
        risk_band_padding_index=NUMBER_OF_RISK_BANDS,
    )


PROFILE_VOCABULARY = _build_profile_vocabulary()


class ProfileCoherentLightGCNModel(nn.Module):
    """LightGCN extended with optional profile-conditioning embeddings."""

    def __init__(
        self,
        number_of_users: int,
        number_of_assets: int,
        embedding_dimension: int,
        number_of_layers: int,
        profile_embedding_dimension: int,
        profile_embedding_enabled: bool,
    ) -> None:
        """Initialise base, profile, and convolution embeddings."""
        super().__init__()
        self.number_of_users = number_of_users
        self.number_of_assets = number_of_assets
        self.number_of_layers = number_of_layers
        self.profile_embedding_enabled = profile_embedding_enabled

        self.user_embedding = nn.Embedding(number_of_users, embedding_dimension)
        self.asset_embedding = nn.Embedding(number_of_assets, embedding_dimension)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.asset_embedding.weight)

        if profile_embedding_enabled:
            self.risk_band_embedding = nn.Embedding(
                NUMBER_OF_RISK_BANDS + 1,
                profile_embedding_dimension,
            )
            self.customer_type_embedding = nn.Embedding(
                len(PROFILE_VOCABULARY.customer_type_values),
                profile_embedding_dimension,
            )
            self.investment_capacity_embedding = nn.Embedding(
                len(PROFILE_VOCABULARY.investment_capacity_values),
                profile_embedding_dimension,
            )
            nn.init.xavier_uniform_(self.risk_band_embedding.weight)
            nn.init.xavier_uniform_(self.customer_type_embedding.weight)
            nn.init.xavier_uniform_(self.investment_capacity_embedding.weight)
            self.profile_projection = nn.Linear(
                profile_embedding_dimension, embedding_dimension, bias=False
            )
            nn.init.xavier_uniform_(self.profile_projection.weight)
        else:
            self.risk_band_embedding = None
            self.customer_type_embedding = None
            self.investment_capacity_embedding = None
            self.profile_projection = None

        self.convolution_layers = nn.ModuleList(
            [LGConv(normalize=False) for _ in range(number_of_layers)]
        )

    def _profile_offset(
        self,
        risk_band_indices: torch.Tensor,
        customer_type_indices: torch.Tensor,
        investment_capacity_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Return the per-user profile offset added to the base user embedding."""
        if not self.profile_embedding_enabled:
            return torch.zeros_like(self.user_embedding.weight)

        assert self.risk_band_embedding is not None
        assert self.customer_type_embedding is not None
        assert self.investment_capacity_embedding is not None
        assert self.profile_projection is not None

        risk_vectors = self.risk_band_embedding(risk_band_indices)
        type_vectors = self.customer_type_embedding(customer_type_indices)
        capacity_vectors = self.investment_capacity_embedding(
            investment_capacity_indices
        )
        summed = risk_vectors + type_vectors + capacity_vectors
        return self.profile_projection(summed)

    def forward(
        self,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        risk_band_indices: torch.Tensor,
        customer_type_indices: torch.Tensor,
        investment_capacity_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return final user and asset embeddings after LightGCN propagation."""
        user_initial = self.user_embedding.weight + self._profile_offset(
            risk_band_indices, customer_type_indices, investment_capacity_indices
        )
        all_embeddings = torch.cat([user_initial, self.asset_embedding.weight], dim=0)

        layer_outputs = [all_embeddings]
        current_embeddings = all_embeddings

        for convolution_layer in self.convolution_layers:
            current_embeddings = convolution_layer(
                current_embeddings, edge_index, edge_weight
            )
            layer_outputs.append(current_embeddings)

        final_embeddings = torch.stack(layer_outputs, dim=0).mean(dim=0)

        user_embeddings = final_embeddings[: self.number_of_users]
        asset_embeddings = final_embeddings[self.number_of_users :]
        return user_embeddings, asset_embeddings

    def bpr_with_profile_loss(
        self,
        user_indices: torch.Tensor,
        positive_asset_indices: torch.Tensor,
        negative_asset_indices: torch.Tensor,
        positive_discordance: torch.Tensor,
        user_embeddings: torch.Tensor,
        asset_embeddings: torch.Tensor,
        regularization_weight: float,
        profile_coherence_lambda: float,
    ) -> torch.Tensor:
        """Compute BPR loss with an optional profile-coherence regulariser."""
        user_vectors = user_embeddings[user_indices]
        positive_vectors = asset_embeddings[positive_asset_indices]
        negative_vectors = asset_embeddings[negative_asset_indices]

        positive_scores = (user_vectors * positive_vectors).sum(dim=1)
        negative_scores = (user_vectors * negative_vectors).sum(dim=1)

        loss = F.softplus(negative_scores - positive_scores).mean()

        if regularization_weight > 0.0:
            initial_user = self.user_embedding.weight[user_indices]
            initial_positive = self.asset_embedding.weight[positive_asset_indices]
            initial_negative = self.asset_embedding.weight[negative_asset_indices]
            l2_term = (
                initial_user.norm(2).pow(2)
                + initial_positive.norm(2).pow(2)
                + initial_negative.norm(2).pow(2)
            ) / (2 * user_indices.shape[0])
            loss = loss + regularization_weight * l2_term

        if profile_coherence_lambda > 0.0:
            sigmoid_positive = torch.sigmoid(positive_scores)
            coherence_penalty = (positive_discordance * sigmoid_positive).mean()
            loss = loss + profile_coherence_lambda * coherence_penalty

        return loss


class _BPRWithProfileDataset(torch.utils.data.Dataset):
    """Dataset of user, positive, negative, and discordance tuples for BPR training."""

    def __init__(
        self,
        training_interactions: dict[str, set[str]],
        customer_id_to_index: dict[str, int],
        asset_id_to_index: dict[str, int],
        number_of_assets: int,
        customer_band_by_index: dict[int, int | None],
        asset_band_by_index: dict[int, int],
        squared: bool,
    ) -> None:
        """Build training tuples with positive-pair discordance values."""
        self._all_asset_indices = set(range(number_of_assets))
        self._triples: list[tuple[int, int, int, float]] = []

        for customer_id, asset_ids in training_interactions.items():
            if customer_id not in customer_id_to_index:
                continue
            user_index = customer_id_to_index[customer_id]
            customer_band = customer_band_by_index.get(user_index)

            positive_indices: set[int] = set()
            for asset_id in asset_ids:
                if asset_id not in asset_id_to_index:
                    continue
                positive_indices.add(asset_id_to_index[asset_id])

            negative_candidates = list(self._all_asset_indices - positive_indices)

            for positive_index in positive_indices:
                if negative_candidates:
                    negative_index = random.choice(negative_candidates)
                else:
                    negative_index = random.randrange(len(self._all_asset_indices))

                discordance = self._compute_discordance(
                    customer_band,
                    asset_band_by_index.get(positive_index),
                    squared=squared,
                )
                self._triples.append(
                    (user_index, positive_index, negative_index, discordance)
                )

    @staticmethod
    def _compute_discordance(
        customer_band: int | None, asset_band: int | None, squared: bool
    ) -> float:
        """Return the ordinal band distance between customer and asset bands."""
        if customer_band is None or asset_band is None:
            return 0.0
        distance = abs(customer_band - asset_band)
        return float(distance * distance) if squared else float(distance)

    def __len__(self) -> int:
        """Return the number of training tuples."""
        return len(self._triples)

    def __getitem__(self, index: int) -> tuple[int, int, int, float]:
        """Return the tuple at the given index."""
        return self._triples[index]


def _drop_edges_with_inverted_scaling(
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    keep_probability: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Inverted-dropout edge sampling for graph regularisation."""
    if keep_probability >= 1.0:
        return edge_index, edge_weight

    number_of_edges = edge_index.size(1)
    keep_mask = torch.rand(number_of_edges, device=edge_index.device) < keep_probability

    if not keep_mask.any():
        return edge_index, edge_weight

    kept_edge_index = edge_index[:, keep_mask]
    kept_edge_weight = edge_weight[keep_mask] / keep_probability
    return kept_edge_index, kept_edge_weight


class ProfileCoherentLightGCNBaseline:
    """Profile-Coherent LightGCN recommender baseline."""

    def __init__(
        self,
        config: ProfileCoherentLightGCNConfig,
        customer_profiles: dict[str, CustomerProfile],
        asset_risk_classes: dict[str, int],
    ) -> None:
        """Store configuration, profile, and asset risk inputs."""
        self._config = config
        self._customer_profiles = customer_profiles
        self._asset_risk_classes = asset_risk_classes
        self._model: ProfileCoherentLightGCNModel | None = None
        self._user_embeddings: torch.Tensor | None = None
        self._asset_embeddings: torch.Tensor | None = None
        self._customer_id_to_index: dict[str, int] = {}
        self._asset_id_to_index: dict[str, int] = {}
        self._index_to_asset_id: dict[int, str] = {}
        self._eligible_asset_indices: set[int] = set()

    @property
    def name(self) -> str:
        """Return the display name used in evaluation results."""
        return "ProfileCoherentLightGCN"

    def train_on_split(self, split: TemporalSplitData, **kwargs: object) -> None:
        """Train Profile-Coherent LightGCN on the full interaction graph."""
        device_name = kwargs.get("device", "cpu")
        device = torch.device(str(device_name))

        all_customer_ids = sorted(split.training_interactions.keys())
        all_asset_ids = sorted(
            {
                asset_id
                for asset_ids in split.training_interactions.values()
                for asset_id in asset_ids
            }
        )

        self._customer_id_to_index = {
            customer_id: index for index, customer_id in enumerate(all_customer_ids)
        }
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

        number_of_users = len(self._customer_id_to_index)
        number_of_assets = len(self._asset_id_to_index)

        customer_band_by_index, customer_type_indices, capacity_indices = (
            self._build_user_profile_index_arrays(all_customer_ids)
        )
        asset_band_by_index = {
            index: self._asset_risk_classes[asset_id]
            for asset_id, index in self._asset_id_to_index.items()
            if asset_id in self._asset_risk_classes
        }

        risk_band_indices = self._materialize_risk_band_indices(
            customer_band_by_index, number_of_users
        )

        risk_band_tensor = torch.tensor(
            risk_band_indices, dtype=torch.long, device=device
        )
        type_tensor = torch.tensor(
            customer_type_indices, dtype=torch.long, device=device
        )
        capacity_tensor = torch.tensor(
            capacity_indices, dtype=torch.long, device=device
        )

        dataset = _BPRWithProfileDataset(
            training_interactions=split.training_interactions,
            customer_id_to_index=self._customer_id_to_index,
            asset_id_to_index=self._asset_id_to_index,
            number_of_assets=number_of_assets,
            customer_band_by_index=customer_band_by_index,
            asset_band_by_index=asset_band_by_index,
            squared=self._config.profile_coherence_squared,
        )

        edge_index = self._build_edge_index(split, number_of_users, device)
        number_of_nodes = number_of_users + number_of_assets
        normalized_edge_index, normalized_edge_weight = gcn_norm(
            edge_index,
            edge_weight=None,
            num_nodes=number_of_nodes,
            add_self_loops=False,
            dtype=torch.float32,
        )

        self._model = ProfileCoherentLightGCNModel(
            number_of_users=number_of_users,
            number_of_assets=number_of_assets,
            embedding_dimension=self._config.embedding_dimension,
            number_of_layers=self._config.number_of_layers,
            profile_embedding_dimension=self._config.profile_embedding_dimension,
            profile_embedding_enabled=self._config.profile_embedding_enabled,
        )
        self._model.to(device)

        optimizer = torch.optim.Adam(
            self._model.parameters(),
            lr=self._config.learning_rate,
        )

        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self._config.batch_size,
            shuffle=True,
        )

        effective_lambda = (
            self._config.profile_coherence_lambda
            if self._config.profile_coherence_enabled
            else 0.0
        )

        self._model.train()
        for _ in range(self._config.number_of_epochs):
            for batch in data_loader:
                user_indices = batch[0].to(device)
                positive_indices = batch[1].to(device)
                negative_indices = batch[2].to(device)
                positive_discordance = batch[3].to(device=device, dtype=torch.float32)

                dropped_edge_index, dropped_edge_weight = (
                    _drop_edges_with_inverted_scaling(
                        normalized_edge_index,
                        normalized_edge_weight,
                        self._config.keep_probability,
                    )
                )
                user_embeddings, asset_embeddings = self._model(
                    dropped_edge_index,
                    dropped_edge_weight,
                    risk_band_tensor,
                    type_tensor,
                    capacity_tensor,
                )

                loss = self._model.bpr_with_profile_loss(
                    user_indices=user_indices,
                    positive_asset_indices=positive_indices,
                    negative_asset_indices=negative_indices,
                    positive_discordance=positive_discordance,
                    user_embeddings=user_embeddings,
                    asset_embeddings=asset_embeddings,
                    regularization_weight=self._config.weight_decay,
                    profile_coherence_lambda=effective_lambda,
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        self._model.eval()
        with torch.no_grad():
            self._user_embeddings, self._asset_embeddings = self._model(
                normalized_edge_index,
                normalized_edge_weight,
                risk_band_tensor,
                type_tensor,
                capacity_tensor,
            )

    def _build_user_profile_index_arrays(
        self, all_customer_ids: list[str]
    ) -> tuple[dict[int, int | None], list[int], list[int]]:
        """Build per-user index arrays for the profile embedding tables."""
        customer_band_by_index: dict[int, int | None] = {}
        customer_type_indices: list[int] = []
        capacity_indices: list[int] = []

        for index, customer_id in enumerate(all_customer_ids):
            profile = self._customer_profiles.get(customer_id)
            if profile is None:
                customer_band_by_index[index] = None
                customer_type_indices.append(PROFILE_VOCABULARY.unknown_index)
                capacity_indices.append(PROFILE_VOCABULARY.unknown_index)
                continue

            customer_band_by_index[index] = profile.risk_band
            customer_type_indices.append(
                PROFILE_VOCABULARY.customer_type_to_index.get(
                    profile.customer_type or PROFILE_VOCABULARY.unknown_token,
                    PROFILE_VOCABULARY.unknown_index,
                )
            )
            capacity_indices.append(
                PROFILE_VOCABULARY.investment_capacity_to_index.get(
                    profile.investment_capacity or PROFILE_VOCABULARY.unknown_token,
                    PROFILE_VOCABULARY.unknown_index,
                )
            )

        return customer_band_by_index, customer_type_indices, capacity_indices

    @staticmethod
    def _materialize_risk_band_indices(
        customer_band_by_index: dict[int, int | None], number_of_users: int
    ) -> list[int]:
        """Return per-user risk band indices using the padding index for unknowns."""
        indices: list[int] = []
        for user_index in range(number_of_users):
            band = customer_band_by_index.get(user_index)
            indices.append(
                band if band is not None else PROFILE_VOCABULARY.risk_band_padding_index
            )
        return indices

    def _build_edge_index(
        self,
        split: TemporalSplitData,
        number_of_users: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Build the bidirectional user-asset edge index tensor."""
        edge_source: list[int] = []
        edge_target: list[int] = []
        for customer_id, asset_ids in split.training_interactions.items():
            user_node = self._customer_id_to_index[customer_id]
            for asset_id in asset_ids:
                asset_node = number_of_users + self._asset_id_to_index[asset_id]
                edge_source.extend([user_node, asset_node])
                edge_target.extend([asset_node, user_node])
        return torch.tensor([edge_source, edge_target], dtype=torch.long, device=device)

    def recommend_for_user(
        self,
        user_id: str,
        excluded_assets: set[str],
        k: int = 10,
    ) -> list[str]:
        """Return top-k eligible assets by embedding similarity, excluding held assets."""
        if self._user_embeddings is None or self._asset_embeddings is None:
            return []

        if user_id not in self._customer_id_to_index:
            return []

        user_index = self._customer_id_to_index[user_id]
        user_vector = self._user_embeddings[user_index]

        scores = torch.matmul(self._asset_embeddings, user_vector)

        for asset_index in range(len(scores)):
            if asset_index not in self._eligible_asset_indices:
                scores[asset_index] = float("-inf")

        for asset_id in excluded_assets:
            if asset_id not in self._asset_id_to_index:
                continue
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
