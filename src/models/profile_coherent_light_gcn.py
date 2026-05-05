"""Stratified profile-coherent LightGCN: one sub-model per MiFID risk band."""

from __future__ import annotations

import random

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import LGConv
from torch_geometric.nn.conv.gcn_conv import gcn_norm

from src.config.schemas import CustomerProfile, TemporalSplitData
from src.config.settings import ProfileCoherentLightGCNConfig
from src.utils.profile_coherence import (
    AGGRESSIVE,
    BALANCED,
    CONSERVATIVE,
    INCOME,
)

COHERENCE_TOLERANCE = 1
RISK_BANDS: tuple[int, ...] = (CONSERVATIVE, INCOME, BALANCED, AGGRESSIVE)
FALLBACK_BAND = BALANCED


class _LightGCNBackbone(nn.Module):
    """LightGCN propagation network used by every band sub-model."""

    def __init__(
        self,
        number_of_users: int,
        number_of_assets: int,
        embedding_dimension: int,
        number_of_layers: int,
    ) -> None:
        """Initialise user and asset embeddings and the LGConv stack."""
        super().__init__()
        self.number_of_users = number_of_users
        self.user_embedding = nn.Embedding(number_of_users, embedding_dimension)
        self.asset_embedding = nn.Embedding(number_of_assets, embedding_dimension)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.asset_embedding.weight)
        self.convolution_layers = nn.ModuleList(
            [LGConv(normalize=False) for _ in range(number_of_layers)]
        )

    def forward(
        self, edge_index: torch.Tensor, edge_weight: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return final user and asset embeddings after LightGCN propagation."""
        all_embeddings = torch.cat(
            [self.user_embedding.weight, self.asset_embedding.weight], dim=0
        )
        layer_outputs = [all_embeddings]
        current = all_embeddings
        for layer in self.convolution_layers:
            current = layer(current, edge_index, edge_weight)
            layer_outputs.append(current)
        final = torch.stack(layer_outputs, dim=0).mean(dim=0)
        return final[: self.number_of_users], final[self.number_of_users :]


class _BPRTripleDataset(torch.utils.data.Dataset):
    """User, positive asset, negative asset triples for one band's subgraph."""

    def __init__(
        self,
        filtered_interactions: dict[str, set[str]],
        customer_id_to_index: dict[str, int],
        asset_id_to_index: dict[str, int],
        number_of_assets: int,
    ) -> None:
        """Build BPR triples from the band-filtered training interactions."""
        all_indices = set(range(number_of_assets))
        self._triples: list[tuple[int, int, int]] = []
        for customer_id, asset_ids in filtered_interactions.items():
            if customer_id not in customer_id_to_index:
                continue
            user_index = customer_id_to_index[customer_id]
            positives = {
                asset_id_to_index[asset_id]
                for asset_id in asset_ids
                if asset_id in asset_id_to_index
            }
            if not positives:
                continue
            negatives = list(all_indices - positives)
            for positive_index in positives:
                if negatives:
                    negative_index = random.choice(negatives)
                else:
                    negative_index = random.randrange(number_of_assets)
                self._triples.append((user_index, positive_index, negative_index))

    def __len__(self) -> int:
        """Return the number of training triples."""
        return len(self._triples)

    def __getitem__(self, index: int) -> tuple[int, int, int]:
        """Return the triple at the given index."""
        return self._triples[index]


def _drop_edges(
    edge_index: torch.Tensor, edge_weight: torch.Tensor, keep_probability: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Edge dropout with inverse-probability rescaling for graph regularisation."""
    if keep_probability >= 1.0:
        return edge_index, edge_weight
    mask = torch.rand(edge_index.size(1), device=edge_index.device) < keep_probability
    if not mask.any():
        return edge_index, edge_weight
    return edge_index[:, mask], edge_weight[mask] / keep_probability


class _BandSubModel:
    """LightGCN sub-model trained on customers of one MiFID risk band."""

    def __init__(
        self,
        config: ProfileCoherentLightGCNConfig,
        target_band: int,
        asset_risk_classes: dict[str, int],
    ) -> None:
        """Store config, target band, and asset risk classes for PC-loss sampling."""
        self._config = config
        self._target_band = target_band
        self._asset_risk_classes = asset_risk_classes
        self._backbone: _LightGCNBackbone | None = None
        self._user_embeddings: torch.Tensor | None = None
        self._asset_embeddings: torch.Tensor | None = None
        self._customer_id_to_index: dict[str, int] = {}
        self._asset_id_to_index: dict[str, int] = {}
        self._index_to_asset_id: dict[int, str] = {}
        self._eligible_asset_indices: set[int] = set()

    def fit(
        self,
        filtered_interactions: dict[str, set[str]],
        eligible_asset_ids: list[str],
        device: torch.device,
    ) -> None:
        """Train one LightGCN on the band-filtered subgraph; add PC-loss when active."""
        all_customer_ids = sorted(filtered_interactions.keys())
        all_asset_ids = sorted(
            {
                asset_id
                for asset_ids in filtered_interactions.values()
                for asset_id in asset_ids
            }
        )
        if not all_customer_ids or not all_asset_ids:
            return

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
            for asset_id in eligible_asset_ids
            if asset_id in self._asset_id_to_index
        }

        coherent_indices, discordant_indices = self._partition_assets_by_coherence()

        number_of_users = len(self._customer_id_to_index)
        number_of_assets = len(self._asset_id_to_index)

        dataset = _BPRTripleDataset(
            filtered_interactions,
            self._customer_id_to_index,
            self._asset_id_to_index,
            number_of_assets,
        )
        if len(dataset) == 0:
            return

        edge_source: list[int] = []
        edge_target: list[int] = []
        for customer_id, asset_ids in filtered_interactions.items():
            user_node = self._customer_id_to_index[customer_id]
            for asset_id in asset_ids:
                if asset_id not in self._asset_id_to_index:
                    continue
                asset_node = number_of_users + self._asset_id_to_index[asset_id]
                edge_source.extend([user_node, asset_node])
                edge_target.extend([asset_node, user_node])

        edge_index = torch.tensor(
            [edge_source, edge_target], dtype=torch.long, device=device
        )
        normalized_edge_index, normalized_edge_weight = gcn_norm(
            edge_index,
            edge_weight=None,
            num_nodes=number_of_users + number_of_assets,
            add_self_loops=False,
            dtype=torch.float32,
        )

        self._backbone = _LightGCNBackbone(
            number_of_users=number_of_users,
            number_of_assets=number_of_assets,
            embedding_dimension=self._config.embedding_dimension,
            number_of_layers=self._config.number_of_layers,
        ).to(device)

        optimizer = torch.optim.Adam(
            self._backbone.parameters(), lr=self._config.learning_rate
        )
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=self._config.batch_size, shuffle=True
        )

        coherent_tensor = (
            torch.tensor(coherent_indices, device=device, dtype=torch.long)
            if coherent_indices
            else None
        )
        discordant_tensor = (
            torch.tensor(discordant_indices, device=device, dtype=torch.long)
            if discordant_indices
            else None
        )
        coherence_active = (
            self._config.coherence_loss_weight > 0.0
            and coherent_tensor is not None
            and discordant_tensor is not None
        )

        self._backbone.train()
        for _ in range(self._config.number_of_epochs):
            for batch in data_loader:
                user_indices = batch[0].to(device)
                positive_indices = batch[1].to(device)
                negative_indices = batch[2].to(device)

                dropped_edge_index, dropped_edge_weight = _drop_edges(
                    normalized_edge_index,
                    normalized_edge_weight,
                    self._config.keep_probability,
                )
                user_embeddings, asset_embeddings = self._backbone(
                    dropped_edge_index, dropped_edge_weight
                )

                loss = self._compute_loss(
                    user_indices,
                    positive_indices,
                    negative_indices,
                    user_embeddings,
                    asset_embeddings,
                    coherent_tensor if coherence_active else None,
                    discordant_tensor if coherence_active else None,
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        self._backbone.eval()
        with torch.no_grad():
            self._user_embeddings, self._asset_embeddings = self._backbone(
                normalized_edge_index, normalized_edge_weight
            )

    def recommend(
        self, customer_id: str, excluded_assets: set[str], k: int
    ) -> list[str]:
        """Return top-k assets within this sub-model's universe, excluding held assets."""
        if self._user_embeddings is None or self._asset_embeddings is None:
            return []
        if customer_id not in self._customer_id_to_index:
            return []

        user_index = self._customer_id_to_index[customer_id]
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

    def _partition_assets_by_coherence(self) -> tuple[list[int], list[int]]:
        """Split this sub-model's asset universe into coherent and discordant indices."""
        coherent: list[int] = []
        discordant: list[int] = []
        for asset_id, asset_index in self._asset_id_to_index.items():
            asset_band = self._asset_risk_classes.get(asset_id)
            if asset_band is None:
                continue
            if abs(asset_band - self._target_band) <= COHERENCE_TOLERANCE:
                coherent.append(asset_index)
            else:
                discordant.append(asset_index)
        return coherent, discordant

    def _compute_loss(
        self,
        user_indices: torch.Tensor,
        positive_indices: torch.Tensor,
        negative_indices: torch.Tensor,
        user_embeddings: torch.Tensor,
        asset_embeddings: torch.Tensor,
        coherent_tensor: torch.Tensor | None,
        discordant_tensor: torch.Tensor | None,
    ) -> torch.Tensor:
        """BPR + L2 regularisation, plus the optional profile-coherent margin term."""
        assert self._backbone is not None
        user_vectors = user_embeddings[user_indices]
        positive_vectors = asset_embeddings[positive_indices]
        negative_vectors = asset_embeddings[negative_indices]
        positive_scores = (user_vectors * positive_vectors).sum(dim=1)
        negative_scores = (user_vectors * negative_vectors).sum(dim=1)
        bpr_loss = F.softplus(negative_scores - positive_scores).mean()

        l2_term = (
            self._backbone.user_embedding.weight[user_indices].norm(2).pow(2)
            + self._backbone.asset_embedding.weight[positive_indices].norm(2).pow(2)
            + self._backbone.asset_embedding.weight[negative_indices].norm(2).pow(2)
        ) / (2 * user_indices.shape[0])
        loss = bpr_loss + self._config.weight_decay * l2_term

        if coherent_tensor is None or discordant_tensor is None:
            return loss

        device = user_indices.device
        batch_size = user_indices.shape[0]
        coherent_samples = coherent_tensor[
            torch.randint(0, len(coherent_tensor), (batch_size,), device=device)
        ]
        discordant_samples = discordant_tensor[
            torch.randint(0, len(discordant_tensor), (batch_size,), device=device)
        ]
        coherent_scores = (user_vectors * asset_embeddings[coherent_samples]).sum(dim=1)
        discordant_scores = (user_vectors * asset_embeddings[discordant_samples]).sum(
            dim=1
        )
        coherence_loss = F.softplus(discordant_scores - coherent_scores).mean()
        return loss + self._config.coherence_loss_weight * coherence_loss


class ProfileCoherentLightGCN:
    """Stratified LightGCN: one sub-model per MiFID risk band, with optional PC-loss."""

    def __init__(
        self,
        config: ProfileCoherentLightGCNConfig,
        customer_profiles: dict[str, CustomerProfile],
        asset_risk_classes: dict[str, int],
    ) -> None:
        """Store config and the profile lookups used for stratification and routing."""
        self._config = config
        self._customer_profiles = customer_profiles
        self._asset_risk_classes = asset_risk_classes
        self._sub_models: dict[int, _BandSubModel] = {}

    @property
    def name(self) -> str:
        """Display name reflecting whether the PC-loss term is active."""
        if self._config.coherence_loss_weight > 0.0:
            return f"PC-LGCN (lambda={self._config.coherence_loss_weight})"
        return "Stratified LightGCN"

    def train_on_split(self, split: TemporalSplitData, **kwargs: object) -> None:
        """Train one sub-model per MiFID risk band on the band-filtered subgraph."""
        device_name = kwargs.get("device", "cpu")
        device = torch.device(str(device_name))
        for band in RISK_BANDS:
            band_interactions = self._filter_interactions_to_band(
                split.training_interactions, band
            )
            sub_model = _BandSubModel(
                config=self._config,
                target_band=band,
                asset_risk_classes=self._asset_risk_classes,
            )
            sub_model.fit(
                filtered_interactions=band_interactions,
                eligible_asset_ids=split.eligible_asset_ids,
                device=device,
            )
            self._sub_models[band] = sub_model

    def recommend_for_user(
        self, user_id: str, excluded_assets: set[str], k: int = 10
    ) -> list[str]:
        """Route the customer to their band's sub-model; fall back to Balanced when missing."""
        target_band = self._route_band(user_id)
        sub_model = self._sub_models.get(target_band)
        if sub_model is None:
            return []
        return sub_model.recommend(user_id, excluded_assets, k)

    def _filter_interactions_to_band(
        self, training_interactions: dict[str, set[str]], band: int
    ) -> dict[str, set[str]]:
        """Restrict training interactions to customers whose risk band matches."""
        return {
            customer_id: assets
            for customer_id, assets in training_interactions.items()
            if (profile := self._customer_profiles.get(customer_id)) is not None
            and profile.risk_band == band
        }

    def _route_band(self, user_id: str) -> int:
        """Return the customer's band, falling back to Balanced when missing."""
        profile = self._customer_profiles.get(user_id)
        if profile is None or profile.risk_band is None:
            return FALLBACK_BAND
        return profile.risk_band
