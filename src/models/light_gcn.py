"""LightGCN collaborative filtering model with BPR loss.

Faithful to the original paper (He et al., SIGIR 2020): symmetric
`D^{-1/2} A D^{-1/2}` adjacency normalization with no self-loops, and the
final embedding is the unweighted mean of `E^{(0)}, E^{(1)}, ..., E^{(K)}`.
This matches `torch_geometric.nn.LGConv`'s canonical implementation.

The FAR-Trans paper (https://github.com/JavierSanzCruza/far-trans) runs
LightGCN through the Beta-RecSys library
(https://github.com/beta-team/community/blob/master/beta_recsys/README.md),
whose `beta_rec/models/lightgcn.py` + `beta_rec/data/base_data.py:337-360`
applies NGCF-style asymmetric `D^{-1}(A + I)` normalization with added
self-loops. That is a bug versus the original LightGCN paper (Eq. 3); we
deliberately do not replicate it.

We also deliberately do NOT track the best-validation-epoch checkpoint the
way Beta-RecSys does. Upstream `LightGCN_Train.test()` reloads weights from
the epoch with the highest nDCG on a "validation" set that FAR-Trans sets
equal to the test set (see `data/splitted_data.py:41` in the FAR-Trans
repo), which is leaky model selection. Our `train_on_split` trains the
full `number_of_epochs` and uses the last-epoch weights for inference.
"""

import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import LGConv
from torch_geometric.nn.conv.gcn_conv import gcn_norm

from src.config.schemas import TemporalSplitData
from src.config.settings import LightGCNConfig


class LightGCNModel(nn.Module):
    """LightGCN graph convolution model with BPR loss."""

    def __init__(
        self,
        number_of_users: int,
        number_of_assets: int,
        embedding_dimension: int,
        number_of_layers: int,
    ) -> None:
        super().__init__()
        self.number_of_users = number_of_users
        self.number_of_assets = number_of_assets
        self.number_of_layers = number_of_layers

        self.user_embedding = nn.Embedding(number_of_users, embedding_dimension)
        self.asset_embedding = nn.Embedding(number_of_assets, embedding_dimension)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.asset_embedding.weight)

        self.convolution_layers = nn.ModuleList(
            [LGConv(normalize=False) for _ in range(number_of_layers)]
        )

    def forward(
        self,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return final user and asset embeddings after LightGCN propagation.

        `edge_index` and `edge_weight` must already be symmetrically normalized
        (e.g. via `torch_geometric.nn.conv.gcn_conv.gcn_norm(..., add_self_loops=False)`).
        The convolution layers are instantiated with `normalize=False` so they
        apply the given weights directly, which lets the training loop drop
        edges from the pre-normalized graph and rescale the survivors by
        `1/keep_probability` without the normalization being recomputed on the
        reduced graph.
        """
        all_embeddings = torch.cat(
            [self.user_embedding.weight, self.asset_embedding.weight], dim=0
        )

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

    def bpr_loss(
        self,
        user_indices: torch.Tensor,
        positive_asset_indices: torch.Tensor,
        negative_asset_indices: torch.Tensor,
        user_embeddings: torch.Tensor,
        asset_embeddings: torch.Tensor,
        regularization_weight: float = 0.0,
    ) -> torch.Tensor:
        """Compute BPR loss with L2 regularization on initial (0th-layer) embeddings."""
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

        return loss


class BPRDataset(torch.utils.data.Dataset):
    """Dataset yielding (user_index, positive_asset_index, negative_asset_index) triples."""

    def __init__(
        self,
        training_interactions: dict[str, set[str]],
        customer_id_to_index: dict[str, int],
        asset_id_to_index: dict[str, int],
        number_of_assets: int,
    ) -> None:
        self._all_asset_indices = set(range(number_of_assets))
        self._user_positive_assets: dict[int, set[int]] = {}
        self._triples: list[tuple[int, int, int]] = []

        for customer_id, asset_ids in training_interactions.items():
            if customer_id not in customer_id_to_index:
                continue
            user_index = customer_id_to_index[customer_id]
            positive_indices: set[int] = set()

            for asset_id in asset_ids:
                if asset_id not in asset_id_to_index:
                    continue
                asset_index = asset_id_to_index[asset_id]
                positive_indices.add(asset_index)

            self._user_positive_assets[user_index] = positive_indices
            negative_candidates = list(self._all_asset_indices - positive_indices)

            for positive_index in positive_indices:
                if negative_candidates:
                    negative_index = random.choice(negative_candidates)
                else:
                    negative_index = random.randrange(len(self._all_asset_indices))
                self._triples.append((user_index, positive_index, negative_index))

    def __len__(self) -> int:
        return len(self._triples)

    def __getitem__(self, index: int) -> tuple[int, int, int]:
        """Return (user_index, positive_asset_index, negative_asset_index)."""
        return self._triples[index]


def _drop_edges_with_inverted_scaling(
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    keep_probability: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Drop a fraction of edges and rescale the survivors by `1/keep_probability`.

    This is the inverted-dropout pattern standard in dropout-family
    regularizers: on expectation, the propagated signal magnitude is
    unchanged, so no post-hoc correction is needed at inference time.
    Operates on the already-normalized `(edge_index, edge_weight)` pair so
    that the symmetric `D^{-1/2} A D^{-1/2}` weights computed on the full
    graph are not re-derived on the dropped subgraph.
    """
    if keep_probability >= 1.0:
        return edge_index, edge_weight

    number_of_edges = edge_index.size(1)
    keep_mask = torch.rand(number_of_edges, device=edge_index.device) < keep_probability

    if not keep_mask.any():
        return edge_index, edge_weight

    kept_edge_index = edge_index[:, keep_mask]
    kept_edge_weight = edge_weight[keep_mask] / keep_probability
    return kept_edge_index, kept_edge_weight


class LightGCNBaseline:
    """LightGCN collaborative filtering baseline."""

    def __init__(self, config: LightGCNConfig) -> None:
        self._config = config
        self._model: LightGCNModel | None = None
        self._user_embeddings: torch.Tensor | None = None
        self._asset_embeddings: torch.Tensor | None = None
        self._customer_id_to_index: dict[str, int] = {}
        self._asset_id_to_index: dict[str, int] = {}
        self._index_to_asset_id: dict[int, str] = {}
        self._eligible_asset_indices: set[int] = set()

    @property
    def name(self) -> str:
        """Return the display name used in evaluation results."""
        return "LightGCN"

    def train_on_split(self, split: TemporalSplitData, **kwargs: object) -> None:
        """Train LightGCN on the full interaction graph, filtering at recommendation time."""
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

        dataset = BPRDataset(
            split.training_interactions,
            self._customer_id_to_index,
            self._asset_id_to_index,
            number_of_assets,
        )

        edge_source: list[int] = []
        edge_target: list[int] = []

        for customer_id, asset_ids in split.training_interactions.items():
            user_node = self._customer_id_to_index[customer_id]
            for asset_id in asset_ids:
                asset_node = number_of_users + self._asset_id_to_index[asset_id]
                edge_source.extend([user_node, asset_node])
                edge_target.extend([asset_node, user_node])

        edge_index = torch.tensor(
            [edge_source, edge_target], dtype=torch.long, device=device
        )

        number_of_nodes = number_of_users + number_of_assets
        normalized_edge_index, normalized_edge_weight = gcn_norm(
            edge_index,
            edge_weight=None,
            num_nodes=number_of_nodes,
            add_self_loops=False,
            dtype=torch.float32,
        )

        self._model = LightGCNModel(
            number_of_users=number_of_users,
            number_of_assets=number_of_assets,
            embedding_dimension=self._config.embedding_dimension,
            number_of_layers=self._config.number_of_layers,
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

        self._model.train()
        for _ in range(self._config.number_of_epochs):
            for batch in data_loader:
                user_indices = batch[0].to(device)
                positive_indices = batch[1].to(device)
                negative_indices = batch[2].to(device)

                dropped_edge_index, dropped_edge_weight = (
                    _drop_edges_with_inverted_scaling(
                        normalized_edge_index,
                        normalized_edge_weight,
                        self._config.keep_probability,
                    )
                )
                user_embeddings, asset_embeddings = self._model(
                    dropped_edge_index, dropped_edge_weight
                )

                loss = self._model.bpr_loss(
                    user_indices,
                    positive_indices,
                    negative_indices,
                    user_embeddings,
                    asset_embeddings,
                    regularization_weight=self._config.weight_decay,
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        self._model.eval()
        with torch.no_grad():
            self._user_embeddings, self._asset_embeddings = self._model(
                normalized_edge_index, normalized_edge_weight
            )

    def recommend_for_user(
        self,
        user_id: str,
        excluded_assets: set[str],
        k: int = 10,
    ) -> list[str]:
        """Return top-k eligible assets by embedding similarity, excluding already-acquired ones."""
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
