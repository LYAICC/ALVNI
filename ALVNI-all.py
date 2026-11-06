#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Reproducible ALVNI implementation for the public Hamster example.

Default protocol
----------------
The defaults reproduce the final ALVNI configuration used for the public
example: a six-feature CrossNet--GraphSAGE--MLP predictor, a total SIR-label
budget of 200, an initial query set of 40, and query batches of 40. The
training objective is 0.5 MSE plus 0.5 pairwise ranking loss. The acquisition
weights for uncertainty, diversity, representativeness, and local centrality
are (0.5, 1/6, 1/6, 1/6): uncertainty receives one half of the score, and the three structural criteria share the other half equally.

Run without arguments from this directory to use the bundled Hamster edge list
and precomputed SIR labels at beta/beta_th = 1.25. The script writes a
summary.json containing Kendall tau-b, Jaccard@100, NDCG@100, and
Imprecision@100, plus node-level predictions and the acquired node IDs.

For a new graph, pass --edges and --labels. Labels must be an offline mapping
from original node IDs to SIR influence scores; their generation is separate
from the reported ALVNI training time.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import kendalltau, spearmanr
from torch_geometric.nn import SAGEConv


EPS = 1e-12
FEATURE_NAMES = (
    "degree",
    "mean_neighbor_degree",
    "clustering_coefficient",
    "h_index",
    "unique_two_hop_reach",
    "neighbor_degree_std",
)


@dataclass
class ALVNIConfig:
    """Hyperparameters of the final Core6-Cross-SAGE ALVNI protocol."""

    label_budget: int = 200
    initial_labels: int = 40
    query_batch: int = 40
    initial_clusters: int = 20
    epochs_per_round: int = 160
    final_epochs: int = 240
    hidden_dim: int = 128
    sage_layers: int = 2
    cross_layers: int = 2
    projection_dim: int = 32
    dropout: float = 0.35
    learning_rate: float = 1e-3
    weight_decay: float = 5e-4
    mse_weight: float = 0.50
    rank_weight: float = 0.50
    mc_passes: int = 20
    uncertainty_weight: float = 0.50
    diversity_weight: float = 1.0 / 6.0
    representativeness_weight: float = 1.0 / 6.0
    centrality_weight: float = 1.0 / 6.0
    density_reference_limit: int = 1024
    seed: int = 4307
    device: str = "auto"


def set_seed(seed: int) -> None:
    """Set random seeds for reproducible sampling, training, and MC dropout."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(value: str) -> torch.device:
    """Resolve ``auto`` to CUDA when available, otherwise to CPU."""

    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


# ---------------------------------------------------------------------------
# Data and label preparation
# ---------------------------------------------------------------------------

def read_node_ids(path: Path) -> List[int]:
    """Read optional isolated-node IDs, one ID per nonempty line."""

    ids: List[int] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if text and not text.startswith("#"):
            ids.append(int(text.split()[0]))
    return ids


def load_undirected_graph(edge_path: Path, nodes_path: Path | None = None) -> nx.Graph:
    """Load a simple undirected graph, removing duplicated edges and self-loops."""

    graph = nx.Graph()
    with edge_path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            tokens = text.replace(",", " ").split()
            if len(tokens) < 2:
                raise ValueError(f"Invalid edge at {edge_path}:{line_no}: {text!r}")
            u, v = int(tokens[0]), int(tokens[1])
            if u != v:
                graph.add_edge(u, v)
    if nodes_path is not None:
        graph.add_nodes_from(read_node_ids(nodes_path))
    if graph.number_of_nodes() == 0:
        raise ValueError("The graph has no nodes after loading and preprocessing.")
    return graph


def load_sir_scores(path: Path) -> Dict[int, float]:
    """Load SIR scores from a pickle dictionary/array, NPY file, or CSV file.

    A pickle dictionary should map original node IDs to continuous SIR scores.
    A one-dimensional array is interpreted as scores for integer node IDs
    ``0,\ldots,n-1``.  A CSV file must contain either ``node_id,sir_score`` or
    its first two columns are used as the node ID and score, respectively.
    """

    suffix = path.suffix.lower()
    if suffix in {".pkl", ".pickle"}:
        with path.open("rb") as handle:
            raw = pickle.load(handle)
    elif suffix == ".npy":
        raw = np.load(path, allow_pickle=True)
    elif suffix == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise ValueError(f"No rows found in {path}")
        fields = rows[0].keys()
        id_key = "node_id" if "node_id" in fields else next(iter(fields))
        score_key = "sir_score" if "sir_score" in fields else list(fields)[1]
        return {int(row[id_key]): float(row[score_key]) for row in rows}
    else:
        raise ValueError("Unsupported label format. Use .pkl, .pickle, .npy, or .csv.")

    if isinstance(raw, Mapping):
        return {int(node): float(score) for node, score in raw.items()}
    array = np.asarray(raw, dtype=np.float64).reshape(-1)
    return {int(i): float(value) for i, value in enumerate(array)}


def align_graph_and_labels(graph: nx.Graph, scores: Mapping[int, float]) -> Tuple[nx.Graph, List[int], np.ndarray]:
    """Keep labelled graph nodes and remap them to contiguous model indices."""

    original_nodes = sorted(node for node in graph.nodes if node in scores)
    if not original_nodes:
        raise ValueError("No graph nodes have corresponding SIR scores.")
    missing = graph.number_of_nodes() - len(original_nodes)
    if missing:
        print(f"[Data] Excluding {missing} graph nodes without SIR scores.")
    subgraph = graph.subgraph(original_nodes).copy()
    mapping = {node: index for index, node in enumerate(original_nodes)}
    reindexed = nx.relabel_nodes(subgraph, mapping, copy=True)
    labels = np.asarray([scores[node] for node in original_nodes], dtype=np.float32)
    return reindexed, original_nodes, labels


def epidemic_threshold(graph: nx.Graph) -> float:
    """Compute \(\langle k\rangle/(\langle k^2\rangle-\langle k\rangle)\)."""

    degree = np.asarray([value for _, value in graph.degree()], dtype=np.float64)
    mean_k, mean_k2 = float(degree.mean()), float(np.mean(degree**2))
    denominator = mean_k2 - mean_k
    if denominator <= 0:
        raise ValueError("Cannot compute an epidemic threshold for this graph.")
    return mean_k / denominator


def simulate_sir_from_source(graph: nx.Graph, source: int, infection_rate: float,
                             recovery_rate: float, rng: np.random.Generator,
                             max_steps: int) -> int:
    """Run one discrete-time SIR realization and return its final outbreak size."""

    infected = {source}
    recovered: set[int] = set()
    ever_infected = {source}
    for _ in range(max_steps):
        if not infected:
            break
        next_infected: set[int] = set()
        for node in infected:
            for neighbor in graph.neighbors(node):
                if neighbor not in infected and neighbor not in recovered and rng.random() < infection_rate:
                    next_infected.add(neighbor)
        for node in infected:
            if rng.random() < recovery_rate:
                recovered.add(node)
            else:
                next_infected.add(node)
        infected = next_infected - recovered
        ever_infected.update(infected)
    return len(ever_infected)


def generate_sir_scores(graph: nx.Graph, infection_rate: float, recovery_rate: float,
                        repeats: int, seed: int, max_steps: int = 10000) -> Dict[int, float]:
    """Generate offline mean final-outbreak SIR scores for all nodes.

    This intentionally runs before ALVNI.  Its cost is an external label
    generation cost and is not included in predictor training time.
    """

    if not 0.0 <= infection_rate <= 1.0 or not 0.0 < recovery_rate <= 1.0:
        raise ValueError("SIR probabilities must satisfy beta in [0,1] and mu in (0,1].")
    if repeats <= 0:
        raise ValueError("sir_repeats must be positive.")
    labels: Dict[int, float] = {}
    for position, node in enumerate(graph.nodes):
        node_rng = np.random.default_rng(seed + 1_000_003 * (position + 1))
        total = sum(
            simulate_sir_from_source(graph, node, infection_rate, recovery_rate, node_rng, max_steps)
            for _ in range(repeats)
        )
        labels[int(node)] = total / float(repeats)
        if (position + 1) % max(1, graph.number_of_nodes() // 20) == 0:
            print(f"[SIR] completed {position + 1}/{graph.number_of_nodes()} seed nodes")
    return labels


# ---------------------------------------------------------------------------
# Six local structural descriptors
# ---------------------------------------------------------------------------

def local_h_index(graph: nx.Graph, degree: Mapping[int, int]) -> np.ndarray:
    """Compute each node's local H-index from the degrees of its neighbours."""

    values = np.zeros(graph.number_of_nodes(), dtype=np.float32)
    for node in range(graph.number_of_nodes()):
        neighbour_degrees = sorted((degree[neighbor] for neighbor in graph.neighbors(node)), reverse=True)
        h_value = 0
        for rank, value in enumerate(neighbour_degrees, start=1):
            if value >= rank:
                h_value = rank
            else:
                break
        values[node] = h_value
    return values


def six_topological_features(graph: nx.Graph) -> np.ndarray:
    """Construct the six Core6 descriptors directly, without unused features.

    Count-related quantities receive ``log(1+x)`` compression.  The clustering
    coefficient remains on its native scale.  Standardisation is deliberately
    deferred to ``run_active_learning`` so it uses only training-graph nodes.
    """

    n = graph.number_of_nodes()
    degree = dict(graph.degree())
    clustering = nx.clustering(graph)
    h_index = local_h_index(graph, degree)
    matrix = np.zeros((n, 6), dtype=np.float32)
    for node in range(n):
        neighbours = list(graph.neighbors(node))
        neighbour_degrees = np.asarray([degree[v] for v in neighbours], dtype=np.float32)
        mean_degree = float(neighbour_degrees.mean()) if neighbour_degrees.size else 0.0
        std_degree = float(neighbour_degrees.std()) if neighbour_degrees.size else 0.0
        two_hop: set[int] = set()
        for neighbour in neighbours:
            two_hop.update(graph.neighbors(neighbour))
        two_hop.discard(node)
        two_hop.difference_update(neighbours)
        matrix[node] = (
            np.log1p(degree[node]),
            np.log1p(mean_degree),
            float(clustering[node]),
            np.log1p(h_index[node]),
            np.log1p(len(two_hop)),
            np.log1p(std_degree),
        )
    return matrix


def edge_index_from_graph(graph: nx.Graph, device: torch.device) -> torch.Tensor:
    """Create a bidirectional PyG edge index from the undirected graph."""

    edges = np.asarray(list(graph.edges()), dtype=np.int64)
    if edges.size == 0:
        return torch.empty((2, 0), dtype=torch.long, device=device)
    arcs = np.concatenate((edges, edges[:, ::-1]), axis=0)
    return torch.as_tensor(arcs.T.copy(), dtype=torch.long, device=device)


# ---------------------------------------------------------------------------
# Core6-Cross-SAGE predictor
# ---------------------------------------------------------------------------

class CrossLayer(nn.Module):
    """One DCN layer: \(x_{l+1}=x_0(w_l^\top x_l)+b_l+x_l\)."""

    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(dimension))
        self.bias = nn.Parameter(torch.zeros(dimension))
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, x0: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return x0 * torch.matmul(x, self.weight).unsqueeze(-1) + self.bias + x


class Core6CrossSAGE(nn.Module):
    """Six-feature CrossNet lifting, mean GraphSAGE encoding, and MLP scoring."""

    def __init__(self, config: ALVNIConfig) -> None:
        super().__init__()
        self.dropout = config.dropout
        self.cross_layers = nn.ModuleList([CrossLayer(6) for _ in range(config.cross_layers)])
        self.project = nn.Sequential(nn.Linear(6, config.projection_dim), nn.ReLU())
        self.convolutions = nn.ModuleList()
        self.normalisations = nn.ModuleList()
        for layer in range(config.sage_layers):
            input_dim = config.projection_dim if layer == 0 else config.hidden_dim
            self.convolutions.append(SAGEConv(input_dim, config.hidden_dim, aggr="mean"))
            if layer < config.sage_layers - 1:
                self.normalisations.append(nn.BatchNorm1d(config.hidden_dim))
        self.regression_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = x
        for layer in self.cross_layers:
            z = layer(x, z)
        z = self.project(z)
        for index, convolution in enumerate(self.convolutions):
            z = convolution(z, edge_index)
            if index < len(self.normalisations):
                z = F.dropout(F.relu(self.normalisations[index](z)), p=self.dropout, training=self.training)
        return self.regression_head(z).squeeze(-1), z


# ---------------------------------------------------------------------------
# Ranking-aware optimisation and prediction
# ---------------------------------------------------------------------------

def pairwise_ranking_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Pairwise logistic ranking loss on queried nodes with unequal labels."""

    if prediction.numel() < 2:
        return prediction.new_zeros(())
    difference = target[:, None] - target[None, :]
    upper = torch.triu(difference != 0, diagonal=1)
    if not upper.any():
        return prediction.new_zeros(())
    predicted_difference = (prediction[:, None] - prediction[None, :])[upper]
    signed_difference = torch.sign(difference[upper]) * predicted_difference
    return F.softplus(-signed_difference).mean()


def queried_minmax_targets(labels: torch.Tensor, queried: torch.Tensor) -> torch.Tensor:
    """Normalise only labels currently available to the active learner."""

    subset = labels[queried]
    return (subset - subset.min()) / (subset.max() - subset.min() + 1e-8)


def train_predictor(model: nn.Module, features: torch.Tensor, edge_index: torch.Tensor,
                    labels: torch.Tensor, queried: np.ndarray, config: ALVNIConfig,
                    epochs: int) -> Dict[str, float]:
    """Continue optimisation from the current parameters using queried labels."""

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    index = torch.as_tensor(queried, dtype=torch.long, device=features.device)
    target = queried_minmax_targets(labels, index)
    last_mse = last_rank = last_loss = 0.0
    for _ in range(epochs):
        model.train()
        score, _ = model(features, edge_index)
        mse = F.mse_loss(score[index], target)
        ranking = pairwise_ranking_loss(score[index], target)
        loss = config.mse_weight * mse + config.rank_weight * ranking
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        last_mse, last_rank, last_loss = float(mse.item()), float(ranking.item()), float(loss.item())
    return {"mse": last_mse, "rank_loss": last_rank, "loss": last_loss}


@torch.no_grad()
def predict(model: nn.Module, features: torch.Tensor, edge_index: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    """Return deterministic node scores and GraphSAGE embeddings."""

    model.eval()
    score, embedding = model(features, edge_index)
    return score.cpu().numpy(), embedding.cpu().numpy()


@torch.no_grad()
def mc_dropout_uncertainty(model: nn.Module, features: torch.Tensor, edge_index: torch.Tensor,
                           passes: int) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate score uncertainty by retaining dropout during stochastic passes."""

    model.train()
    # BatchNorm statistics must not change during uncertainty estimation.
    for module in model.modules():
        if isinstance(module, nn.BatchNorm1d):
            module.eval()
    samples = np.stack([model(features, edge_index)[0].cpu().numpy() for _ in range(passes)])
    return samples.mean(axis=0), samples.var(axis=0)


# ---------------------------------------------------------------------------
# Budgeted active acquisition
# ---------------------------------------------------------------------------

def minmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return (values - values.min()) / (values.max() - values.min() + EPS)


def feature_kmeans(values: np.ndarray, clusters: int, rng: np.random.Generator,
                   iterations: int = 30) -> np.ndarray:
    """Small dependency-free K-means used only for label-free initial coverage."""

    n = len(values)
    if n == 0:
        return np.empty(0, dtype=np.int64)
    clusters = max(1, min(int(clusters), n))
    centres = np.empty((clusters, values.shape[1]), dtype=np.float32)
    centres[0] = values[int(rng.integers(n))]
    distance = np.linalg.norm(values - centres[0], axis=1)
    for index in range(1, clusters):
        choice = int(np.argmax(distance))
        centres[index] = values[choice]
        distance = np.minimum(distance, np.linalg.norm(values - centres[index], axis=1))
    labels = np.zeros(n, dtype=np.int64)
    for _ in range(iterations):
        labels = np.argmin(np.linalg.norm(values[:, None] - centres[None, :], axis=2), axis=1)
        for index in range(clusters):
            members = np.flatnonzero(labels == index)
            if members.size:
                centres[index] = values[members].mean(axis=0)
    return labels


def initial_query_set(features: np.ndarray, count: int, clusters: int,
                      rng: np.random.Generator) -> np.ndarray:
    """Select cluster-representative, locally central nodes before training."""

    n = len(features)
    count = min(max(0, int(count)), n)
    if count == 0:
        return np.empty(0, dtype=np.int64)
    standardised = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)
    labels = feature_kmeans(standardised, min(clusters, count), rng)
    degree_proxy = features[:, 0]
    selected: List[int] = []
    for cluster in range(int(labels.max()) + 1):
        members = np.flatnonzero(labels == cluster)
        if members.size:
            selected.append(int(members[np.argmax(degree_proxy[members])]))
    # Fill any remaining positions by maximum distance from the selected set.
    while len(selected) < count:
        anchors = standardised[np.asarray(selected, dtype=np.int64)]
        distance = np.linalg.norm(standardised[:, None] - anchors[None, :], axis=2).min(axis=1)
        distance[np.asarray(selected, dtype=np.int64)] = -np.inf
        selected.append(int(np.argmax(distance)))
    return np.asarray(selected[:count], dtype=np.int64)


def embedding_density(embedding: np.ndarray, candidates: np.ndarray,
                      reference_limit: int) -> np.ndarray:
    """Compute inverse mean distance to a bounded candidate reference pool.

    The reference pool is the full candidate set for small graphs and a
    deterministic evenly spaced subset otherwise.  This exactly defines the
    representativeness implementation used by this standalone version.
    """

    if candidates.size == 0:
        return np.empty(0, dtype=np.float32)
    pool = candidates
    if pool.size > reference_limit:
        positions = np.linspace(0, pool.size - 1, num=reference_limit, dtype=np.int64)
        pool = pool[positions]
    reference = embedding[pool].astype(np.float32, copy=False)
    result = np.empty(candidates.size, dtype=np.float32)
    for start in range(0, candidates.size, 512):
        stop = min(start + 512, candidates.size)
        current = embedding[candidates[start:stop]].astype(np.float32, copy=False)
        distance = np.linalg.norm(current[:, None, :] - reference[None, :, :], axis=2)
        result[start:stop] = 1.0 / (distance.mean(axis=1) + EPS)
    return result


def acquire_batch(model: nn.Module, features_tensor: torch.Tensor, edge_index: torch.Tensor,
                  raw_features: np.ndarray, queried: np.ndarray, candidates: np.ndarray,
                  batch_size: int, config: ALVNIConfig) -> np.ndarray:
    """Greedily combine U, D, R, and C into the next SIR query batch."""

    _, variance = mc_dropout_uncertainty(model, features_tensor, edge_index, config.mc_passes)
    _, embedding = predict(model, features_tensor, edge_index)
    density = embedding_density(embedding, candidates, config.density_reference_limit)
    density_by_node = {int(node): float(value) for node, value in zip(candidates, density)}
    centrality = raw_features[:, 0]  # log(1 + degree), a label-free local prior
    anchors = embedding[queried]
    remaining = candidates.copy()
    selected: List[int] = []
    for _ in range(min(batch_size, remaining.size)):
        diversity = np.linalg.norm(embedding[remaining, None] - anchors[None, :], axis=2).min(axis=1)
        representativeness = np.asarray([density_by_node[int(node)] for node in remaining], dtype=np.float32)
        utility = (
            config.uncertainty_weight * minmax(variance[remaining])
            + config.diversity_weight * minmax(diversity)
            + config.representativeness_weight * minmax(representativeness)
            + config.centrality_weight * minmax(centrality[remaining])
        )
        chosen = int(remaining[np.argmax(utility)])
        selected.append(chosen)
        anchors = np.vstack((anchors, embedding[chosen]))
        remaining = remaining[remaining != chosen]
    return np.asarray(selected, dtype=np.int64)


# ---------------------------------------------------------------------------
# Evaluation and end-to-end experiment
# ---------------------------------------------------------------------------

def imprecision_at_k(labels: np.ndarray, prediction: np.ndarray, k: int) -> float:
    k = min(int(k), len(labels))
    truth_top = np.argsort(labels)[-k:]
    predicted_top = np.argsort(prediction)[-k:]
    return max(0.0, 1.0 - float(labels[predicted_top].mean()) / (float(labels[truth_top].mean()) + EPS))


def evaluation_metrics(labels: np.ndarray, prediction: np.ndarray,
                       k_values: Iterable[int] = (10, 50, 100)) -> Dict[str, float]:
    """Compute tie-corrected Kendall tau-b, Top-K overlap, NDCG, and imprecision."""

    tau = kendalltau(labels, prediction).statistic
    rho = spearmanr(labels, prediction).statistic
    metrics: Dict[str, float] = {
        "kendall_tau_b": 0.0 if np.isnan(tau) else float(tau),
        "spearman_rho": 0.0 if np.isnan(rho) else float(rho),
    }
    for k in k_values:
        k = min(int(k), len(labels))
        truth_top = np.argsort(labels)[-k:]
        predicted_top = np.argsort(prediction)[-k:]
        metrics[f"jaccard_at_{k}"] = len(set(truth_top) & set(predicted_top)) / max(1, len(set(truth_top) | set(predicted_top)))
        ranked = predicted_top[np.argsort(prediction[predicted_top])[::-1]]
        ideal = truth_top[np.argsort(labels[truth_top])[::-1]]
        dcg = sum(float(labels[node]) / math.log2(rank + 2) for rank, node in enumerate(ranked))
        idcg = sum(float(labels[node]) / math.log2(rank + 2) for rank, node in enumerate(ideal))
        metrics[f"ndcg_at_{k}"] = dcg / idcg if idcg else 0.0
        metrics[f"imprecision_at_{k}"] = imprecision_at_k(labels, prediction, k)
    return metrics


def run_active_learning(graph: nx.Graph, labels: np.ndarray, config: ALVNIConfig) -> Tuple[Core6CrossSAGE, np.ndarray, np.ndarray, np.ndarray, List[Dict[str, float]], np.ndarray, np.ndarray]:
    """Train ALVNI on a graph under a fixed SIR-label budget.

    The model is initialised once and continues optimisation across acquisition
    rounds, matching the final ALVNI implementation.  Only the final model is
    returned for prediction.
    """

    if not math.isclose(config.mse_weight + config.rank_weight, 1.0, abs_tol=1e-6):
        raise ValueError("mse_weight and rank_weight must sum to one.")
    acquisition_weight_sum = (
        config.uncertainty_weight + config.diversity_weight
        + config.representativeness_weight + config.centrality_weight
    )
    if not math.isclose(acquisition_weight_sum, 1.0, abs_tol=1e-6):
        raise ValueError("Acquisition weights must sum to one.")
    device = choose_device(config.device)
    raw_features = six_topological_features(graph)
    mean, std = raw_features.mean(axis=0), raw_features.std(axis=0)
    features = torch.as_tensor((raw_features - mean) / (std + 1e-8), dtype=torch.float32, device=device)
    edge_index = edge_index_from_graph(graph, device)
    target = torch.as_tensor(labels, dtype=torch.float32, device=device)
    budget = min(config.label_budget, graph.number_of_nodes())
    if budget < 2:
        raise ValueError("At least two labelled nodes are required for ranking-aware optimisation.")
    rng = np.random.default_rng(config.seed)
    queried = initial_query_set(raw_features, min(config.initial_labels, budget), config.initial_clusters, rng)
    model = Core6CrossSAGE(config).to(device)
    history: List[Dict[str, float]] = []
    while queried.size < budget:
        info = train_predictor(model, features, edge_index, target, queried, config, config.epochs_per_round)
        candidates = np.setdiff1d(np.arange(graph.number_of_nodes(), dtype=np.int64), queried, assume_unique=False)
        batch = min(config.query_batch, budget - queried.size)
        next_nodes = acquire_batch(model, features, edge_index, raw_features, queried, candidates, batch, config)
        queried = np.sort(np.concatenate((queried, next_nodes)))
        prediction, _ = predict(model, features, edge_index)
        history.append({"queried_labels": int(queried.size), **info, **evaluation_metrics(labels, prediction)})
        print(f"[ALVNI] queried={queried.size}/{budget} tau-b={history[-1]['kendall_tau_b']:.4f}")
    train_predictor(model, features, edge_index, target, queried, config, config.final_epochs)
    prediction, _ = predict(model, features, edge_index)
    return model, queried, prediction, raw_features, history, mean, std


def arrival_graph(observed_raw_graph: nx.Graph, arrival_edge_path: Path,
                  arrival_nodes_path: Path | None, observed_nodes: Sequence[int]) -> Tuple[nx.Graph, List[int], np.ndarray]:
    """Form \(G_{\mathrm{arr}}\), keeping observed-node indices fixed.

    ``arrival_edge_path`` contains newly observable edges.  It may include an
    edge between a new and an observed node or between two new nodes.  New
    nodes listed in ``arrival_nodes_path`` are also retained when isolated.
    """

    added = load_undirected_graph(arrival_edge_path, arrival_nodes_path)
    combined = observed_raw_graph.copy()
    combined.add_nodes_from(added.nodes)
    combined.add_edges_from(added.edges)
    observed_set = set(observed_nodes)
    new_nodes = sorted(set(combined.nodes) - observed_set)
    ordered_nodes = list(observed_nodes) + new_nodes
    reindexed = nx.relabel_nodes(combined.subgraph(ordered_nodes),
                                 {node: index for index, node in enumerate(ordered_nodes)}, copy=True)
    return reindexed, ordered_nodes, np.arange(len(observed_nodes), len(ordered_nodes), dtype=np.int64)


def infer_new_nodes(model: nn.Module, graph_arrival: nx.Graph, observed_feature_mean: np.ndarray,
                    observed_feature_std: np.ndarray, new_indices: np.ndarray,
                    device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    """Apply the frozen predictor to newly arrived nodes without label queries."""

    raw_features = six_topological_features(graph_arrival)
    features = torch.as_tensor((raw_features - observed_feature_mean) / (observed_feature_std + 1e-8),
                               dtype=torch.float32, device=device)
    edge_index = edge_index_from_graph(graph_arrival, device)
    scores, _ = predict(model, features, edge_index)
    return scores[new_indices], raw_features


def save_experiment(output_dir: Path, graph: nx.Graph, original_nodes: Sequence[int], labels: np.ndarray,
                    prediction: np.ndarray, queried: np.ndarray, features: np.ndarray,
                    history: Sequence[Mapping[str, float]], config: ALVNIConfig,
                    dataset: str, elapsed_seconds: float) -> None:
    """Write reviewable experiment outputs without overwriting source data."""

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = evaluation_metrics(labels, prediction)
    metadata = {
        "dataset": dataset,
        "graph": {
            "nodes": graph.number_of_nodes(),
            "edges": graph.number_of_edges(),
            "connected_components": nx.number_connected_components(graph),
            "self_loops": nx.number_of_selfloops(graph),
        },
        "feature_names": FEATURE_NAMES,
        "config": asdict(config),
        "queried_labels": int(queried.size),
        "runtime_seconds_excluding_sir_generation": elapsed_seconds,
        "metrics": metrics,
        "history": list(history),
    }
    (output_dir / "summary.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output_dir / "node_scores.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["local_id", "original_node_id", "sir_score", "prediction", "queried"])
        queried_set = set(queried.tolist())
        for local_id, original_id in enumerate(original_nodes):
            writer.writerow([local_id, original_id, float(labels[local_id]), float(prediction[local_id]), int(local_id in queried_set)])
    np.savetxt(output_dir / "six_topological_features.csv", features, delimiter=",", header=",".join(FEATURE_NAMES), comments="")
    with (output_dir / "queried_nodes.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["local_id", "original_node_id", "sir_score"])
        for local_id in queried:
            writer.writerow([int(local_id), original_nodes[int(local_id)], float(labels[int(local_id)])])


def save_inductive_scores(output_dir: Path, arrival_nodes: Sequence[int], new_indices: np.ndarray,
                          new_scores: np.ndarray) -> None:
    """Save only model predictions for newly arrived nodes; no SIR labels are read."""

    with (output_dir / "arrival_new_node_scores.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["arrival_local_id", "original_node_id", "prediction"])
        for index, score in zip(new_indices, new_scores):
            writer.writerow([int(index), arrival_nodes[int(index)], float(score)])


def parse_args() -> argparse.Namespace:
    package_data = Path(__file__).resolve().parent / "data"
    parser = argparse.ArgumentParser(description="Reproducible ALVNI Core6-Cross-SAGE experiment")
    parser.add_argument("--edges", type=Path, default=package_data / "Hamster_edges.txt",
                        help="Whitespace- or comma-separated undirected edge list. Defaults to bundled Hamster data.")
    parser.add_argument("--labels", type=Path, default=package_data / "Hamster_sir_beta1.25.pkl",
                        help="Offline SIR scores: .pkl, .npy, or .csv. Defaults to bundled Hamster labels.")
    parser.add_argument("--nodes-file", type=Path, default=None, help="Optional IDs of isolated nodes, one per line.")
    parser.add_argument("--dataset", default=None, help="Dataset name written to output metadata.")
    parser.add_argument("--output-dir", type=Path, default=Path("alvni_output"))
    parser.add_argument("--arrival-edges", type=Path, default=None,
                        help="Optional edge list of newly observable edges for inductive inference.")
    parser.add_argument("--arrival-nodes-file", type=Path, default=None,
                        help="Optional IDs of newly arrived isolated nodes, one per line.")
    parser.add_argument("--generate-sir", action="store_true", help="Generate offline SIR labels before ALVNI training.")
    parser.add_argument("--sir-beta-multiplier", type=float, default=1.25, help="Multiplier of the epidemic threshold when generating labels.")
    parser.add_argument("--sir-infection-rate", type=float, default=None, help="Direct beta override for optional SIR generation.")
    parser.add_argument("--sir-recovery-rate", type=float, default=1.0)
    parser.add_argument("--sir-repeats", type=int, default=1000)
    parser.add_argument("--sir-max-steps", type=int, default=10000)
    parser.add_argument("--budget", type=int, default=200)
    parser.add_argument("--initial-budget", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=40)
    parser.add_argument("--epochs-per-round", type=int, default=160)
    parser.add_argument("--final-epochs", type=int, default=240)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--sage-layers", type=int, default=2)
    parser.add_argument("--cross-layers", type=int, default=2)
    parser.add_argument("--projection-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--mse-weight", type=float, default=0.5)
    parser.add_argument("--rank-weight", type=float, default=0.5)
    parser.add_argument("--mc-passes", type=int, default=20)
    parser.add_argument("--uncertainty-weight", type=float, default=0.5)
    parser.add_argument("--diversity-weight", type=float, default=1.0 / 6.0)
    parser.add_argument("--representativeness-weight", type=float, default=1.0 / 6.0)
    parser.add_argument("--centrality-weight", type=float, default=1.0 / 6.0)
    parser.add_argument("--density-reference-limit", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=4307)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or a concrete torch device.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.labels is None and not args.generate_sir:
        raise SystemExit("Provide --labels or use --generate-sir to prepare offline labels.")
    set_seed(args.seed)
    raw_graph = load_undirected_graph(args.edges, args.nodes_file)
    dataset = args.dataset or args.edges.stem
    sir_start = time.perf_counter()
    if args.generate_sir:
        beta = args.sir_infection_rate
        if beta is None:
            beta = args.sir_beta_multiplier * epidemic_threshold(raw_graph)
        scores = generate_sir_scores(raw_graph, beta, args.sir_recovery_rate, args.sir_repeats, args.seed, args.sir_max_steps)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        label_path = args.output_dir / f"{dataset}_sir_scores.pkl"
        with label_path.open("wb") as handle:
            pickle.dump(scores, handle)
        (args.output_dir / f"{dataset}_sir_scores.meta.json").write_text(json.dumps({
            "dataset": dataset, "beta": beta, "beta_multiplier": args.sir_beta_multiplier,
            "mu": args.sir_recovery_rate, "repeats": args.sir_repeats, "seed": args.seed,
            "generation_seconds": time.perf_counter() - sir_start,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[SIR] offline labels saved to {label_path}")
    else:
        scores = load_sir_scores(args.labels)
    graph, original_nodes, labels = align_graph_and_labels(raw_graph, scores)
    print(f"[Data] {dataset}: N={graph.number_of_nodes()}, M={graph.number_of_edges()}, device={choose_device(args.device)}")
    config = ALVNIConfig(
        label_budget=args.budget, initial_labels=args.initial_budget,
        query_batch=args.batch_size, epochs_per_round=args.epochs_per_round,
        final_epochs=args.final_epochs, hidden_dim=args.hidden_dim,
        sage_layers=args.sage_layers, cross_layers=args.cross_layers,
        projection_dim=args.projection_dim, dropout=args.dropout,
        learning_rate=args.learning_rate, weight_decay=args.weight_decay,
        mse_weight=args.mse_weight, rank_weight=args.rank_weight,
        mc_passes=args.mc_passes, uncertainty_weight=args.uncertainty_weight,
        diversity_weight=args.diversity_weight,
        representativeness_weight=args.representativeness_weight,
        centrality_weight=args.centrality_weight,
        density_reference_limit=args.density_reference_limit,
        seed=args.seed, device=args.device,
    )
    started = time.perf_counter()
    model, queried, prediction, features, history, feature_mean, feature_std = run_active_learning(graph, labels, config)
    elapsed = time.perf_counter() - started
    save_experiment(args.output_dir, graph, original_nodes, labels, prediction, queried, features, history, config, dataset, elapsed)
    if args.arrival_edges is not None:
        graph_arr, arrival_nodes, new_indices = arrival_graph(raw_graph, args.arrival_edges,
                                                               args.arrival_nodes_file, original_nodes)
        if new_indices.size == 0:
            print("[Inference] no newly arrived nodes were found in the supplied arrival graph.")
        else:
            new_scores, _ = infer_new_nodes(model, graph_arr, feature_mean, feature_std, new_indices,
                                             choose_device(config.device))
            save_inductive_scores(args.output_dir, arrival_nodes, new_indices, new_scores)
            print(f"[Inference] scored {new_indices.size} newly arrived nodes without retraining or SIR queries.")
    result = evaluation_metrics(labels, prediction)
    primary_metrics = {
        "kendall_tau_b": result["kendall_tau_b"],
        "jaccard_at_100": result["jaccard_at_100"],
        "ndcg_at_100": result["ndcg_at_100"],
        "imprecision_at_100": result["imprecision_at_100"],
    }
    (args.output_dir / "primary_metrics.json").write_text(
        json.dumps(primary_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[Metrics] " + "; ".join(f"{name}={value:.6f}" for name, value in primary_metrics.items()))
    print(f"[Done] runtime excluding offline SIR={elapsed:.2f}s")
    print(f"[Done] results saved in {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
