"""Build a k-nearest-neighbour spatial graph over Indian districts using real
lat/lon centroids (in place of a full polygon-adjacency shapefile, which is
unnecessary for the graph-propagation term of the model -- k-NN over
centroids is a standard, much lighter proxy for "neighbouring districts" used
throughout the crime-forecasting GNN literature).
"""
from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from src.config import DATA_PROCESSED, KNN_NEIGHBORS


def haversine_knn_edges(lat: np.ndarray, lon: np.ndarray, k: int) -> list[tuple[int, int, float]]:
    coords_rad = np.radians(np.column_stack([lat, lon]))
    tree = BallTree(coords_rad, metric="haversine")
    dist, idx = tree.query(coords_rad, k=k + 1)  # +1 because self is included
    earth_radius_km = 6371.0
    edges = []
    for i in range(len(lat)):
        for j_rank in range(1, k + 1):  # skip self at rank 0
            j = idx[i, j_rank]
            d_km = dist[i, j_rank] * earth_radius_km
            edges.append((int(i), int(j), float(d_km)))
    return edges


def build_graph(k: int = KNN_NEIGHBORS) -> nx.Graph:
    df = pd.read_csv(DATA_PROCESSED / "districts_harmonized.csv")
    edges = haversine_knn_edges(df["lat"].to_numpy(), df["lon"].to_numpy(), k)

    g = nx.Graph()
    for _, row in df.iterrows():
        g.add_node(
            int(row["district_id"]),
            state=row["state"],
            district=row["district"],
            lat=row["lat"],
            lon=row["lon"],
            population=row["population"],
        )
    for i, j, d_km in edges:
        weight = 1.0 / (1.0 + d_km)  # closer districts get a stronger edge weight
        if g.has_edge(i, j):
            g[i][j]["weight"] = max(g[i][j]["weight"], weight)
        else:
            g.add_edge(i, j, weight=weight, distance_km=d_km)
    return g


def save_graph(g: nx.Graph, path=None):
    path = path or (DATA_PROCESSED / "district_graph.gpickle")
    nx.write_gpickle(g, path) if hasattr(nx, "write_gpickle") else _write_gpickle_fallback(g, path)
    return path


def _write_gpickle_fallback(g: nx.Graph, path):
    import pickle

    with open(path, "wb") as f:
        pickle.dump(g, f)


def load_graph(path=None) -> nx.Graph:
    path = path or (DATA_PROCESSED / "district_graph.gpickle")
    import pickle

    with open(path, "rb") as f:
        return pickle.load(f)


def adjacency_matrix(g: nx.Graph) -> np.ndarray:
    n = g.number_of_nodes()
    a = np.zeros((n, n), dtype=np.float32)
    for i, j, data in g.edges(data=True):
        a[i, j] = a[j, i] = data["weight"]
    return a


def main():
    g = build_graph()
    path = save_graph(g)
    print(f"Graph with {g.number_of_nodes()} nodes, {g.number_of_edges()} edges saved to {path}")
    degrees = [d for _, d in g.degree()]
    print(f"Degree: min={min(degrees)} max={max(degrees)} mean={np.mean(degrees):.1f}")


if __name__ == "__main__":
    main()
