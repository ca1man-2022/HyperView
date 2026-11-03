"""Small NPZ representation for node-hyperedge incidence data."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass(frozen=True)
class Hypergraph:
    x: np.ndarray
    y: np.ndarray
    nodes: np.ndarray
    edges: np.ndarray
    n_edges: int

    @property
    def n_nodes(self):
        return int(self.x.shape[0])

    def validate(self):
        if self.x.ndim != 2 or self.x.shape[0] < 1 or self.x.shape[1] < 1:
            raise ValueError('x should have shape [N,D]')
        if self.y.ndim != 1 or len(self.y) != self.n_nodes:
            raise ValueError('y should have shape [N]')
        if self.nodes.ndim != 1 or self.edges.ndim != 1 or len(self.nodes) != len(self.edges) or not len(self.nodes):
            raise ValueError('incidence indices must be non-empty matching vectors')
        if not np.isfinite(self.x).all():
            raise ValueError('nonfinite node features')
        if not np.issubdtype(self.y.dtype, np.integer):
            raise ValueError('labels should be integer indices')
        if self.nodes.min() < 0 or self.nodes.max() >= self.n_nodes:
            raise ValueError('node IDs out of bounds')
        if self.n_edges < 1 or self.edges.min() < 0 or self.edges.max() >= self.n_edges:
            raise ValueError('hyperedge IDs out of bounds')
        # The historical PyG generator may output repeated memberships;
        # intentionally do not coalesce these before exporting.
        return self

    def save(self, path, node_idx=None):
        self.validate()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = dict(x=self.x.astype('float32'), y=self.y.astype('int64'),
                      node_ids=self.nodes.astype('int64'),
                      edge_ids=self.edges.astype('int64'),
                      n_edges=np.int64(self.n_edges))
        if node_idx is not None:
            arrays['node_idx'] = np.asarray(node_idx, dtype=np.int64)
        np.savez_compressed(path, **arrays)


def load(path):
    with np.load(path, allow_pickle=False) as data:
        required = {'x', 'y', 'node_ids', 'edge_ids', 'n_edges'}
        missing = required - set(data.files)
        if missing:
            raise ValueError(f'missing keys: {sorted(missing)}')
        result = Hypergraph(np.asarray(data['x'], dtype=np.float32),
                            np.asarray(data['y'], dtype=np.int64),
                            np.asarray(data['node_ids'], dtype=np.int64),
                            np.asarray(data['edge_ids'], dtype=np.int64),
                            int(data['n_edges']))
    return result.validate()
