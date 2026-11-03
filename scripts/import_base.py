"""Local-only conversion of source benchmark formats to pickle-free HyperView NPZ.

WARNING: legacy .pickle input executes arbitrary code during loading. Supply only
files you produced or independently trust; never load untrusted downloads.
This converter does not download or redistribute any upstream dataset.
"""
import argparse
from pathlib import Path
import pickle
import hashlib
import json
import numpy as np
from hyperview_data.schema import Hypergraph


def _fingerprint(path):
    d = hashlib.sha256()
    with open(path,'rb') as f:
        for part in iter(lambda:f.read(1048576),b''):
            d.update(part)
    return d.hexdigest()


def _from_sets(x, y, groups):
    x = np.asarray(x, dtype=np.float32)
    raw_y = np.asarray(y).reshape(-1)
    if x.ndim != 2 or x.shape[0] != raw_y.shape[0]:
        raise ValueError('source features and labels have incompatible shapes')
    # Preserve original label IDs only when they are integral; relabel other formats.
    if np.issubdtype(raw_y.dtype, np.integer):
        labels = raw_y.astype(np.int64)
    else:
        _, labels = np.unique(raw_y, return_inverse=True)
        labels = labels.astype(np.int64)
    node_ids, edge_ids = [], []
    for eid, members in enumerate(groups):
        for member in sorted(set(int(m) for m in members)):
            node_ids.append(member)
            edge_ids.append(eid)
    return Hypergraph(x, labels, np.asarray(node_ids,np.int64),
                      np.asarray(edge_ids,np.int64), len(groups)).validate()


def from_pickle_directory(folder):
    with open(folder/'features.pickle','rb') as f:
        x = pickle.load(f)
    with open(folder/'labels.pickle','rb') as f:
        y = pickle.load(f)
    with open(folder/'hypergraph.pickle','rb') as f:
        h = pickle.load(f)
    if hasattr(x, 'toarray'):
        x = x.toarray()
    if not isinstance(h, dict):
        raise ValueError('hypergraph.pickle must be a dict of hyperedge -> node IDs')
    # Dict insertion order is significant. Snapshot an explicit hash of source files.
    return _from_sets(x, y, list(h.values()))


def from_cornell(labels_path, edges_path, noise_scale, feature_dim, seed):
    labels = np.loadtxt(labels_path, dtype=np.int64, ndmin=1)
    _, cls_idx = np.unique(labels, return_inverse=True)
    classes = int(cls_idx.max())+1
    d = classes if feature_dim is None else feature_dim
    if d < classes or noise_scale < 0:
        raise ValueError('feature dimension must cover classes; noise must be non-negative')
    x = np.zeros((len(labels),d),np.float32)
    x[np.arange(len(labels)), cls_idx] = 1.
    rng = np.random.default_rng(seed)
    x += rng.normal(0.,noise_scale,x.shape).astype(np.float32)
    groups = []
    with open(edges_path,encoding='utf-8') as f:
        for line in f:
            if line.strip():
                groups.append([int(part.strip()) for part in line.split(',') if part.strip()])
    if not groups:
        raise ValueError('no hyperedges found')
    used_ids = sorted({n for group in groups for n in group})
    # Cornell source has occurred with zero-based and one-based node IDs.
    if used_ids and min(used_ids) == 1 and max(used_ids) <= len(labels):
        groups = [[n-1 for n in group] for group in groups]
    return _from_sets(x,cls_idx,groups)


def from_le(content_path, edges_path):
    """Read LEGCN's bipartite .content and .edges files.

    The .content file lists both node and hyperedge identifiers. The .edges
    file uses those identifiers, not zero-based incidence IDs. Unlike the
    historical PyG loader, only the true node rows become sample features.
    """
    rows = np.genfromtxt(content_path, dtype=str, ndmin=2)
    pairs = np.genfromtxt(edges_path, dtype=str, ndmin=2)
    if rows.shape[1] < 3 or pairs.shape[1] != 2:
        raise ValueError('expected .content [id, features..., label], .edges [node_id, hyperedge_id]')
    ids = rows[:, 0].tolist()
    if len(set(ids)) != len(ids):
        raise ValueError('duplicate identifiers in .content file')
    row_map = {name: i for i, name in enumerate(ids)}
    node_id_set = set(pairs[:, 0].tolist())
    hyperedge_id_set = set(pairs[:, 1].tolist())
    if node_id_set & hyperedge_id_set:
        raise ValueError('node and hyperedge identifiers overlap')
    if len(node_id_set) < 4 or not hyperedge_id_set:
        raise ValueError('too few nodes/hyperedges in source')
    missing = (node_id_set | hyperedge_id_set) - set(ids)
    if missing:
        raise ValueError(f'.edges identifiers absent from .content: {sorted(missing)[:5]}')
    # Preserve the original row order; sorting identifiers can silently
    # reindex nodes differently from the historical dataset loader.
    ordered_nodes = [name for name in ids if name in node_id_set]
    ordered_hyperedges = [name for name in ids if name in hyperedge_id_set]
    x = rows[[row_map[name] for name in ordered_nodes], 1:-1].astype(np.float32)
    y = rows[[row_map[name] for name in ordered_nodes], -1]
    node_index = {name: i for i, name in enumerate(ordered_nodes)}
    hyperedge_index = {name: i for i, name in enumerate(ordered_hyperedges)}
    groups = [[] for _ in ordered_hyperedges]
    for node_name, edge_name in pairs:
        groups[hyperedge_index[edge_name]].append(node_index[node_name])
    return _from_sets(x, y, groups)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sp=p.add_subparsers(dest='format',required=True)
    a=sp.add_parser('pickles',help='local (features, labels, hypergraph) .pickle files')
    a.add_argument('--directory',type=Path,required=True)
    a.add_argument('--trusted-source',action='store_true',required=False)
    b=sp.add_parser('cornell',help='Cornell node-labels and hyperedges text files')
    b.add_argument('--labels',type=Path,required=True)
    b.add_argument('--hyperedges',type=Path,required=True)
    b.add_argument('--feature-dim',type=int)
    b.add_argument('--noise-scale',type=float,required=True)
    b.add_argument('--seed',type=int,required=True)
    c=sp.add_parser('le',help='LE .content and .edges files')
    c.add_argument('--content',type=Path,required=True)
    c.add_argument('--edges',type=Path,required=True)
    for parser in (a,b,c):
        parser.add_argument('--output',type=Path,required=True)
    opts=p.parse_args()
    if opts.format=='pickles':
        if not opts.trusted_source:
            p.error('--trusted-source required: pickle inputs can execute arbitrary code')
        data=from_pickle_directory(opts.directory)
        source_paths=[opts.directory/name for name in ('features.pickle','labels.pickle','hypergraph.pickle')]
    elif opts.format=='cornell':
        data=from_cornell(opts.labels,opts.hyperedges,opts.noise_scale,opts.feature_dim,opts.seed)
        source_paths=[opts.labels,opts.hyperedges]
    else:
        data=from_le(opts.content,opts.edges)
        source_paths=[opts.content,opts.edges]
    data.save(opts.output)
    metadata={'format':opts.format,'source_sha256': {p.name:_fingerprint(p) for p in source_paths},
              'output':opts.output.name,'node_count':data.n_nodes,'hyperedge_count':data.n_edges,
              'feature_dim':int(data.x.shape[1]),'description':'source provenance only; check licensing before redistributing converted data'}
    opts.output.with_suffix('.source.json').write_text(json.dumps(metadata,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    print(f'Converted {data.n_nodes} nodes, {data.n_edges} hyperedges to {opts.output}')


if __name__=='__main__':
    main()
