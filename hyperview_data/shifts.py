"""HyperView data shifts.
"""
from dataclasses import dataclass
import random
import numpy as np
import torch
from .schema import Hypergraph


@dataclass(frozen=True)
class Shift:
    data: Hypergraph
    node_idx: np.ndarray  # Exactly the OOD-node convention of the original generator


def _memberships(base):
    # Reproduce extraction of forward (node -> hyperedge) incidence entries.
    groups = {}
    for node, edge in zip(base.nodes, base.edges):
        groups.setdefault(int(edge), []).append(int(node))
    return groups


def _assemble(base, groups):
    nodes, edges = [], []
    for edge, members in groups.items():
        nodes.extend(members)
        edges.extend([edge] * len(members))
    return Hypergraph(base.x.copy(), base.y.copy(),
                      np.asarray(nodes, dtype=np.int64),
                      np.asarray(edges, dtype=np.int64),
                      base.n_edges).validate()


def create_feature_ood(base, ood_ratio=0.3, noise_scale=1.0):
    """Randomly perturb 30% of all nodes with Gaussian feature noise."""
    x = torch.tensor(base.x, dtype=torch.float32).clone()
    n = base.n_nodes
    selected = torch.randperm(n)[:int(n * ood_ratio)]
    x[selected] += noise_scale * torch.randn_like(x[selected])
    shifted = Hypergraph(x.numpy(), base.y.copy(), base.nodes.copy(),
                         base.edges.copy(), base.n_edges).validate()
    return Shift(shifted, selected.numpy().astype(np.int64))


def create_semantic_ood(base, ood_ratio=0.3, drop_ratio=0.5):
    """Select hyperedges; replace sampled members by nodes of a different class.

    The original generator does not match node degrees or guarantee that sampled
    replacements are distinct from existing members. This is intentional here.
    """
    groups = _memberships(base)
    all_edges = list(groups)
    selected = set(random.sample(all_edges, int(len(all_edges) * ood_ratio)))
    y = torch.as_tensor(base.y, dtype=torch.long)
    class_to_nodes = {int(c): torch.where(y == c)[0]
                      for c in y.unique().tolist()}
    changed = {}
    new_nodes = set()
    for edge, member_list in groups.items():
        members = torch.tensor(member_list, dtype=torch.long)
        if edge not in selected:
            result = members
        else:
            k = len(members)
            replace_k = max(1, int(k * drop_ratio))
            major_label = int(torch.mode(y[members]).values.item())
            candidate_classes = [c for c in class_to_nodes
                                 if c != major_label and len(class_to_nodes[c]) > 0]
            if not candidate_classes:
                result = members
            else:
                target_class = random.choice(candidate_classes)
                pool = class_to_nodes[target_class]
                replacements = pool[torch.randperm(len(pool))[:replace_k]]
                keep = members[torch.randperm(k)[:k-replace_k]]
                result = torch.cat([keep, replacements])
                new_nodes.update(replacements.tolist())
        changed[edge] = result.tolist()
    shifted = _assemble(base, changed)
    return Shift(shifted, np.array(sorted(new_nodes), dtype=np.int64))


def create_structure_ood(base, ood_ratio=0.3, scale=1.5):
    """Select hyperedges; resample their members at scale*original hyperedge size."""
    groups = _memberships(base)
    all_edges = list(groups)
    selected = set(random.sample(all_edges, int(len(all_edges) * ood_ratio)))
    changed = {}
    new_nodes = set()
    for edge, members in groups.items():
        if edge not in selected:
            result = members
        else:
            new_k = max(1, int(len(members) * scale))
            result = torch.randperm(base.n_nodes)[:new_k].tolist()
            new_nodes.update(result)
        changed[edge] = result
    shifted = _assemble(base, changed)
    return Shift(shifted, np.array(sorted(new_nodes), dtype=np.int64))


def set_seed(seed):
    """Reproducible generated examples; original script did not seed before OOD sampling."""
    random.seed(seed)
    torch.manual_seed(seed)
