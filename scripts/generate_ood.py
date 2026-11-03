"""Produce separate train/test OOD samples from a converted source hypergraph."""
import argparse
from pathlib import Path
from hyperview_data import (load, set_seed, create_feature_ood,
                            create_semantic_ood, create_structure_ood)


def run(base_file, output, shift='all', seed=42, ood_ratio=0.3,
        noise_scale=0.1, drop_ratio=0.5, structure_scale=1.5):
    base = load(base_file)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    choices = ['feature', 'semantic', 'structure'] if shift == 'all' else [shift]
    for kind in choices:
        # Separate CLI runs of different shift types would start with this seed.
        set_seed(seed)
        if kind == 'feature':
            fn = lambda: create_feature_ood(base, ood_ratio, noise_scale)
        elif kind == 'semantic':
            fn = lambda: create_semantic_ood(base, ood_ratio, drop_ratio)
        elif kind == 'structure':
            fn = lambda: create_structure_ood(base, ood_ratio, structure_scale)
        else:
            raise ValueError(f'unknown shift: {kind}')
        # Consistent with original get_ood_datasets(): two independent draws.
        for name in ('train', 'test'):
            sample = fn()
            file = output / f'{kind}_{name}.npz'
            sample.data.save(file, node_idx=sample.node_idx)
            print(f'{file}: {len(sample.node_idx)} OOD node IDs')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--shift', choices=['all','feature','semantic','structure'], default='all')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--ood-ratio', type=float, default=0.3)
    p.add_argument('--noise-scale', type=float, default=0.1)
    p.add_argument('--drop-ratio', type=float, default=0.5)
    p.add_argument('--structure-scale', type=float, default=1.5)
    a = p.parse_args()
    run(a.input, a.output, a.shift, a.seed, a.ood_ratio,
        a.noise_scale, a.drop_ratio, a.structure_scale)


if __name__ == '__main__':
    main()
