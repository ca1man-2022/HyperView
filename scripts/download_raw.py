"""Download upstream raw hypergraph files; no processed tensors are distributed."""
import argparse
from pathlib import Path
from urllib.request import urlopen
import shutil

# Pinned snapshot of an external ICLR 2023 dataset collection.
_REV = '28b7930cb54880988e6ce711ff5d30bb49769282'
_BASE = 'https://huggingface.co/datasets/peihaowang/edgnn-hypergraph-dataset/resolve/' + _REV

SOURCES = {
    'cora-ca': ('coauthorship/cora', ('features.pickle', 'labels.pickle', 'hypergraph.pickle')),
    'cora': ('cocitation/cora', ('features.pickle', 'labels.pickle', 'hypergraph.pickle')),
    'citeseer': ('cocitation/citeseer', ('features.pickle', 'labels.pickle', 'hypergraph.pickle')),
    'house-committees-100': ('house-committees', ('node-labels-house-committees.txt', 'hyperedges-house-committees.txt')),
    'ntu2012': ('NTU2012', ('NTU2012.content', 'NTU2012.edges')),
    'mushroom': ('Mushroom', ('Mushroom.content', 'Mushroom.edges')),
}


def download(dataset, root, overwrite=False):
    selected = SOURCES if dataset == 'all' else {dataset: SOURCES[dataset]}
    for _name, (prefix, filenames) in selected.items():
        for filename in filenames:
            path = root / prefix / filename
            if path.is_file() and not overwrite:
                print(f'Skip {path}')
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            url = f'{_BASE}/{prefix}/{filename}'
            temp = path.with_name(path.name + '.download')
            print(f'GET {url}')
            try:
                with urlopen(url, timeout=90) as response, temp.open('wb') as target:
                    shutil.copyfileobj(response, target)
                if temp.stat().st_size == 0:
                    raise ValueError(f'Empty source file: {url}')
                temp.replace(path)
            finally:
                temp.unlink(missing_ok=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=['all', *SOURCES], default='all')
    p.add_argument('--root', type=Path, default=Path('data/raw'))
    p.add_argument('--overwrite', action='store_true')
    args = p.parse_args()
    download(args.dataset, args.root, args.overwrite)


if __name__ == '__main__':
    main()
