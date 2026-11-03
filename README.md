# HyperView

Scripts to construct node-level hypergraph OOD benchmarks with feature, semantic, and structural shifts.

## Raw datasets

| Dataset | Source | Upstream reference |
|---|---|---|
| Cora-CA | [coauthorship/cora](https://huggingface.co/datasets/peihaowang/edgnn-hypergraph-dataset/tree/main/coauthorship/cora) | [HyperGCN](https://github.com/malllabiisc/HyperGCN) |
| Cora | [cocitation/cora](https://huggingface.co/datasets/peihaowang/edgnn-hypergraph-dataset/tree/main/cocitation/cora) | [HyperGCN](https://github.com/malllabiisc/HyperGCN) |
| Citeseer | [cocitation/citeseer](https://huggingface.co/datasets/peihaowang/edgnn-hypergraph-dataset/tree/main/cocitation/citeseer) | [HyperGCN](https://github.com/malllabiisc/HyperGCN) |
| House-committees-100 | [Cornell House committees](https://www.cs.cornell.edu/~arb/data/house-committees/) | [AllSet](https://github.com/jianhao2016/AllSet) |
| NTU2012 | [NTU2012](https://huggingface.co/datasets/peihaowang/edgnn-hypergraph-dataset/tree/main/NTU2012) | [LEGCN](https://github.com/ycq091044/LEGCN) |
| Mushroom | [Mushroom](https://huggingface.co/datasets/peihaowang/edgnn-hypergraph-dataset/tree/main/Mushroom) | [LEGCN](https://github.com/ycq091044/LEGCN) |

## Usage

```bash
python -m pip install -r requirements.txt
python -m scripts.download_raw --dataset all
python -m scripts.import_base pickles --directory data/raw/cocitation/cora --trusted-source --output data/base/cora.npz
python -m scripts.generate_ood --input data/base/cora.npz --output data/generated/cora --shift all --seed 42
```

To convert other datasets, see `python -m scripts.import_base --help`. The source files are obtained from third-party repositories. Only load `.pickle` data from sources you trust.

The three OOD generators in `hyperview_data/shifts.py` follow the **statistical sampling rules** used in the development file `newdataset_HOOD.py`: 30% perturbed nodes with Gaussian noise for feature OOD; 30% selected hyperedges and 50% cross-class member replacement for semantic OOD; and 30% selected hyperedges with 1.5× membership resampling for structure OOD. Separate training and testing draws are produced. The original `node_idx` convention is retained, including its limitations.
