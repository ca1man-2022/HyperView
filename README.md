# HyperView

Scripts to construct node-level hypergraph OOD benchmarks with feature, semantic, and structural shifts.

## Raw data and upstream code

All six source datasets are available from the [EDGNN dataset collection](https://huggingface.co/datasets/peihaowang/edgnn-hypergraph-dataset/tree/28b7930cb54880988e6ce711ff5d30bb49769282):

| Dataset | Source directory | Earlier research code |
|---|---|---|
| Cora-CA | `coauthorship/cora` | [HyperGCN](https://github.com/malllabiisc/HyperGCN) |
| Cora | `cocitation/cora` | [HyperGCN](https://github.com/malllabiisc/HyperGCN) |
| Citeseer | `cocitation/citeseer` | [HyperGCN](https://github.com/malllabiisc/HyperGCN) |
| House-committees-100 | `house-committees` | [Cornell source](https://www.cs.cornell.edu/~arb/data/house-committees/) |
| NTU2012 | `NTU2012` | [LEGCN](https://github.com/ycq091044/LEGCN) |
| Mushroom | `Mushroom` | [LEGCN](https://github.com/ycq091044/LEGCN) |

[AllSet](https://github.com/jianhao2016/AllSet) provides related hypergraph preprocessing code and original raw-data instructions. Please cite the relevant dataset papers and upstream repositories. Downloaded `.pickle` files must be treated as trusted inputs; Python pickle loading is unsafe for untrusted sources.
