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

Use Python 3.10 in a fresh virtual environment (these pinned versions support
Python 3.8-3.11, not 3.12+). The data-preparation scripts run on CPU:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pip check
python -m scripts.download_raw --dataset all
python -m scripts.import_base pickles --directory data/raw/cocitation/cora --trusted-source --output data/base/cora.npz
python -m scripts.generate_ood --input data/base/cora.npz --output data/generated/cora --shift all --seed 42
```

On Windows, activate with `.venv\Scripts\activate` instead. For a CUDA 11.8
PyTorch build on supported Linux/Windows systems, replace `/cpu` with `/cu118`
in the PyTorch install command; a compatible NVIDIA driver is required.
The CPU/CUDA local version suffix satisfies `torch==2.1.2` in the requirements.

`requirements.txt` contains the dependencies used by this public data-preparation
code, including SciPy for loading sparse feature matrices in trusted pickles.
PyG extensions, plotting tools and other unused packages from the original
environment are not needed here. `environment_full.yml` is retained as a historical
training-environment export, not as a tested installation recipe.

Validated on Linux x86_64 with Python 3.10.22 and PyTorch 2.1.2+cpu: clean
installation, `pip check`, all three CLI help commands, and synthetic sparse-pickle
conversion followed by all six train/test shift exports. CUDA, other platforms
and full dataset experiments were not tested in this dependency check.

To convert other datasets, see `python -m scripts.import_base --help`. The source files are obtained from third-party repositories. Only load `.pickle` data from sources you trust.

The three OOD generators in `hyperview_data/shifts.py` follow the **statistical sampling rules** used in the development file `newdataset_HOOD.py`: 30% perturbed nodes with Gaussian noise for feature OOD; 30% selected hyperedges and 50% cross-class member replacement for semantic OOD; and 30% selected hyperedges with 1.5× membership resampling for structure OOD. Separate training and testing draws are produced. The original `node_idx` convention is retained, including its limitations.
