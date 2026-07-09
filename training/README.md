# training

GNN node-embedding training on HGCal layer clusters (HDF5 format from
`caloembed/data/preprocess.py`). Model and loss are swappable by name via
small registries; the train loop is generic.

## Run

```bash
cd training
./run_train.sh                                        # configs/default.yaml
./run_train.sh --max-files 20 --epochs 5              # quick run
./run_train.sh --resume                               # continue from last.pt
python train.py --config configs/default.yaml --device cpu   # bare, no env setup

# on condor (from repo root):
condor_submit training/condor/train.sub
```

Outputs land in `output.dir` (see config): `config.yaml` (resolved snapshot),
`metrics.csv` (per-epoch losses), `best.pt` (model weights, lowest val loss),
`last.pt` (full state for `--resume`), `summary.json` (git hash + results).
A fresh run refuses to write into a directory that already has a `last.pt`.

## Current setup

- **Model** `edgeconv` (`src/models/edgeconv.py`): port of the reference
  contrastive Net from `geant4sim/scripts/training/Contrastive/src/model.py`
  — MLP encoder → kNN graph built once in the latent space → static EdgeConv
  stack → 64→32→`out_dim` projection head with dropout. Reference defaults
  (hidden 64, 4 layers, dropout 0.3, out_dim 8, k 20); `in_dim` follows the
  configured feature list.
- **Loss** `infonce` (`src/losses/infonce.py`): per-event supervised InfoNCE;
  positives = layer clusters with the same truth shower id
  (`truth/argmax_cp_idx`), node-equal reduction, temperature-scaled cosine
  similarity.
- **Data** (`src/data.py`): whole files loaded to memory as normalized
  tensors (fixed scales in `FEATURE_TRANSFORMS`); one PyG `Data` per event
  with `x` (features), `y` (shower id), `frac` (argmax energy fraction —
  available to losses that want to down-weight ambiguous/noise nodes).
- **Split**: file-level train/val/test, stratified by the file's particle
  count (each file holds a single n_cp), deterministic in `train.seed`.
  Test files are recorded in `<output>/splits.json` but never loaded.

## Tests

```bash
cd training
python -m pytest -q                     # 19 tests: loss math, model, data
python -m pytest -q -m integration      # also touch one real d5 file
```

InfoNCE is checked against an independent loop-based reference, analytic
closed-form cases (2- and 3-node events, the log(k) many-positive floor),
and cross-event independence; the model against the reference architecture,
batch non-mixing, and gradient flow; the data pipeline against synthetic
HDF5 files with known values through to a full forward/backward.

## Adding a model or loss

1. New module in `src/models/` (or `src/losses/`) with
   `@register_model("myname")` on an `nn.Module`.
   - model contract: `forward(data) -> (N_nodes, out_dim)` embeddings
   - loss contract: `forward(embeddings, data) -> scalar`
2. Import it at the bottom of the package `__init__.py`.
3. Select it in a config: `model.name: myname`; all other keys in that block
   are passed to the constructor.

## Environment

Uses the `caloembed` micromamba env (torch 2.5 + cu124), plus
`torch_geometric` and `torch-cluster`:

```bash
pip install torch_geometric
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.5.0+cu124.html
```
