# ALVNI reproduction package

This package contains the ALVNI1104 implementation, nine network edge lists,
and cached SIR influence labels at
\(\beta/\beta_{\mathrm{th}}\in\{0.5,0.75,1,1.25,1.5,1.75,2\}\).

## Files

- `ALVNI1104投稿版脚本.py`: Core6-Cross-SAGE ALVNI implementation.
- `九个数据集与sir标签/`: nine edge lists and 63 cached SIR-label files.
- `ALVNI-肯德尔折线图可视化脚本.py`: plotting source for the paper's Kendall curves.

The six local topological features are constructed from the edge list at run time.
No feature cache or model checkpoint is required.

## Environment

The reference environment is Python 3.9.23, PyTorch 2.8.0+cu128, and
PyTorch Geometric 2.6.1. Install the dependencies with:

```powershell
pip install -r requirements.txt
```

For a CPU-only computer, install a PyTorch CPU build compatible with the local
platform, then install the remaining packages. Use `--device cpu` when running.

## Run one experiment

The example below trains ALVNI on Hamster with the cached labels at
\(\beta/\beta_{\mathrm{th}}=1.25\):

```powershell
python .\ALVNI1104投稿版脚本.py `
  --edges ".\九个数据集与sir标签\Hamster.txt" `
  --labels ".\九个数据集与sir标签\Hamster_SIR分数\Hamster_beta_1.25_sir_scores.pkl" `
  --dataset Hamster `
  --output-dir ".\outputs\Hamster_beta1.25_seed4307" `
  --seed 4307 `
  --device auto
```

Available dataset names are:

```text
Hamster, cora, powergrid, CAGrQc, DD21, LastFM, CA-HepTh, Chicago, twitter
```

For another propagation condition, replace only the SIR-label filename, for
example `Hamster_beta_0.5_sir_scores.pkl` or `Hamster_beta_2_sir_scores.pkl`.

## Default protocol

```text
B=200, B0=40, batch size=40
MSE/ranking weights=(0.5, 0.5)
Acquisition weights (U,D,R,C)=(0.5, 1/6, 1/6, 1/6)
Two CrossNet layers; two mean-GraphSAGE layers; projection=32; hidden=128
Dropout=0.35; learning rate=0.001; weight decay=0.0005
MC-dropout passes=20; seed=4307
```

The script writes `summary.json`, predictions, queried-node IDs, constructed
features, and the four metrics: Kendall's \(\tau_b\), Jaccard@100, NDCG@100,
and Imprecision@100.

## Note on the plotting script

`ALVNI-肯德尔折线图可视化脚本.py` requires the aggregated paper-result CSV,
including baseline rows. That CSV is not included in this package, so the
plotting script is supplied as source only and is not directly runnable here.