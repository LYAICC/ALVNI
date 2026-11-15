ALVNI reproduction package
This package provides the ALVNI implementation and a small reproduction example using the Twitter network and cached SIR influence labels at $\beta/\beta_{\mathrm{th}} = 1.25$. 

Files
- ALVNI-all.py: ALVNI implementation.
- twitter.txt: Twitter network edge list for the example run.
- twitter_beta_1.25_sir_scores.pkl: Cached Twitter SIR influence labels at $\beta/\beta_{\mathrm{th}} = 1.25$.
- requirements.txt: Python dependencies.
- README.md: Instructions for the example run.

Environment
The reference environment is Python 3.9.23, PyTorch 2.8.0+cu128, and
PyTorch Geometric 2.6.1. Install the dependencies with:
pip install -r requirements.txt
For a CPU-only computer, install a PyTorch CPU build compatible with the local
platform, then install the remaining packages. Use --device cpu when running.
Run one experiment

The example below trains ALVNI on Twitter with the included cached labels at
$\beta/\beta_{\mathrm{th}} = 1.25$:
python .\ALVNI-all.py `
  --edges ".\twitter.txt" `
  --labels ".\twitter_beta_1.25_sir_scores.pkl" `
  --dataset twitter `
  --output-dir ".\outputs\twitter_beta1.25_seed4307" `
  --seed 4307 `
  --device auto

Default protocol
B=200, B0=40, batch size=40
MSE/ranking weights=(0.5, 0.5)
Acquisition weights (U,D,R,C)=(0.5, 1/6, 1/6, 1/6)
Two CrossNet layers; two mean-GraphSAGE layers; projection=32; hidden=128
Dropout=0.35; learning rate=0.001; weight decay=0.0005
MC-dropout passes=20; seed=4307
