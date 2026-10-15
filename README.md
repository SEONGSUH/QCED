# QCED — Quantum Compass-Gradient Edge Detection

Quantum implementation of 8-directional compass-gradient edge detection
(Prewitt, Sobel, Kirsch). By exploiting the circulant structure of the
eight directional masks, all eight directional responses are computed in
a single **4-qubit circuit (3 data + 1 ancilla)** via the QFT and an
ancilla-assisted block encoding of the operator eigenvalues.

Backends: statevector (ideal), shots, density-matrix noisy simulation,
and **real IBM Quantum hardware**. Classical post-processing produces
three edge maps (magnitude response, NMS-thinned maximum, Otsu binary)
and reports MSE/PSNR, cosine similarity, and direction accuracy.

## Install

```bash
conda create -n qced python=3.11 -y
conda activate qced
pip install -r requirements.txt
pip install -e .
```

For real quantum hardware, additionally:

```bash
pip install pennylane-qiskit
# edit ibm_account.py: paste your IBM Quantum token + instance CRN, then:
python ibm_account.py
```

## Quick start

```bash
# ideal (statevector), magnitude output
python demo.py --image synthetic --operator kirsch --backend statevector --detection response

# noisy simulation, shot sweep, binary edge map
python demo.py --image pirate.tif --operator kirsch --backend noisy --detection binary
```

Options: `--image {synthetic,<file>}`,
`--operator {prewitt,sobel,kirsch}`,
`--backend {statevector,shots,noisy,qpu}`,
`--detection {response,maximum,binary}`.
Outputs (edge maps, metrics `.xlsx`, figures) are written under
`results/<image>_<operator>_<backend>_<detection>/`.

## Reproduce the paper's results

`run_all.sh` reruns every configuration reported in the paper
(both images × three operators × simulation backends × detections),
generates the figures, and runs the IBM-hardware experiments:

```bash
bash run_all.sh
```

It is organised in three parts:

1. **Simulation** — `demo.py` over `{pirate, synthetic}` ×
   `{prewitt, sobel, kirsch}` × `{statevector, noisy}` ×
   `{response, binary}` (uses `--skip-existing` so it is resumable).
2. **Figures** — `plot_dual_axis.py` on the noisy-backend `.xlsx`
   (MSE/PSNR and cosine/direction-accuracy vs. shots).
3. **Quantum hardware** — `run_qpu_experiment.py` submits the circuit to
   `ibm_fez` (checkpointed/resumable) for shots 512 and 1024 over three
   independent runs; `demo.py --backend qpu` then post-processes the
   saved counts into results **without re-submitting** the QPU.

> The hardware part needs a configured `ibm_account.py`. To rehearse the
> checkpoint/resume flow without an account, add `--simulate` to
> `run_qpu_experiment.py`.

## Repository layout
```
src/compass_gradient/      core package
    operators.py             compass operators: h, circulant matrix, eigenvalues
    quantum.py               QCED circuit + backends (statevector/shots/noisy/qpu)
    core.py                  classical compass_edges + post-processing

demo.py                    main pipeline: run a backend, post-process, save results
run_qpu_experiment.py      checkpointed / resumable IBM-hardware runner
run_all.sh                 full paper-reproduction script

plot_dual_axis.py          MSE/PSNR & cos/dir-acc vs. shots figures
compare_backends.py        overlay two backends (e.g. shots vs. noisy)

ibm_account.py             IBM Quantum credential setup (fill in your token)
pirate.tif                 test image (synthetic is generated on the fly)
requirements.txt           dependencies

results/                   output edge maps, metrics (.xlsx), figures
qpu_runs/                  raw IBM-hardware measurement checkpoints (JSON)
```