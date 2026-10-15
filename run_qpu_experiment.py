"""Run the QCED circuit on real quantum hardware (IBM QPU) with
checkpointing, so an interrupted run (token exhaustion, queue error,
crash) can be resumed without re-executing already-completed pixels.

How it works
------------
Every pixel's raw post-selected counts are written to a JSON checkpoint
file as soon as its chunk finishes. On restart the checkpoint is loaded,
its metadata is validated against the current settings, and the already-
completed pixels are passed back to the QPU backend via `resume_state`
so only the missing pixels are submitted.

The checkpoint file (`--checkpoint`) is the single source of truth. Keep
it; delete it only to start fresh. When all interior pixels are present
the script assembles the full result maps and evaluates the same three
metrics used in demo.py (direction accuracy, cosine similarity,
MSE/PSNR) against the classical ground truth.

Usage
-----
    # first run (creates checkpoint, runs until done or interrupted)
    python run_qpu_experiment.py --image synthetic.tif --operator kirsch \
        --shots 1024 --backend-name ibm_fez --chunk-size 16

    # resume after an interruption: exactly the same command
    python run_qpu_experiment.py --image synthetic.tif --operator kirsch \
        --shots 1024 --backend-name ibm_fez --chunk-size 16

    # dry run on a simulator (no IBM account needed), to rehearse resume
    python run_qpu_experiment.py --image synthetic.tif --operator kirsch \
        --shots 1024 --simulate
"""
import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np
from compass_gradient import DIRECTIONS, compass_edges, get_operator
from compass_gradient.quantum import (
    neighbors_from_patch, run_qced_batch, QCEDShotResult, QPUUsageLimitError,
    _M)


# --------------------------------------------------------------------------
# checkpoint I/O
# --------------------------------------------------------------------------
def checkpoint_meta(args, n_interior):
    """Identity of a run; a checkpoint may only be reused if these match.
    `lam` (max |eigenvalue|) is stored so absolute edge magnitudes can be
    reconstructed from the raw counts when scoring MSE/PSNR."""
    r = get_operator(args.operator).eigenvalues()
    lam = float(np.max(np.abs(r)))
    return {
        "image": args.image, "operator": args.operator,
        "shots": args.shots, "remove_dc": args.remove_dc,
        "backend_name": args.backend_name, "n_interior": n_interior,
        "lam": lam,
    }


def load_checkpoint(path: Path, meta: dict):
    """Load a checkpoint, verifying its metadata matches `meta`.
    Returns {index -> record}. Missing file -> empty state."""
    if not path.exists():
        return {}
    with open(path) as f:
        data = json.load(f)
    old = data.get("meta", {})
    check_keys = [k for k in meta if k != "lam"]
    mismatch = {k: (old.get(k), meta[k]) for k in check_keys
                if old.get(k) != meta[k]}
    if mismatch:
        raise SystemExit(
            f"Checkpoint {path} was produced with different settings:\n"
            f"  {mismatch}\nUse a different --checkpoint or delete the file "
            f"to start fresh.")
    pixels = {int(k): v for k, v in data.get("pixels", {}).items()}
    print(f"resume: loaded {len(pixels)} completed pixels from {path}")
    return pixels


def atomic_write_json(path: Path, obj: dict):
    """Write JSON atomically (temp file + os.replace) so a crash mid-write
    cannot corrupt the checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def make_saver(path: Path, meta: dict, state: dict):
    """Return an on_chunk_done callback that merges each finished chunk
    into `state` and rewrites the checkpoint atomically."""
    def on_chunk_done(records):
        for rec in records:
            state[int(rec["index"])] = {
                "counts": rec["counts"],
                "valid_shots": rec["valid_shots"],
                "input_norm": rec["input_norm"],
            }
        atomic_write_json(path, {
            "meta": meta,
            "pixels": {str(k): v for k, v in state.items()},
        })
        done = len(state)
        print(f"  checkpoint saved: {done}/{meta['n_interior']} pixels "
              f"(+{len(records)} this chunk)")
    return on_chunk_done


# --------------------------------------------------------------------------
# device construction
# --------------------------------------------------------------------------
def build_device(args, n_wires=4):
    """Construct the PennyLane device. With --simulate, a local
    shot-based simulator stands in for hardware (for rehearsing the
    checkpoint/resume flow); otherwise a qiskit.remote IBM backend."""
    import pennylane as qml
    if args.simulate:
        print("simulate: using local default.qubit (no IBM account)")
        # shots are carried by each circuit tape (set in the qpu backend),
        # so the device itself needs none; omitting avoids a deprecation
        # warning about setting shots on the device.
        return qml.device("default.qubit", wires=n_wires)
    from qiskit_ibm_runtime import QiskitRuntimeService
    service = QiskitRuntimeService(channel="ibm_quantum_platform")
    backend = service.backend(args.backend_name)
    print(f"qpu: {args.backend_name} ready")
    return qml.device("qiskit.remote", wires=n_wires, backend=backend)


# --------------------------------------------------------------------------
# metrics (mirror demo.py, evaluated only when the run is complete)
# --------------------------------------------------------------------------
def evaluate(img, results, coords, operator):
    """Assemble quantum maps from per-pixel results and score them
    against the classical GT over ALL pixels (classical uses reflect
    padding at the border, and so does the quantum patch extraction)."""
    h, w = img.shape
    qvec = np.zeros((h, w, _M))
    qdir = np.full((h, w), -1, dtype=int)
    for (i, j), res in zip(coords, results):
        if res is None or res.valid_shots == 0:
            continue
        vec = res.edge_magnitudes
        qvec[i, j] = vec
        qdir[i, j] = res.direction

    edges = compass_edges(img, operator=operator)     # H x W, reflect border
    gt_dir = np.argmax(np.abs(edges), axis=0)

    dir_acc = float(np.mean(qdir == gt_dir) * 100)

    cV = np.abs(edges).reshape(_M, -1).T
    qV = qvec.reshape(-1, _M)
    dot = np.sum(qV * cV, axis=1)
    nq = np.linalg.norm(qV, axis=1)
    nc = np.linalg.norm(cV, axis=1)
    valid = (nq > 0) & (nc > 0)
    cos = np.zeros(len(cV))
    cos[valid] = dot[valid] / (nq[valid] * nc[valid])

    cM = np.abs(edges).max(axis=0).ravel()
    qM = qvec.max(axis=2).ravel()
    mse = float(np.mean((qM - cM) ** 2))
    psnr = 10.0 * np.log10(cM.max() ** 2 / mse) if mse > 0 else float("inf")

    return {
        "dir_acc_pct": dir_acc,
        "cos_sim_median": float(np.median(cos)),
        "mag_mse": mse, "mag_psnr": psnr,
        "qdir": qdir, "gt_dir": gt_dir,
    }


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", default="synthetic.tif")
    ap.add_argument("--operator", default="kirsch",
                    choices=["sobel", "prewitt", "kirsch"])
    ap.add_argument("--shots", type=int, default=1024)
    ap.add_argument("--remove-dc", action=argparse.BooleanOptionalAction,
                    default=True)
    ap.add_argument("--backend-name", default="ibm_fez",
                    help="IBM backend name (ignored with --simulate)")
    ap.add_argument("--chunk-size", type=int, default=16,
                    help="circuits submitted per batch = checkpoint granularity")
    ap.add_argument("--checkpoint", default=None,
                    help="explicit checkpoint JSON path; overrides --qpu-dir")
    ap.add_argument("--qpu-dir", default="qpu_runs",
                    help="folder that collects the checkpoint + result JSONs "
                         "(default: qpu_runs/); ignored if --checkpoint is set")
    ap.add_argument("--run-id", default=None,
                    help="tag for collecting multiple INDEPENDENT QPU "
                         "realisations of the same setting (image/operator/"
                         "shots). Each run-id gets its own checkpoint+result "
                         "JSON and resumes only itself, e.g. --run-id 1, "
                         "--run-id 2, ... Post-process each with demo.py "
                         "--backend qpu --qpu-run-id <id>.")
    ap.add_argument("--simulate", action="store_true",
                    help="use a local simulator instead of IBM hardware")
    args = ap.parse_args()

    # load image (must exist; generate with demo.py if needed)
    from PIL import Image
    img = np.asarray(Image.open(args.image).convert("L"), dtype=float)
    h, w = img.shape
    # process EVERY pixel; boundary pixels get a reflect-padded 3x3 patch,
    # matching the classical compass_edges border handling (mode="reflect")
    coords = [(i, j) for i in range(h) for j in range(w)]
    n_interior = len(coords)
    print(f"image: {args.image}  shape: {img.shape}  pixels: {n_interior} "
          f"(all, reflect-padded border)")

    meta = checkpoint_meta(args, n_interior)
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
    else:
        qpu_dir = Path(args.qpu_dir)
        qpu_dir.mkdir(parents=True, exist_ok=True)
        tag = f"_run{args.run_id}" if args.run_id not in (None, "") else ""
        ckpt_path = qpu_dir / (f"qpu_ckpt_{Path(args.image).stem}_"
                               f"{args.operator}_s{args.shots}{tag}.json")

    # resume state keyed by the *active-input index*, which for the QPU
    # backend is the position within the batch we submit below. We submit
    # exactly the interior pixels in `coords` order, so batch index ==
    # position in `coords`.
    state = load_checkpoint(ckpt_path, meta)

    remaining = n_interior - len(state)
    if remaining == 0:
        print("all interior pixels already in checkpoint; skipping QPU run")
    else:
        print(f"to run: {remaining} pixels "
              f"(chunk size {args.chunk_size}) on "
              f"{'simulator' if args.simulate else args.backend_name}")
        device = build_device(args)
        img_pad = np.pad(img, 1, mode="reflect")
        patches = [neighbors_from_patch(img_pad[i:i + 3, j:j + 3])
                   for (i, j) in coords]
        saver = make_saver(ckpt_path, meta, state)
        # run_qced_batch passes resume_state / on_chunk_done / chunk_size
        # through to the qpu backend via **backend_kwargs
        try:
            run_qced_batch(
                patches, args.operator, backend="qpu",
                remove_dc=args.remove_dc, shots=args.shots, device=device,
                chunk_size=args.chunk_size, resume_state=state,
                on_chunk_done=saver,
            )
        except QPUUsageLimitError as exc:
            # out of runtime (or a submission error): everything completed so
            # far is already saved by `saver`. Stop cleanly, no traceback.
            print(f"\n[stopped] {exc}")
            print(f"progress: {len(state)}/{n_interior} pixels checkpointed "
                  f"in {ckpt_path}")
            return
        print("QPU run finished (or all pixels completed)")

    # if complete, assemble results and evaluate
    if len(state) < n_interior:
        print(f"checkpoint has {len(state)}/{n_interior} pixels; "
              f"re-run the same command to continue.")
        return

    lam = float(meta["lam"])
    results = []
    for idx, (i, j) in enumerate(coords):
        rec = state[idx]
        results.append(QCEDShotResult(
            counts=np.asarray(rec["counts"], dtype=np.int64),
            shots=args.shots, valid_shots=int(rec["valid_shots"]),
            scale=lam, input_norm=float(rec["input_norm"])))
    # scale=lam (from meta) reconstructs absolute edge magnitudes:
    #   edge_magnitudes = lam * input_norm * sqrt(counts/shots)
    # so MSE/PSNR are on the same scale as the classical GT.

    m = evaluate(img, results, coords, args.operator)
    print("\n=== QPU result vs classical GT ===")
    print(f"  direction accuracy : {m['dir_acc_pct']:.1f} %")
    print(f"  cosine similarity  : {m['cos_sim_median']:.4f}")
    print(f"  magnitude MSE      : {m['mag_mse']:.3f}")
    print(f"  magnitude PSNR     : {m['mag_psnr']:.2f} dB")

    # confusion over interior pixels
    n_dir = 4 if args.operator in ("sobel", "prewitt") else 8
    cm = np.zeros((n_dir, n_dir), dtype=int)
    for (i, j) in coords:
        c, q = m["gt_dir"][i, j], m["qdir"][i, j]
        if q >= 0:
            cm[c % n_dir, q % n_dir] += 1
    print("\n  confusion (GT rows x QPU cols), all pixels:")
    labels = ([f"{DIRECTIONS[k]}/{DIRECTIONS[k+4]}" for k in range(4)]
              if n_dir == 4 else list(DIRECTIONS))
    print("     " + " ".join(f"{l:>5}" for l in labels))
    for rr in range(n_dir):
        print(f"  {labels[rr]:>4} " + " ".join(f"{cm[rr, cc]:5d}"
                                               for cc in range(n_dir)))

    out = ckpt_path.with_suffix(".result.json")
    atomic_write_json(out, {
        "meta": meta,
        "metrics": {k: m[k] for k in
                    ("dir_acc_pct", "cos_sim_median", "mag_mse", "mag_psnr")},
        "confusion": cm.tolist(), "labels": labels,
    })
    print(f"\nsaved metrics + confusion to {out}")


if __name__ == "__main__":
    main()