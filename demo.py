"""Demo and experiment sweep: classical compass edge detection (ground
truth) vs QCED quantum simulation.

Ground truth (GT) = the classical edge maps of the input image. The
quantum detector is evaluated against this GT with three metrics:
  (1) 8-vector error  : per-pixel L2 distance between the classical and
                        quantum 8-direction response vectors (pixel scale)
  (2) magnitude error : per-pixel |max_q - max_c| of the magnitude map
                        (pixel scale)
  (3) direction acc.  : per-pixel argmax agreement (%)
All three are reported over a shots sweep (median over pixels, mean +/-
std over seeds).

Outputs (under results_<image>_<operator>_<backend>_<detection>[_<threshold>]/):
  qced_sweep_dir_<tag>.png, qced_sweep_vecerr_<tag>.png,
  qced_sweep_magerr_<tag>.png, per-cell maps + confusion figures,
  qced_results_<image>_<backend>.xlsx (sheets: raw, summary)

Usage:
    python demo.py [--image test1.tif] [--operator kirsch]
                   [--backend shots|statevector] [--shots 1024]
                   [--no-cropping] [--crop 48] [--seed 0] [--no-remove-dc]
                   [--sweep 512,1024,2048,4096,8192] [--seeds 5] [--cbar]
"""
import argparse
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from compass_gradient import DIRECTIONS, compass_edges
from compass_gradient.quantum import (BACKENDS, neighbors_from_patch,
                                       run_qced_batch, QCEDShotResult)


# ring thickness (pixels of radius) for the synthetic step-ring image
# spatial frequency of the synthetic concentric-ring image (rings per
# unit radius); smaller = fewer, wider rings
_SYNTH_FREQ = 1.2


def make_synthetic(size: int = 24, freq: float = _SYNTH_FREQ):
    """Generate a synthetic concentric-ring image whose edge ground
    truth is known analytically. Intensity I = 128 + 100 sin(freq * r)
    has a radial gradient, so edges lie on the ring boundaries and all
    eight compass azimuths appear in near-equal proportion. The smooth
    sinusoid gives graded (non-trivial) edges. A small deterministic
    jitter diversifies pixel values (fixed seed => reproducible)."""
    c = (size - 1) / 2.0
    yy, xx = np.mgrid[0:size, 0:size]
    r = np.sqrt((xx - c) ** 2 + (yy - c) ** 2)
    img = 128.0 + 100.0 * np.sin(r * freq)
    #rng = np.random.default_rng(17)
    #img = np.clip(img + rng.uniform(-3, 3, (size, size)), 0, 255)
    img = np.clip(img,  0, 255)
    return img


def synthetic_gt_magnitude(size: int, freq: float = _SYNTH_FREQ):
    """Analytic edge-magnitude ground truth for the concentric-ring
    image. Intensity I = 128 + 100 sin(freq * r) has radial gradient with
    magnitude |dI/dr| = |100 * freq * cos(freq * r)|, so edges lie on the
    ring boundaries. `freq` must match make_synthetic. Only the magnitude
    (edge location) is used as GT; the per-pixel direction of a discrete
    3x3 compass operator does not match the continuous radial gradient
    near the center, so direction is evaluated against the classical
    result rather than this analytic GT."""
    c = (size - 1) / 2.0
    yy, xx = np.mgrid[0:size, 0:size]
    r = np.sqrt((xx - c) ** 2 + (yy - c) ** 2)
    return np.abs(100.0 * freq * np.cos(freq * r))


# ---------------------------------------------------------------------------
# Oriented-grating synthetic image (8 compass azimuths, analytic mag + dir GT)
# ---------------------------------------------------------------------------
# The image is a 3x3 tiling; eight tiles carry a straight parallel grating at
# one of the eight compass azimuths, the ninth is flat. Because the edges are
# straight (unlike the concentric ring's radial gradient near the centre), the
# per-pixel gradient direction of a discrete 3x3 compass matches the analytic
# azimuth, so BOTH magnitude and direction have an analytic ground truth.
#
# Edge sharpness is a single continuous knob passed through the --image string
# as  grating:<sharpness>  (0.0 = smooth sine ... 1.0 = near square wave). The
# intensity along the grating is  I = 128 + 100 * tanh(k sin(theta)) / tanh(k),
# which is a smooth sinusoid as k->0 and a square wave as k->inf. Sharper edges
# concentrate the gradient magnitude into thin bands (few pixels near a
# threshold), so the binary/Otsu threshold-stability effect is expected to
# weaken as sharpness rises -- this is the controlled smooth-vs-sharp knob.
_GRATING_AZIMUTHS_DEG = [0, 45, 90, 135, 180, 225, 270, 315]
_GRATING_CYCLES = 2.0          # full grating cycles across each tile's active
                               # width; frequency & guard scale with tile size
                               # so the pattern looks the same at any image size


def _sharpness_to_k(sharpness: float) -> float:
    """Map sharpness in [0,1] to the tanh steepness k. 0 -> ~sine (k~0.05),
    1 -> near square wave (k~8). Quadratic so low values stay smooth."""
    sh = min(max(float(sharpness), 0.0), 1.0)
    return 0.05 + 8.0 * sh ** 2


def _build_grating(size: int, sharpness: float):
    """Shared builder so the image and its GT can never drift apart. Returns
    (image, analytic_magnitude, analytic_azimuth_deg). azimuth is sign-aware
    (flips by 180 deg every half period) and NaN on flat/guard pixels."""
    az = np.deg2rad(_GRATING_AZIMUTHS_DEG)
    k = _sharpness_to_k(sharpness)
    tanhk = np.tanh(k)
    img = np.full((size, size), 128.0)
    mag = np.zeros((size, size))
    ang = np.full((size, size), np.nan)
    tile = size // 3
    # guard and grating frequency both scale with tile size, so every tile
    # shows the same number of cycles (_GRATING_CYCLES) at any image size; a
    # fixed pixel frequency would degenerate on the small crops actually run.
    g = max(1, int(round(tile * 0.12)))
    active = max(2, tile - 2 * g)
    w = 2.0 * np.pi * _GRATING_CYCLES / active      # angular freq (rad/px)
    for idx, p in enumerate(az):
        ry, rx = divmod(idx, 3)
        y0, y1 = ry * tile, ((ry + 1) * tile if ry < 2 else size)
        x0, x1 = rx * tile, ((rx + 1) * tile if rx < 2 else size)
        ay0, ay1, ax0, ax1 = y0 + g, y1 - g, x0 + g, x1 - g
        if ay1 <= ay0 or ax1 <= ax0:
            continue
        Y, X = np.mgrid[ay0:ay1, ax0:ax1]
        cx, cy = (ax0 + ax1 - 1) / 2.0, (ay0 + ay1 - 1) / 2.0
        proj = (X - cx) * np.cos(p) + (Y - cy) * np.sin(p)
        th = w * proj
        s = np.tanh(k * np.sin(th)) / tanhk
        img[ay0:ay1, ax0:ax1] = 128.0 + 100.0 * s
        # dI/dproj = 100 * w * f'(theta),
        # f'(theta) = k cos(theta) sech^2(k sin theta) / tanh(k)
        dIdp = (100.0 * w
                * (k * np.cos(th) * (1.0 / np.cosh(k * np.sin(th))) ** 2) / tanhk)
        mag[ay0:ay1, ax0:ax1] = np.abs(dIdp)
        ang[ay0:ay1, ax0:ax1] = np.where(
            dIdp >= 0, np.rad2deg(p), (np.rad2deg(p) + 180.0) % 360.0)
    return np.clip(img, 0, 255), mag, ang


def make_synthetic_grating(size: int = 24, sharpness: float = 0.0):
    """Oriented-grating synthetic image (see _build_grating)."""
    return _build_grating(size, sharpness)[0]


def grating_gt_magnitude(size: int, sharpness: float = 0.0):
    """Analytic edge-magnitude GT for the oriented-grating image."""
    return _build_grating(size, sharpness)[1]


def grating_gt_direction(size: int, sharpness: float = 0.0):
    """Analytic sign-aware gradient azimuth (degrees) for the grating image;
    NaN on flat/guard pixels. Not yet wired into the direction metric (that
    needs mapping these degrees onto the operator's DIRECTIONS index order);
    provided for the analytic direction-accuracy evaluation."""
    return _build_grating(size, sharpness)[2]


# ---- synthetic dispatch helpers (rings 'synthetic' or 'grating:<sharpness>') --
def parse_synthetic_spec(path: str):
    """Return (kind, sharpness) for a synthetic --image spec, else None.
    'synthetic' -> ('rings', None); 'grating' / 'grating:0.7' -> ('grating', sh)."""
    if path == "synthetic":
        return ("rings", None)
    if path == "grating" or path.startswith("grating:"):
        sh = 0.0
        if ":" in path:
            try:
                sh = float(path.split(":", 1)[1])
            except ValueError:
                sh = 0.0
        return ("grating", sh)
    return None


def is_synthetic(src: str) -> bool:
    """True if src is one of the analytic-GT synthetic images."""
    return src == "synthetic" or src.startswith("grating")


def _sh_tag(sharpness: float) -> str:
    return str(sharpness).replace(".", "p").replace("-", "m")


def synthetic_stem(src: str) -> str:
    """Filesystem-safe stem for output folders/xlsx, unique per sharpness."""
    if src == "synthetic":
        return "synthetic"
    if src.startswith("grating"):
        sh = src.split(":", 1)[1] if ":" in src else "0.0"
        return "grating_s" + _sh_tag(sh)
    return Path(src).stem


def analytic_gt_magnitude(src: str, size: int):
    """Dispatch to the correct analytic magnitude GT for the synthetic in use."""
    if src.startswith("grating"):
        sh = float(src.split(":", 1)[1]) if ":" in src else 0.0
        return grating_gt_magnitude(size, sh)
    return synthetic_gt_magnitude(size)


def add_noise_at_snr(img, snr_db, seed=0):
    """Add zero-mean Gaussian noise at a target signal-to-noise ratio
    `snr_db` (in dB) and clip to [0, 255]. The signal power is the image
    variance, so the noise std is  std(img) / 10**(snr_db/20). A value of
    +inf means no noise (clean image). Deterministic given `seed`."""
    if np.isinf(snr_db):
        return img.copy()
    signal_std = float(np.std(img))
    noise_std = signal_std / (10.0 ** (snr_db / 20.0))
    if noise_std <= 0:
        return img.copy()
    rng = np.random.default_rng(seed)
    return np.clip(img + rng.normal(0.0, noise_std, img.shape), 0, 255)


def plot_noise_lines(df, metric, ylabel, outdir, tag, logy=False):
    """For each shot count, plot the metric vs input SNR for BOTH the
    classical detector and the quantum detector, so their trends under
    increasing noise can be compared directly (robustness). One figure
    per shot count. x = input SNR (dB, higher = cleaner), two curves
    (classical is shot-independent but drawn per figure for reference).
    Only magnitude metrics (MSE/PSNR vs the analytic GT) are meaningful
    here; direction/cosine use the classical result as reference and are
    excluded from the noise sweep."""
    shots_vals = sorted(df["shots"].unique())

    def snr_axis(vals):
        # map inf -> a finite tick to the right of the max finite SNR
        finite = [v for v in vals if np.isfinite(v)]
        hi = (max(finite) if finite else 0) + 10
        return [hi if np.isinf(v) else v for v in vals], hi

    for s in shots_vals:
        sub = df[df["shots"] == s]
        snrs = sorted(sub["noise_snr"].unique(),
                      key=lambda v: (np.isinf(v), v))
        x, inf_pos = snr_axis(snrs)
        qy, qerr, cy = [], [], []
        for nz in snrs:
            g = sub[sub["noise_snr"] == nz]
            qy.append(g[metric].mean())
            qerr.append(g[metric].std())
            cy.append(g[f"classical_{metric}"].mean())
        order = np.argsort(x)
        x = np.array(x)[order]
        qy = np.array(qy)[order]; qerr = np.array(qerr)[order]
        cy = np.array(cy)[order]

        fig, ax = plt.subplots(figsize=(5.5, 4))
        ax.errorbar(x, qy, yerr=qerr, marker="o", capsize=4, lw=2.2,
                    color="C0", label="quantum")
        ax.plot(x, cy, marker="s", lw=2.2, ls="--", color="crimson",
                label="classical")
        ax.set_xlabel("Input SNR (dB)  —  higher = cleaner",
                      fontweight="bold")
        ax.set_ylabel(ylabel, fontweight="bold")
        if logy:
            ax.set_yscale("log")
        # relabel the inf tick as 'clean'
        ticks = [t for t in x]
        labels = ["clean" if np.isclose(t, inf_pos) else f"{t:g}"
                  for t in ticks]
        ax.set_xticks(ticks, labels)
        ax.set_title(f"{ylabel} vs input noise  (shots = {s})",
                     fontsize=12, fontweight="bold")
        ax.grid(True, ls="--", alpha=0.5)
        ax.legend(fontsize=11)
        plt.tight_layout()
        f = outdir / f"noise_{metric}_s{s}_{tag}.png"
        fig.savefig(f, dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(f"saved: {f}")


def load_image(path: str, cropping: bool, crop: int) -> "tuple[np.ndarray, str]":
    """Load a grayscale image from `path` (optionally cropped). The
    special name 'synthetic' (or a missing file) generates the
    concentric-ring image with known edge GT and saves it to
    synthetic.tif."""
    spec = parse_synthetic_spec(path)
    if spec is not None:
        from PIL import Image
        kind, sh = spec
        size = crop if cropping else 24
        if kind == "grating":
            if size < 48:
                print(f"WARNING: grating size {size} is small; each of the 8 "
                      f"tiles is ~{size // 3}px and degenerates. Use "
                      f"--cropping --crop 64 (or larger) for clean gratings.")
            img = make_synthetic_grating(size, sh)
            src = f"grating:{sh}"
            fname = f"grating_s{_sh_tag(sh)}.tif"
            Image.fromarray(img.astype(np.uint8)).save(fname)
            print(f"synthetic {size}x{size} oriented-grating image "
                  f"(sharpness={sh}, analytic mag+dir GT) saved to {fname}")
            return img, src
        img = make_synthetic(size)
        Image.fromarray(img.astype(np.uint8)).save("synthetic.tif")
        print(f"synthetic {size}x{size} concentric-ring image "
              f"(known edge GT) saved to synthetic.tif")
        return img, "synthetic"
    try:
        from PIL import Image
        img = np.asarray(Image.open(path).convert("L"), dtype=float)
        if cropping:
            img = img[:crop, :crop]
        return img, path
    except FileNotFoundError:
        from PIL import Image
        size = crop if cropping else 24
        img = make_synthetic(size)
        Image.fromarray(img.astype(np.uint8)).save("synthetic.tif")
        print(f"synthetic {size}x{size} concentric-ring image "
              f"(known edge GT) saved to synthetic.tif")
        return img, "synthetic"


def _otsu_threshold(values: np.ndarray, nbins: int = 256) -> float:
    """Otsu's method: pick the threshold that maximizes between-class
    variance of the magnitude histogram. Pure-numpy implementation,
    matching skimage.filters.threshold_otsu."""
    v = values[np.isfinite(values)]
    if v.size == 0 or v.max() <= v.min():
        return float(v.max()) if v.size else 0.0
    hist, edges = np.histogram(v, bins=nbins)
    centers = (edges[:-1] + edges[1:]) / 2.0
    total = hist.sum()
    w0 = np.cumsum(hist)                       # background cumulative count
    w1 = total - w0                            # foreground cumulative count
    sum0 = np.cumsum(hist * centers)           # background cumulative sum
    sum_total = sum0[-1]
    mean0 = np.divide(sum0, w0, out=np.zeros_like(sum0), where=w0 > 0)
    mean1 = np.divide(sum_total - sum0, w1,
                      out=np.zeros_like(sum0), where=w1 > 0)
    between = w0.astype(float) * w1.astype(float) * (mean0 - mean1) ** 2
    between[w1 <= 0] = -1.0  

    '''import matplotlib.pyplot as plt
    tstar = centers[np.argmax(between)]
    plt.figure(figsize=(8,4))
    plt.bar(centers, hist, width=(centers[1]-centers[0]), color='gray')
    plt.axvline(tstar, color='crimson', ls='--', label=f'T*={tstar:.1f}')
    plt.xlabel('magnitude'); plt.ylabel('count'); plt.legend()
    plt.savefig('otsu_hist.png', dpi=150, bbox_inches='tight'); plt.close()'''


                       # ignore empty-foreground bins
    return float(centers[np.argmax(between)])


def edge_threshold(mag_map: np.ndarray, method: str, alpha: float) -> float:
    """Return a scalar edge threshold T for a magnitude map.
    method='otsu': Otsu's automatic threshold (data-driven, reproducible).
    method='fixed': a fixed fraction alpha of the map's maximum,
    T = alpha * max(mag) — the simple thresholding used by related
    quantum edge-detection works (edge iff G >= T)."""
    if method == "otsu":
        return _otsu_threshold(mag_map.ravel())
    return float(alpha) * float(np.max(mag_map))


def binary_edge_mask(mag_map: np.ndarray, thr: float) -> np.ndarray:
    """Binary edge map: 1 where magnitude >= threshold, else 0 (the
    standard magnitude-thresholding rule used across the literature)."""
    return (mag_map >= thr).astype(np.int32)


def classical_maps(img: np.ndarray, operator: str):
    """Compute the classical 8-direction response stack, the per-pixel
    magnitude map max|e(m)|, and the direction map argmax|e(m)|."""
    edges = compass_edges(img, operator=operator)
    mag = np.max(np.abs(edges), axis=0)
    direction = np.argmax(np.abs(edges), axis=0)
    return edges, mag, direction


def binary_classification_metrics(pred_mask, gt_mask, inner=None):
    """Accuracy (%) and F1 score of a predicted binary edge map against a
    binary GT edge map (values in {0,1}), treating 1 (edge) as the
    positive class. If `inner` is given, only that region is scored
    (boundary pixels are excluded because they depend on reflect
    padding). F1 is nan when there are no positives in either map."""
    if inner is not None:
        pred_mask = pred_mask[inner]
        gt_mask = gt_mask[inner]
    p = pred_mask.astype(int).ravel()
    g = gt_mask.astype(int).ravel()
    if p.size == 0:
        return float("nan"), float("nan")
    accuracy = float(np.mean(p == g)) * 100.0
    tp = int(np.sum((p == 1) & (g == 1)))
    fp = int(np.sum((p == 1) & (g == 0)))
    fn = int(np.sum((p == 0) & (g == 1)))
    denom = 2 * tp + fp + fn
    f1 = float(2 * tp / denom) if denom > 0 else float("nan")
    return accuracy, f1


def _nms_offsets():
    """(dr, dc) neighbor offset along each compass direction's gradient
    axis, in image coordinates (row increases downward). Direction m has
    gradient angle 90 + 45*m degrees (operators.py convention)."""
    offs = {}
    for m in range(8):
        ang = np.radians(90 + 45 * m)
        offs[m] = (-int(round(np.sin(ang))), int(round(np.cos(ang))))
    return offs


def non_maximum_suppression(mag, direction):
    """Thin edges by non-maximum suppression: keep each pixel's magnitude
    only if it is >= its two neighbors along the pixel's gradient
    direction, otherwise set it to 0. No hyperparameters. `direction` is
    the per-pixel 8-compass index (negative = undefined -> suppressed)."""
    h, w = mag.shape
    out = np.zeros_like(mag)
    offs = _nms_offsets()
    for i in range(h):
        True
        for j in range(w):
            d = int(direction[i, j])
            if d < 0:
                continue
            dr, dc = offs[d]
            m0 = mag[i, j]
            n1 = mag[i + dr, j + dc] if 0 <= i + dr < h and 0 <= j + dc < w else -1.0
            n2 = mag[i - dr, j - dc] if 0 <= i - dr < h and 0 <= j - dc < w else -1.0
            if m0 >= n1 and m0 >= n2:
                out[i, j] = m0
    return out


def quantum_maps(img: np.ndarray, operator: str, backend: str,
                 shots: int, seed: int, remove_dc: bool,
                 chunk: int = 512, progress: bool = True):
    """Run QCED for all inner pixels via chunked batch submissions;
    returns the magnitude map, direction map, per-pixel 8-vector stack,
    success probability map, and runtime."""
    h, w = img.shape
    # process EVERY pixel; boundary pixels get a reflect-padded 3x3 patch,
    # matching the classical compass_edges border handling (mode="reflect")
    coords = [(i, j) for i in range(h) for j in range(w)]
    img_pad = np.pad(img, 1, mode="reflect")
    vecs = [neighbors_from_patch(img_pad[i : i + 3, j : j + 3])
            for i, j in coords]
    qmag = np.zeros((h, w))
    qdir = np.full((h, w), -1, dtype=int)
    qvec = np.zeros((h, w, 8))
    rate_map = np.zeros((h, w))
    t0 = time.perf_counter()
    starts = range(0, len(coords), chunk)
    it = tqdm(starts, desc=f"QCED {backend} s{shots}", unit="chunk",
              leave=False) if progress else starts
    for c0 in it:
        batch = vecs[c0 : c0 + chunk]
        results = run_qced_batch(
            batch, operator, backend=backend, remove_dc=remove_dc,
            shots=shots, seed=(seed + c0 if seed is not None else None),
        )
        for (i, j), res in zip(coords[c0 : c0 + chunk], results):
            rate_map[i, j] = res.success_probability
            if backend != "statevector":
                if res.valid_shots == 0:
                    continue
                vec = res.edge_magnitudes
            else:
                vec = np.abs(res.edge_values)
            qvec[i, j] = vec
            qmag[i, j] = vec.max()
            qdir[i, j] = res.direction
    return qmag, qdir, qvec, rate_map, time.perf_counter() - t0


# --- quantum-result cache -------------------------------------------------
# quantum_maps() is the ONLY part that touches the backend/QPU; its output
# (qmag, qdir, qvec, rate_map) is detection-INDEPENDENT -- binary, response
# and maximum are all classical post-processing of the same maps. Caching the
# maps keyed by everything that determines them lets a second run (e.g. the
# 'response' pass after the 'binary' pass) REUSE the first run's counts
# instead of re-submitting the circuits, so a QPU job is paid for only once.
# The detection mode is deliberately NOT part of the key.
_QCACHE_VERSION = "v1"   # bump if quantum_maps / noise defaults change


def _quantum_cache_path(cache_dir, img, operator, backend, shots, seed,
                        remove_dc):
    """Deterministic cache file for one quantum_maps() result. The key hashes
    the exact input image (so a different image, crop, or noise realisation
    misses) plus operator/backend/shots/seed/remove_dc."""
    h = hashlib.sha1()
    h.update(np.ascontiguousarray(img, dtype=np.float64).tobytes())
    meta = (f"{operator}|{backend}|{shots}|{seed}|{int(bool(remove_dc))}"
            f"|{_QCACHE_VERSION}")
    h.update(meta.encode())
    name = f"qmaps_{backend}_s{shots}_seed{seed}_{h.hexdigest()[:16]}.npz"
    return Path(cache_dir) / name


def quantum_maps_cached(img, operator, backend, shots, seed, remove_dc,
                        cache_dir=None, use_cache=True, **kw):
    """quantum_maps() wrapped with an on-disk cache. On a hit the backend is
    NOT run. The cache is keyed by the image content and run parameters, not
    the detection mode, so binary/response/maximum share one backend/QPU run.
    Assumes the default noise model for the 'noisy' backend (bump
    _QCACHE_VERSION or use --no-quantum-cache if you change it)."""
    if not use_cache or not cache_dir:
        return quantum_maps(img, operator, backend, shots, seed,
                            remove_dc, **kw)
    path = _quantum_cache_path(cache_dir, img, operator, backend, shots,
                               seed, remove_dc)
    if path.exists():
        try:
            d = np.load(path)
            print(f"    [cache] reuse quantum maps ({path.name}) "
                  f"-- backend NOT run")
            return (d["qmag"], d["qdir"], d["qvec"], d["rate_map"],
                    float(d["dt"]))
        except Exception as exc:
            print(f"    [cache] unreadable ({exc}); recomputing")
    qmag, qdir, qvec, rate_map, dt = quantum_maps(
        img, operator, backend, shots, seed, remove_dc, **kw)
    try:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, qmag=qmag, qdir=qdir, qvec=qvec,
                            rate_map=rate_map, dt=dt)
        print(f"    [cache] saved quantum maps ({path.name})")
    except Exception as exc:
        print(f"    [cache] save failed ({exc}); continuing")
    return qmag, qdir, qvec, rate_map, dt


def quantum_maps_from_checkpoint(img, operator, shots, remove_dc,
                                 json_path=None, qpu_dir="qpu_runs",
                                 image_stem="synthetic", run_id=None):
    """Build the same (qmag, qdir, qvec, rate_map, dt) maps as quantum_maps()
    but from the raw counts saved by run_qpu_experiment.py -- so the QPU is
    NOT re-run. The counts are detection-independent, so every detection
    (binary/response/maximum) reads the same checkpoint. The checkpoint's
    pixel index order matches quantum_maps' coords (row-major, all pixels)."""
    h, w = img.shape
    if json_path:
        path = Path(json_path)
    else:
        tag = f"_run{run_id}" if run_id not in (None, "") else ""
        path = Path(qpu_dir) / (f"qpu_ckpt_{image_stem}_{operator}"
                                f"_s{shots}{tag}.json")
    if not path.exists():
        raise FileNotFoundError(
            f"QPU checkpoint not found: {path}\n"
            f"Run run_qpu_experiment.py for (image={image_stem}, "
            f"operator={operator}, shots={shots}) first, or pass "
            f"--qpu-json <path>.")
    with open(path) as f:
        data = json.load(f)
    meta = data.get("meta", {})
    pixels = data.get("pixels", {})

    # validate the checkpoint matches this run (image/operator/shots/size)
    problems = []
    if Path(str(meta.get("image", ""))).stem != image_stem:
        problems.append(f"image {meta.get('image')!r} != {image_stem!r}")
    if str(meta.get("operator")) != operator:
        problems.append(f"operator {meta.get('operator')!r} != {operator!r}")
    if int(meta.get("shots", -1)) != int(shots):
        problems.append(f"shots {meta.get('shots')} != {shots}")
    if bool(meta.get("remove_dc")) != bool(remove_dc):
        problems.append(f"remove_dc {meta.get('remove_dc')} != {remove_dc}")
    if int(meta.get("n_interior", -1)) != h * w:
        problems.append(f"n_interior {meta.get('n_interior')} != {h * w} "
                        f"(image size {h}x{w} mismatch)")
    if problems:
        raise ValueError(f"QPU checkpoint {path} does not match this run: "
                         + "; ".join(problems))
    if len(pixels) < h * w:
        raise ValueError(f"QPU checkpoint {path} is incomplete "
                         f"({len(pixels)}/{h * w} pixels). Finish the "
                         f"run_qpu_experiment.py run first.")

    lam = float(meta["lam"])
    coords = [(i, j) for i in range(h) for j in range(w)]
    qmag = np.zeros((h, w))
    qdir = np.full((h, w), -1, dtype=int)
    qvec = np.zeros((h, w, 8))
    rate_map = np.zeros((h, w))
    for idx, (i, j) in enumerate(coords):
        rec = pixels[str(idx)]
        res = QCEDShotResult(
            counts=np.asarray(rec["counts"], dtype=np.int64), shots=int(shots),
            valid_shots=int(rec["valid_shots"]), scale=lam,
            input_norm=float(rec["input_norm"]))
        rate_map[i, j] = res.success_probability
        if res.valid_shots == 0:
            continue
        vec = res.edge_magnitudes
        qvec[i, j] = vec
        qmag[i, j] = vec.max()
        qdir[i, j] = res.direction
    print(f"    [qpu] loaded counts from {path.name} "
          f"({len(pixels)} pixels) -- no QPU submission")
    return qmag, qdir, qvec, rate_map, 0.0


def dir_accuracy(test_dir, gt_dir, inner, mask=None):
    """(3) Direction agreement (%) of the quantum direction map against
    the GT (classical) direction map. Over all inner pixels by default;
    if `mask` (inner-shaped bool) is given, only over masked (detected-
    edge) pixels."""
    td = test_dir[inner]; gd = gt_dir[inner]
    if mask is not None:
        td = td[mask]; gd = gd[mask]
    if td.size == 0:
        return float("nan")
    return float(np.mean(td == gd) * 100)


def cosine_similarity(gt_edges, vec_map, inner, mask=None):
    """(1) Per-pixel cosine similarity between the GT classical and
    quantum 8-direction response vectors. Bounded in [0, 1] (1 =
    identical direction distribution, 0 = orthogonal). Dimensionless.
    Returns one value per inner pixel; degenerate pixels (zero vector)
    yield 0."""
    # restrict to the interior region (inner) before flattening, so the
    # per-pixel vectors match the mask built on the same region
    cV = np.abs(gt_edges[:, inner[0], inner[1]]).reshape(8, -1).T   # (P, 8)
    qV = vec_map[inner].reshape(-1, 8)                              # (P, 8)
    if mask is not None:
        keep = mask.ravel()
        cV = cV[keep]; qV = qV[keep]
    dot = np.sum(qV * cV, axis=1)
    nc = np.linalg.norm(cV, axis=1)
    nq = np.linalg.norm(qV, axis=1)
    valid = (nc > 0) & (nq > 0)
    out = np.zeros(len(cV))
    out[valid] = dot[valid] / (nc[valid] * nq[valid])
    return out


def _rescale_0_255(a):
    """Linearly rescale an array to the [0, 255] range (min->0, max->255).
    A flat array maps to zeros."""
    lo, hi = float(np.min(a)), float(np.max(a))
    if hi <= lo:
        return np.zeros_like(a)
    return (a - lo) / (hi - lo) * 255.0


def mag_mse_psnr(gt_edges, vec_map, inner, max_i=None, mask=None,
                 mag_norm=False, gt_mag_map=None):
    """(2) MSE and PSNR of the quantum magnitude map against a reference
    magnitude map, following [18] eqs. (17)-(18):
        MSE  = mean_(i,j) (I(i,j) - G(i,j))^2
        PSNR = 10 * log10(MAX_I^2 / MSE)

    Reference I:
      - default: the classical compass magnitude map (from gt_edges);
      - if `gt_mag_map` is given (synthetic analytic GT), that map is the
        reference instead.
    G is the quantum magnitude map.

    mag_norm: if True, both I and G are independently rescaled to [0,255]
    before the comparison (MAX_I = 255). This removes the operator's gain
    so the quantum/classical maps can be compared against the analytic GT
    (whose raw scale differs by ~20x). If False, raw magnitudes are used
    and MAX_I is the reference peak (matching [17,18] absolute MSE)."""
    if gt_mag_map is not None:
        cM = gt_mag_map[inner].ravel()                    # analytic GT mag
    else:
        cM = np.abs(gt_edges[:, inner[0], inner[1]]).max(axis=0).ravel()
    qM = vec_map[inner].max(axis=2).ravel()               # quantum mag
    if mask is not None:
        keep = mask.ravel()
        cM = cM[keep]; qM = qM[keep]
    if cM.size == 0:
        return float("nan"), float("nan")
    if mag_norm:
        cM = _rescale_0_255(cM)
        qM = _rescale_0_255(qM)
        max_i = 255.0
    mse = float(np.mean((qM - cM) ** 2))
    if max_i is None:
        max_i = float(cM.max())
    psnr = (10.0 * np.log10(max_i ** 2 / max(mse, 1e-10))
            if max_i > 0 else 0.0)
    return mse, psnr


def confusion_matrix(gt_dir, test_dir, inner, n_dir, mask=None):
    """Row-normalized confusion matrix over inner pixels. n_dir == 8 uses
    the full compass directions; n_dir == 4 folds opposite directions
    (index mod 4) for the sign-degenerate Sobel/Prewitt operators.
    Unresolved pixels (direction < 0) are dropped."""
    cm = np.zeros((n_dir, n_dir))
    cd = gt_dir[inner].ravel()
    qd = test_dir[inner].ravel()
    if mask is not None:
        keep = mask.ravel()
        cd = cd[keep]; qd = qd[keep]
    for c, q in zip(cd, qd):
        if q < 0:
            continue
        cm[c % n_dir, q % n_dir] += 1
    rowsum = cm.sum(axis=1, keepdims=True)
    rowsum[rowsum == 0] = 1
    return cm / rowsum * 100


def statevector_l2_reference(img, operator, remove_dc, inner, n_sample=100):
    """Implementation-fidelity reference: normalized L2 error of the
    statevector backend on a sample of inner pixels."""
    h, w = img.shape
    coords = [(i, j) for i in range(h) for j in range(w)]
    img_pad = np.pad(img, 1, mode="reflect")
    rng = np.random.default_rng(0)
    sel = rng.choice(len(coords), size=min(n_sample, len(coords)),
                     replace=False)
    nbs = [neighbors_from_patch(img_pad[i : i + 3, j : j + 3])
           for i, j in (coords[s] for s in sel)]
    results = run_qced_batch(nbs, operator, backend="statevector",
                             remove_dc=remove_dc)
    errs = []
    for res in results:
        c = np.abs(res.edge_values)
        q = np.abs(res.amplitudes) * res.scale * res.input_norm
        if c.max() > 0:
            errs.append(np.linalg.norm(c / np.linalg.norm(c)
                                       - q / np.linalg.norm(q)))
    return float(np.median(errs)) if errs else float("nan")


def append_xlsx(path: Path, sheet_rows: "dict[str, list[dict]]"):
    """Append rows to sheets of an xlsx results file, creating the file
    or sheets as needed; falls back to CSV on failure."""
    try:
        sheets = pd.read_excel(path, sheet_name=None) if path.exists() else {}
        with pd.ExcelWriter(path, engine="openpyxl") as w:
            for name in set(sheets) | set(sheet_rows):
                parts = []
                if name in sheets:
                    parts.append(sheets[name])
                if sheet_rows.get(name):
                    parts.append(pd.DataFrame(sheet_rows[name]))
                pd.concat(parts, ignore_index=True).to_excel(
                    w, sheet_name=name, index=False)
    except Exception as exc:
        for name, rows in sheet_rows.items():
            if rows:
                backup = path.with_name(f"{path.stem}_{name}_backup.csv")
                pd.DataFrame(rows).to_csv(
                    backup, mode="a", index=False,
                    header=not backup.exists())
                print(f"xlsx failed ({exc}); rows saved to {backup}")


def load_done_points(xlsx_path):
    """Return {(noise_snr, shots, seed): record_dict} already recorded in
    the results xlsx 'raw' sheet, for --skip-existing. The stored records
    let skipped points still contribute to the summary and sweep plots.
    Missing file -> empty dict."""
    from pathlib import Path as _P
    if not _P(xlsx_path).exists():
        return {}
    try:
        raw = pd.read_excel(xlsx_path, sheet_name="raw")
    except Exception:
        return {}
    done = {}
    for _, r in raw.iterrows():
        nz = r.get("noise_snr", float("inf"))
        try:
            nz = float(nz)
        except (TypeError, ValueError):
            nz = float("inf")
        s = r.get("shots")
        sd = r.get("seed")
        if pd.notna(s) and pd.notna(sd):
            done[(nz, int(s), int(sd))] = r.to_dict()
    return done


def plot_magnitude_hist(qmag, cmag, inner, args, shots, outdir, tag,
                        gt_mag_map=None):
    """Save a histogram of the per-pixel edge magnitudes (interior),
    overlaying quantum and classical (and the analytic GT when given),
    with each map's Otsu threshold marked. Reveals how the backend
    reshapes the magnitude distribution (e.g. hardware noise flattening
    it), which drives the binary threshold decision. Saved under an
    outdir/histogram/ subfolder."""
    q = qmag[inner].ravel()
    c = cmag[inner].ravel()
    fig, ax = plt.subplots(figsize=(6, 4))
    bins = 40
    ax.hist(c, bins=bins, alpha=0.5, color="crimson", label="classical",
            density=True)
    ax.hist(q, bins=bins, alpha=0.5, color="C0", label="quantum",
            density=True)
    tq = edge_threshold(qmag, args.threshold, args.threshold_alpha)
    tc = edge_threshold(cmag, args.threshold, args.threshold_alpha)
    ax.axvline(tq, color="C0", ls="--", lw=1.6,
               label=f"quantum T*={tq:.0f}")
    ax.axvline(tc, color="crimson", ls="--", lw=1.6,
               label=f"classical T*={tc:.0f}")
    if gt_mag_map is not None:
        gt = gt_mag_map[inner].ravel()
        ax.hist(gt, bins=bins, alpha=0.35, color="green", label="GT",
                density=True)
    ax.set_xlabel("edge magnitude", fontweight="bold")
    # fit the x-axis to the data (magnitude scale differs by image/operator;
    # a fixed upper bound squished or clipped non-ring images)
    _vals = [v for v in (q, c) if v.size]
    if gt_mag_map is not None:
        _g = gt_mag_map[inner].ravel()
        if _g.size:
            _vals.append(_g)
    _xhi = max((float(np.nanmax(v)) for v in _vals if v.size), default=1.0)
    # ax.set_xlim(0, 1.05 * _xhi if np.isfinite(_xhi) and _xhi > 0 else 1.0)
    ax.set_xlim(0, 3500)
    ax.set_ylabel("density", fontweight="bold")
    ax.set_title(f"Magnitude histogram ({args.backend}, shots={shots})",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    hist_dir = outdir / "histogram"
    hist_dir.mkdir(exist_ok=True)
    f = hist_dir / f"H_maghist_{tag}.png"
    fig.savefig(f, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {f}")


def plot_maps(img, cmag, qmag, inner, args, shots, outdir, tag, thr=None,
              save_static=True, upscale=10, gt_binary=None,
              gt_mag_map=None, mag_norm=False, cbar=False):
    """Save the input, classical (GT), quantum, and error maps as
    individual tif files. In binary mode (thr given), also save the
    thresholded binary edge masks for the classical GT and the quantum
    result (edge=black=1, non-edge=white=0), using the same threshold
    for a fair comparison. save_static: whether to save A, B, Bb (these
    don't depend on shots/seed). upscale: magnification factor for small
    images (saves original + upscaled versions). cbar: additionally
    save a colorbar'd PNG (…_cbar.png) next to every A/B/C/D(/Gm) map
    (native, upscaled, and interior-only versions)."""
    from scipy.ndimage import zoom

    def save_cbar(data, order, name, cmap, suffix="", vmin=None, vmax=None):
        """Save the map as a PNG with a colorbar (…{suffix}_cbar.png),
        since a plain .tif has no room for one."""
        fig, ax = plt.subplots(figsize=(4, 4))
        # im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
        im = ax.imshow(data, cmap=cmap, vmin=0, vmax=2800)
        ax.axis("off")
        fig.colorbar(im, fraction=0.046, pad=0.04)
        fig.savefig(outdir / f"{order}_{name}_{tag}{suffix}_cbar.png",
                    dpi=200, bbox_inches="tight")
        plt.close(fig)

    def save_img(data, order, name, cmap, vmin=None, vmax=None, with_cbar=False):
        """Save image at native size and, when upscaling, both a full
        upscaled version and a boundary-excluded (interior-only) upscaled
        version (…_inner) since border pixels rely on reflect padding.
        with_cbar: also save a colorbar'd PNG of every saved version."""
        kw = dict(cmap=cmap, format="tiff")
        if vmin is not None:
            kw.update(vmin=vmin, vmax=vmax)
        plt.imsave(outdir / f"{order}_{name}_{tag}.tif", data, **kw)
        if with_cbar:
            save_cbar(data, order, name, cmap, "", vmin=vmin, vmax=vmax)
        if upscale > 1:
            data_up = zoom(data, upscale, order=0)
            plt.imsave(outdir / f"{order}_{name}_{tag}_x{upscale}.tif",
                       data_up, **kw)
            # interior-only (drop the 1-pixel border, then upscale)
            inner_up = zoom(data[1:-1, 1:-1], upscale, order=0)
            plt.imsave(outdir / f"{order}_{name}_{tag}_x{upscale}_inner.tif",
                       inner_up, **kw)
            if with_cbar:
                save_cbar(data_up, order, name, cmap, f"_x{upscale}",
                          vmin=vmin, vmax=vmax)
                save_cbar(inner_up, order, name, cmap, f"_x{upscale}_inner",
                          vmin=vmin, vmax=vmax)

    # in mag-norm mode, rescale each magnitude map independently to
    # [0,255] before drawing, so the colorbars read 0-255 and match the
    # scale used by the MSE/PSNR metric (each map by its own max).
    cmag_d = _rescale_0_255(cmag) if mag_norm else cmag
    qmag_d = _rescale_0_255(qmag) if mag_norm else qmag
    vlo, vhi = (0.0, 255.0) if mag_norm else (None, None)

    diff = np.zeros_like(img)
    diff[inner] = np.abs(qmag_d[inner] - cmag_d[inner])
    # magnitude maps use reversed grayscale so that edges (high
    # magnitude) render dark on a white background (edge=black,
    # non-edge=white), matching the convention of printed edge maps.
    panels = [(img, "A", "input", "gray_r", save_static, None, None),
              (cmag_d, "B", "classical", "gray_r", save_static, vlo, vhi),
              (qmag_d, "C", "quantum", "gray_r", True, vlo, vhi),
              (diff, "D", "error", "magma", True, None, None)]
    for im, order, name, cmap_, should_save, vmn, vmx in panels:
        if should_save:
            save_img(im, order, name, cmap_, vmin=vmn, vmax=vmx,
                     with_cbar=cbar)

    # analytic ground-truth MAGNITUDE map (synthetic + mag-norm mode):
    # rescaled to [0,255] the same way, so its colorbar matches.
    if gt_mag_map is not None and save_static:
        gtm_d = _rescale_0_255(gt_mag_map) if mag_norm else gt_mag_map
        save_img(gtm_d, "Gm", "gt_magnitude", "gray_r",
                 vmin=vlo, vmax=vhi, with_cbar=cbar)

    # binary edge masks (binary detection mode only)
    if thr is not None:
        def save_binary(mask01, order, name, static):
            if not static:
                return
            save_img(mask01, order, name, "gray_r", vmin=0, vmax=1)

        # each map is thresholded on its OWN magnitude (classical and
        # quantum have different scales; a shared threshold would
        # misclassify pixels). Mirrors the metric computation.
        c_thr = edge_threshold(cmag, args.threshold, args.threshold_alpha)
        q_thr = edge_threshold(qmag, args.threshold, args.threshold_alpha)
        c_mask = (cmag >= c_thr).astype(float)   # classical edge map
        q_mask = (qmag >= q_thr).astype(float)   # quantum edge map
        # gray_r: 1 (edge) -> black, 0 (non-edge) -> white
        save_binary(c_mask, "Bb", "classical_binary", save_static)
        save_binary(q_mask, "Cb", "quantum_binary", True)
        # analytic ground-truth binary edge map (synthetic mode only)
        if gt_binary is not None:
            save_binary(gt_binary.astype(float), "G", "gt_binary", save_static)
        print(f"saved maps + binary masks: {tag}")
    else:
        print(f"saved maps: {tag}")


def plot_confusion(cm, args, shots, outdir, tag):
    """Direction confusion heatmap (GT rows vs quantum columns).
    4x4 axes for Sobel/Prewitt (sign-degenerate), 8x8 for Kirsch."""
    n_dir = cm.shape[0]
    if n_dir == 4:
        labels = [f"{DIRECTIONS[i]}/{DIRECTIONS[i + 4]}" for i in range(4)]
    else:
        labels = list(DIRECTIONS)
    fig, ax = plt.subplots(figsize=(5, 4) if n_dir == 8 else (4, 3.4))
    # sequential dark-for-high colormap: detected directions (high %)
    # render as dark cells, drawing the eye to the meaningful entries.
    im = ax.imshow(cm, cmap="Greys", vmin=0, vmax=100)
    ax.set_xticks(range(n_dir), labels)
    ax.set_yticks(range(n_dir), labels)
    ax.set_xlabel("Quantum Direction", fontsize=12, fontweight="bold")
    ax.set_ylabel("GT (classical) Direction", fontsize=12, fontweight="bold")
    ax.set_title(f"Direction Confusion (%), {args.operator}, shots={shots}",
                 fontsize=13, fontweight="bold")
    for r in range(n_dir):
        for c in range(n_dir):
            if cm[r, c] >= 1:
                # dark cell -> white text, light cell -> black text
                ax.text(c, r, f"{cm[r, c]:.0f}", ha="center", va="center",
                        color="white" if cm[r, c] >= 50 else "black",
                        fontsize=13, fontweight="bold")
    fig.colorbar(im, fraction=0.046)
    plt.tight_layout()
    fname = outdir / f"E_confusion_{tag}.png"
    plt.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"saved: {fname}")


def plot_sweep(df, outdir, tag, pixel_data=None, max_scatter=2000,
               classical_ref=None, detection="maximum"):
    """Shots-axis sweep of the four metrics, each saved as a separate
    figure qced_sweep_<name>_<tag>.png:
      (diracc)  direction accuracy [%]      - higher better
      (cossim)  cosine similarity [0,1]     - higher better (pixel scatter)
    For detection='binary' the magnitude-error pair is replaced by binary
    edge-classification accuracy/F1 (evaluated on detected-edge pixels):
      (acc)     binary edge accuracy [%]    - higher better
      (f1)      binary edge F1 score [0,1]  - higher better
    Otherwise (response/maximum):
      (mse)     magnitude-map MSE           - lower better (log y)
      (psnr)    magnitude-map PSNR [dB]     - higher better
    """
    plt.rcParams.update({"font.size": 13, "axes.labelsize": 15,
                         "lines.linewidth": 2.2, "lines.markersize": 7})
    g = df.groupby("shots")
    xs = np.array(sorted(g.groups))
    # (column, ylabel, filename, log_y, scatter_key, value_format)
    if detection == "binary":
        metric_panels = [("accuracy_pct", "Binary edge accuracy (%)", "acc", False, None, "{:.1f}"),
                         ("f1_score", "Binary edge F1 score", "f1", False, None, "{:.4f}"),
                         ("mag_mse", "Binary edge MSE (error rate)", "mse", False, None, "{:.4f}"),
                         ("mag_psnr", "Binary edge PSNR (dB)", "psnr", False, None, "{:.2f}")]
    else:
        metric_panels = [("mag_mse", "Magnitude MSE", "mse", False, None, "{:.2f}"),
                         ("mag_psnr", "Magnitude PSNR (dB)", "psnr", False, None, "{:.2f}")]
    panels = [("dir_acc_pct", "Accuracy of direction (%)", "diracc", False, None, "{:.1f}"),
              #("cos_sim_median", "Cosine similarity", "cossim", False, "cossim", "{:.4f}"),
              ("cos_sim_median", "Cosine similarity", "cossim", False, "None", "{:.4f}")
              ] + metric_panels
    rng = np.random.default_rng(0)
    for base, ylabel, name, logy, skey, vfmt in panels:
        fig, ax = plt.subplots(figsize=(5, 4))
        # faint per-pixel scatter (first seed) when available
        scatter_vals = []
        if skey is not None:
            for s in xs:
                vals = (pixel_data or {}).get(s, {}).get(skey)
                if vals is None:
                    continue
                vals = np.asarray(vals, dtype=float)
                if 0 < max_scatter < vals.size:
                    vals = vals[rng.choice(vals.size, max_scatter, replace=False)]
                scatter_vals.append(vals)
                ax.scatter(np.full(vals.size, s), vals, marker="X", s=10,
                           color="gray", alpha=0.03, linewidths=0, zorder=1)
        qm = g[base].mean().loc[xs].to_numpy()
        qs = g[base].std().fillna(0).loc[xs].to_numpy()
        ax.errorbar(xs, qm, yerr=qs, marker="o", capsize=4,
                    color="black", zorder=2, label="quantum")
        # classical-vs-GT reference as a horizontal line (shot-independent)
        cval = None
        if classical_ref is not None:
            cval = classical_ref.get(base)
        if cval is not None and np.isfinite(cval):
            ax.axhline(cval, color="crimson", ls="--", lw=1.8,
                       zorder=3, label="classical")
            ax.legend(fontsize=10, loc="best", framealpha=0.7)
        # annotate every point with its mean value (skip non-finite)
        for x, m, e in zip(xs, qm, qs):
            if not np.isfinite(m):
                continue
            ax.annotate(vfmt.format(m), (x, m + e), textcoords="offset points",
                        xytext=(0, 8), ha="center", va="bottom",
                        fontsize=10, color="black", zorder=4,
                        bbox=dict(boxstyle="round,pad=0.15", fc="white",
                                  ec="none", alpha=0.7))
        # fit y-axis to the data range (linear-scale panels) with 8% padding
        # (extra headroom on top so the value labels stay inside the axes)
        if not logy:
            finite = qm[np.isfinite(qm)]
            if finite.size == 0:
                finite = np.array([0.0])
            fe = qs[np.isfinite(qm)] if np.isfinite(qm).any() else np.array([0.0])
            lo = float(np.min(finite - fe))
            hi = float(np.max(finite + fe))
            if scatter_vals:
                allv = np.concatenate(scatter_vals)
                allv = allv[np.isfinite(allv)]
                if allv.size:
                    lo = min(lo, float(np.percentile(allv, 1)))
                    hi = max(hi, float(np.percentile(allv, 99)))
            if cval is not None and np.isfinite(cval):
                lo = min(lo, cval); hi = max(hi, cval)
            if not (np.isfinite(lo) and np.isfinite(hi)):
                lo, hi = 0.0, 1.0
            pad = (hi - lo) * 0.08 or abs(hi) * 0.01 or 0.01
            ax.set_ylim(lo - pad, hi + 3 * pad)
        else:
            top = float(np.max(qm[np.isfinite(qm)])) if np.isfinite(qm).any() else 1.0
            ax.set_ylim(top=top * 1.6)
        ax.set_xscale("log", base=2)
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel("Shots")
        ax.set_ylabel(ylabel)
        ax.grid(True, ls="--", alpha=0.5)
        plt.tight_layout()
        fname = outdir / f"{name}_{tag}.png"
        fig.savefig(fname, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"saved: {fname}")


def evaluate_point(img, gt_edges, gt_cmag, gt_cdir, inner, args, shots,
                   seed, run_id, stem, outdir, make_figures,
                   gt_binary=None, save_static=True, upscale=1,
                   noise_snr=float("inf"), gt_mag_map=None):
    """Evaluate one (shots, seed) point: run the quantum detector and
    compare it against the classical GT with the three metrics. Returns
    a metrics record. save_static: save A, B, Bb (first shots only).
    upscale: magnification for small images."""
    if args.backend == "qpu":
        # QPU is not submitted here: reuse the raw counts saved by
        # run_qpu_experiment.py, so binary/response/maximum all read the
        # same one-off QPU run.
        qmag, qdir, qvec, rate_map, dt = quantum_maps_from_checkpoint(
            img, args.operator, shots, args.remove_dc,
            json_path=getattr(args, "qpu_json", None),
            qpu_dir=getattr(args, "qpu_dir", "qpu_runs"),
            image_stem=stem, run_id=getattr(args, "qpu_run_id", None),
        )
    else:
        qmag, qdir, qvec, rate_map, dt = quantum_maps_cached(
            img, args.operator, args.backend, shots, seed, args.remove_dc,
            cache_dir=getattr(args, "cache_dir", None),
            use_cache=not getattr(args, "no_quantum_cache", False),
        )

    mask = None
    thr = None

    dir_acc = dir_accuracy(qdir, gt_cdir, inner, mask=mask)
    cos_sim = cosine_similarity(gt_edges, qvec, inner, mask=mask)
    # maximum detection: thin the edges by non-maximum suppression along
    # each pixel's gradient direction. Suppressed pixels are zeroed in the
    # magnitude map (and their response vector), so downstream metrics see
    # only the thinned edges.
    if args.detection == "maximum":
        qmag_nms = non_maximum_suppression(qmag, qdir)
        keep = qmag_nms > 0
        qmag = qmag_nms
        qvec = qvec * keep[:, :, None]
        # apply the SAME NMS to the classical result so the two are
        # compared (and drawn) on equal footing
        gt_cmag = non_maximum_suppression(gt_cmag, gt_cdir)
        gt_edges = gt_edges * (gt_cmag > 0)[None, :, :]
    # binary detection: threshold the GT magnitude into an edge mask and
    # evaluate the metrics only on detected-edge pixels. The same GT-based
    # threshold defines the edge set for a fair quantum-vs-classical
    # comparison (edge iff GT magnitude >= T).

    if args.detection == "binary":
        gt_inner_mag = gt_cmag[inner]
        thr = edge_threshold(gt_inner_mag, args.threshold, args.threshold_alpha)
        mask = binary_edge_mask(gt_inner_mag, thr).astype(bool)


    accuracy = f1 = float("nan")
    if args.detection == "binary":
        # threshold the quantum magnitude on its OWN scale (mirrors the
        # classical/quantum binary masks saved by plot_maps), then score
        # accuracy/F1 against the GT binary edge map: the analytic GT
        # (synthetic mode) or else the classical GT threshold mask.
        q_thr = edge_threshold(qmag, args.threshold, args.threshold_alpha)
        q_binary_inner = binary_edge_mask(qmag, q_thr)[inner]
        gt_binary_inner = gt_binary[inner] if gt_binary is not None else mask
        accuracy, f1 = binary_classification_metrics(q_binary_inner, gt_binary_inner)
        # also report MSE/PSNR of the two binary edge maps (values in
        # {0,1}), for ANY input image: MSE = fraction of mismatched
        # pixels (error rate), PSNR = 10*log10(1/MSE) with MAX_I = 1.
        if gt_binary_inner is not None:
            p = np.asarray(q_binary_inner, dtype=float).ravel()
            g = np.asarray(gt_binary_inner, dtype=float).ravel()
            mse = float(np.mean((p - g) ** 2)) if p.size else float("nan")
            psnr = (10.0 * np.log10(1.0 / max(mse, 1e-10))
                    if np.isfinite(mse) and mse >= 0 else float("nan"))
        else:
            mse, psnr = float("nan"), float("nan")
    else:
        mse, psnr = mag_mse_psnr(gt_edges, qvec, inner, mask=mask,
                                 mag_norm=args.mag_norm,
                                 gt_mag_map=gt_mag_map)

    record = {
        "run_id": run_id, "image": stem,
        "operator": args.operator, "backend": args.backend,
        "detection": args.detection,
        "threshold_method": (args.threshold if args.detection == "binary"
                             else "-"),
        "threshold_value": (float(thr) if thr is not None else float("nan")),
        "edge_pixels": (int(mask.sum()) if mask is not None
                       else int(gt_cdir[inner].size)),
        "remove_dc": args.remove_dc,
        "shots": shots, "seed": seed,
        "runtime_s": round(dt, 2),
        "post_selection_mean": float(rate_map[inner].mean()),
        # quantum vs classical GT (over detected-edge pixels in binary mode)
        "dir_acc_pct": dir_acc,          # (3) direction agreement [%], higher better
        "cos_sim_median": float(np.median(cos_sim)) if cos_sim.size else float("nan"),
        "mag_mse": mse,                  # (2) magnitude-map MSE, lower better (response/maximum)
        "mag_psnr": psnr,                # (2) magnitude-map PSNR [dB], higher better (response/maximum)
        "accuracy_pct": accuracy,        # (2') binary edge accuracy [%], higher better (binary only)
        "f1_score": f1,                  # (2') binary edge F1 score [0,1], higher better (binary only)
    }

    if make_figures:
        tag = (f"{stem}_{args.operator}_s{shots}_seed{seed}"
               + ("_dcrm" if args.remove_dc else "")
               + ("" if np.isinf(noise_snr)
                  else f"_snr{noise_snr:g}dB"))
        plot_maps(img, gt_cmag, qmag, inner, args, shots, outdir, tag,
                  thr=thr, save_static=save_static, upscale=upscale,
                  gt_binary=gt_binary,
                  gt_mag_map=(gt_mag_map if args.mag_norm else None),
                  mag_norm=args.mag_norm, cbar=args.cbar)
        # magnitude histogram (quantum vs classical, with Otsu thresholds)
        plot_magnitude_hist(qmag, gt_cmag, inner, args, shots, outdir, tag,
                            gt_mag_map=gt_mag_map)
        n_dir = 4 if args.operator in ("sobel", "prewitt") else 8
        plot_confusion(confusion_matrix(gt_cdir, qdir, inner, n_dir,
                                        mask=mask), args,
                       shots, outdir, tag)

    if args.detection == "binary":
        print(f"    [shots={shots} seed={seed}] "
              f"dir={dir_acc:.1f}% cos={record['cos_sim_median']:.4f} "
              f"acc={accuracy:.1f}% F1={f1:.4f}")
    else:
        print(f"    [shots={shots} seed={seed}] "
              f"dir={dir_acc:.1f}% cos={record['cos_sim_median']:.4f} "
              f"MSE={mse:.2f} PSNR={psnr:.2f}dB")
    pixels = {"cossim": cos_sim}
    return record, pixels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="synthetic",
                        help="'synthetic' (concentric rings), "
                             "'grating' or 'grating:<sharpness>' (oriented "
                             "8-azimuth grating with analytic mag+dir GT; "
                             "sharpness 0.0=smooth sine .. 1.0=near square), "
                             "or a path to an image file")
    parser.add_argument("--operator", default="kirsch",
                        choices=["sobel", "prewitt", "kirsch"])
    parser.add_argument("--backend", default="statevector",
                        choices=sorted(BACKENDS))
    parser.add_argument("--shots", type=int, default=1024)
    parser.add_argument("--cropping", action=argparse.BooleanOptionalAction,
                        default=False)
    parser.add_argument("--crop", type=int, default=48)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--remove-dc", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--sweep", default="512,1024,2048,4096,8192",
                        help="comma-separated shots list (empty disables)")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--detection", default="binary",
                        choices=["response", "maximum", "binary"],
                        help="response: raw magnitude over all pixels; "
                             "maximum: non-maximum suppression (thin the "
                             "edges by keeping only local maxima along the "
                             "gradient direction; no hyperparameters); "
                             "binary: threshold the magnitude into an "
                             "edge/non-edge mask and evaluate only on "
                             "detected-edge pixels")
    parser.add_argument("--threshold", default="otsu",
                        choices=["fixed", "otsu"],
                        help="binary threshold rule: otsu (automatic) or "
                             "fixed (alpha * max magnitude)")
    parser.add_argument("--mag-norm", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="rescale magnitude maps to [0,255] before "
                             "MSE/PSNR (removes operator gain; enables "
                             "response/maximum comparison against the "
                             "synthetic analytic GT)")
    parser.add_argument("--threshold-alpha", type=float, default=0.15,
                        help="fraction of max magnitude for --threshold fixed")
    parser.add_argument("--noise-sweep", default="",
                        help="comma-separated INPUT SNR list in dB, e.g. "
                             "inf,40,30,20,10,0 (higher = cleaner; 'inf' or "
                             "'clean' = no noise). Gaussian noise is added to "
                             "reach each SNR. The GT stays the clean-image "
                             "edge map and classical/quantum both process the "
                             "noisy image; results go to per-SNR subfolders "
                             "(snr_<x>dB/) and quantum-minus-classical "
                             "heatmaps (SNR x shots) are produced.")
    parser.add_argument("--cbar", action="store_true",
                        help="also save a colorbar'd PNG (…_cbar.png) for "
                             "every A/B/C/D(/Gm) map: native, upscaled, "
                             "and interior-only versions")
    parser.add_argument("--skip-existing", action="store_true",
                        help="skip (shots, seed, noise) points already "
                             "present in the results xlsx raw sheet, so an "
                             "interrupted run can be resumed and duplicates "
                             "are avoided")
    parser.add_argument("--cache-dir", default="quantum_cache",
                        help="directory for cached quantum maps. The backend/"
                             "QPU run is reused across detection modes (binary/"
                             "response/maximum) keyed by image+operator+backend"
                             "+shots+seed+remove_dc, so a QPU job is paid for "
                             "once. Shared across runs; keep it stable.")
    parser.add_argument("--no-quantum-cache", action="store_true",
                        help="disable the quantum-maps cache (always re-run "
                             "the backend/QPU)")
    parser.add_argument("--results-dir", default="results",
                        help="parent directory that collects every run's "
                             "output folder (default: results/)")
    parser.add_argument("--qpu-json", default=None,
                        help="(backend qpu) explicit path to the counts JSON "
                             "saved by run_qpu_experiment.py; overrides "
                             "--qpu-dir auto-lookup")
    parser.add_argument("--qpu-dir", default="qpu_runs",
                        help="(backend qpu) folder to auto-find the counts "
                             "JSON qpu_ckpt_<image>_<operator>_s<shots>.json "
                             "(default: qpu_runs/)")
    parser.add_argument("--qpu-run-id", default=None,
                        help="(backend qpu) realisation tag to pick the counts "
                             "JSON written by run_qpu_experiment.py --run-id. "
                             "Pair with --seed to record each realisation as a "
                             "separate row in the same results xlsx, so the "
                             "summary gives mean +/- std across QPU runs.")
    args = parser.parse_args()

    img, src = load_image(args.image, args.cropping, args.crop)
    stem = synthetic_stem(src)
    mode_tag = args.detection
    if args.detection == "binary":
        mode_tag += f"_{args.threshold}"
    if args.mag_norm:
        mode_tag += "_magnorm"
    # collect every run's output folder under one parent (default results/)
    results_root = Path(getattr(args, "results_dir", "results"))
    results_root.mkdir(parents=True, exist_ok=True)
    outdir = results_root / f"{stem}_{args.operator}_{args.backend}_{mode_tag}"
    outdir.mkdir(parents=True, exist_ok=True)
    xlsx_path = outdir / f"qced_results_{stem}_{args.backend}.xlsx"
    det_desc = args.detection
    if args.detection == "binary":
        det_desc += f" ({args.threshold}" + (
            f", alpha={args.threshold_alpha})" if args.threshold == "fixed"
            else ")")
    print(f"image: {src}, cropping: {args.cropping}, shape: {img.shape}, "
          f"operator: {args.operator}, backend: {args.backend}, "
          f"remove_dc: {args.remove_dc}, detection: {det_desc}")
    print(f"results -> {outdir}/")

    # GT for direction = classical compass response (the discrete 3x3
    # operator's own directions). For the synthetic concentric-ring image
    # the edge LOCATION is also known analytically; that analytic edge
    # magnitude is thresholded into a GT binary edge map used only for the
    # binary accuracy/F1 comparison.
    gt_edges, gt_cmag, gt_cdir = classical_maps(img, args.operator)
    # Evaluate on interior pixels only: boundary pixels rely on reflect
    # padding, which is inaccurate for both classical and quantum (they
    # share the same padding), so excluding the 1-pixel border keeps the
    # metrics clean. Quantum still COMPUTES every pixel (for the figures);
    # only the scoring range is restricted here.
    inner = (slice(1, -1), slice(1, -1))

    gt_binary = None            # analytic GT binary edge map (synthetic only)
    gt_mag_map = None           # analytic GT magnitude map (synthetic only)
    classical_ref = None        # classical-vs-GT reference for the sweep
    # synthetic + response/maximum: use the analytic GT magnitude as the
    # MSE/PSNR reference (only meaningful with --mag-norm, which puts both
    # maps on the same [0,255] scale)
    if is_synthetic(src) and args.detection in ("response", "maximum"):
        if args.mag_norm:
            gt_mag_map = analytic_gt_magnitude(src, img.shape[0])
            # maximum: apply the SAME NMS to the analytic GT magnitude so
            # it is thinned like the classical/quantum maps. NMS uses the
            # classical direction (gt_cdir) since the analytic radial
            # direction does not match the discrete 3x3 compass.
            if args.detection == "maximum":
                gt_mag_map = non_maximum_suppression(gt_mag_map, gt_cdir)
                print("synthetic mode: analytic GT magnitude + NMS "
                      "(mag-norm on) for maximum")
            else:
                print("synthetic mode: analytic GT magnitude reference "
                      "(mag-norm on) for response")
        else:
            print("synthetic mode: response/maximum uses CLASSICAL as the "
                  "MSE reference (enable --mag-norm to compare vs analytic GT)")
    if is_synthetic(src) and args.detection == "binary":
        gt_analytic_mag = analytic_gt_magnitude(src, img.shape[0])
        thr_gt = edge_threshold(gt_analytic_mag, args.threshold,
                                args.threshold_alpha)
        gt_binary = binary_edge_mask(gt_analytic_mag, thr_gt).astype(int)
        # classical edge map vs GT binary map (constant over the sweep)
        c_mag = np.abs(gt_edges).max(axis=0)
        thr_c = edge_threshold(c_mag, args.threshold, args.threshold_alpha)
        c_binary = binary_edge_mask(c_mag, thr_c).astype(int)
        c_acc, c_f1 = binary_classification_metrics(c_binary, gt_binary, inner=inner)
        cbp = c_binary[inner].astype(float).ravel()
        cbg = gt_binary[inner].astype(float).ravel()
        c_mse = float(np.mean((cbp - cbg) ** 2)) if cbp.size else float("nan")
        c_psnr = (10.0 * np.log10(1.0 / max(c_mse, 1e-10))
                  if np.isfinite(c_mse) else float("nan"))
        classical_ref = {"accuracy_pct": c_acc, "f1_score": c_f1,
                         "mag_mse": c_mse, "mag_psnr": c_psnr}
        print(f"synthetic mode: analytic GT binary edge map "
              f"({int(gt_binary.sum())} edge pixels); "
              f"direction GT = classical")
        print(f"classical vs GT (binary): accuracy={c_acc:.1f}% "
              f"F1={c_f1:.4f}")

    l2_ref = statevector_l2_reference(img, args.operator, args.remove_dc, inner)
    print(f"L2 statevector reference (implementation fidelity): {l2_ref:.2e}")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    do_sweep = args.backend != "statevector" and bool(args.sweep)
    shots_list = ([int(x) for x in args.sweep.split(",")] if do_sweep
                  else [args.shots])
    seeds = args.seeds if do_sweep else 1
    upscale = 10 if img.shape[0] < 48 else 1

    def _parse_snr(x):
        x = x.strip().lower()
        return float("inf") if x in ("inf", "clean", "none") else float(x)
    noise_list = ([_parse_snr(x) for x in args.noise_sweep.split(",")]
                  if args.noise_sweep else [float("inf")])
    noise_mode = bool(args.noise_sweep)
    if noise_mode:
        outdir = Path(f"{outdir}_noisesweep")
        outdir.mkdir(parents=True, exist_ok=True)

    done_points = load_done_points(xlsx_path) if args.skip_existing else {}
    if args.skip_existing and done_points:
        print(f"skip-existing: {len(done_points)} points already in xlsx "
              f"will be skipped")

    all_records = []
    for nz in noise_list:
        # noisy input; GT (clean-image edges) is kept fixed
        noisy_img = add_noise_at_snr(img, nz, seed=args.seed) \
            if noise_mode else img
        c_edges, c_cmag, c_cdir = classical_maps(noisy_img, args.operator)

        # classical-vs-GT reference for THIS noise level (binary
        # accuracy/F1 vs analytic GT). Only this pair is meaningful for
        # the noise sweep: direction and cosine use the classical result
        # itself as reference, so they cannot judge quantum-vs-classical
        # robustness and are excluded.
        cls_ref = None
        if gt_binary is not None:
            # binary: classical accuracy/F1 vs analytic GT binary map
            cmag_n = np.abs(c_edges).max(axis=0)
            thr_cn = edge_threshold(cmag_n, args.threshold, args.threshold_alpha)
            c_bin_n = binary_edge_mask(cmag_n, thr_cn).astype(int)
            ca, cf = binary_classification_metrics(c_bin_n, gt_binary, inner=inner)
            cbp = c_bin_n[inner].astype(float).ravel()
            cbg = gt_binary[inner].astype(float).ravel()
            cm_ = float(np.mean((cbp - cbg) ** 2)) if cbp.size else float("nan")
            cp_ = (10.0 * np.log10(1.0 / max(cm_, 1e-10))
                   if np.isfinite(cm_) else float("nan"))
            cls_ref = {"accuracy_pct": ca, "f1_score": cf,
                       "mag_mse": cm_, "mag_psnr": cp_}
        elif gt_mag_map is not None:
            # response/maximum + mag-norm: classical MSE/PSNR vs analytic GT
            # magnitude. Build the classical magnitude map the same way the
            # quantum path does (NMS-thinned for maximum) and compare on the
            # same [0,255] scale as the metric.
            c_vec = np.abs(c_edges).transpose(1, 2, 0)   # (H,W,8)
            if args.detection == "maximum":
                c_mag_nms = non_maximum_suppression(
                    np.abs(c_edges).max(axis=0), c_cdir)
                c_vec = c_vec * (c_mag_nms > 0)[:, :, None]
            cm, cp = mag_mse_psnr(c_edges, c_vec, inner,
                                  mag_norm=args.mag_norm,
                                  gt_mag_map=gt_mag_map)
            cls_ref = {"mag_mse": cm, "mag_psnr": cp}

        nz_tag = "clean" if np.isinf(nz) else f"{nz:g}dB"
        sub = outdir / f"snr_{nz_tag}" if noise_mode else outdir
        sub.mkdir(exist_ok=True)
        pixel_data = {}
        if noise_mode:
            print()
            print("#" * 68)
            print(f"  INPUT SNR = {nz_tag}   ->   {sub}/")
            print("#" * 68)

        for si, s in enumerate(shots_list, start=1):
            print(f"  [SNR {nz_tag}] SWEEP {si}/{len(shots_list)} shots={s}")
            point_records = []
            for k in range(seeds):
                seed_k = args.seed + k
                if args.skip_existing and (nz, s, seed_k) in done_points:
                    print(f"    skip (noise={nz_tag}, shots={s}, "
                          f"seed={seed_k}) — already in xlsx")
                    prev = done_points[(nz, s, seed_k)]
                    all_records.append(prev)
                    point_records.append(prev)
                    continue
                # evaluate quantum on the NOISY image, compare to fixed GT
                record, pixels = evaluate_point(
                    noisy_img, gt_edges, gt_cmag, gt_cdir, inner, args, s,
                    seed_k, run_id, stem, sub, make_figures=(k == 0),
                    gt_binary=gt_binary,
                    save_static=(si == 1 and k == 0),
                    upscale=upscale, noise_snr=nz, gt_mag_map=gt_mag_map,
                )
                # attach noise + classical reference for the heatmap
                record["noise_snr"] = nz
                if cls_ref is not None:
                    for mk, mv in cls_ref.items():
                        record[f"classical_{mk}"] = mv
                if k == 0:
                    pixel_data[s] = pixels
                append_xlsx(xlsx_path, {"raw": [record]})
                all_records.append(record)
                point_records.append(record)

            agg = {"run_id": run_id, "image": stem, "operator": args.operator,
                   "backend": args.backend, "remove_dc": args.remove_dc,
                   "detection": args.detection, "threshold": args.threshold,
                   "noise_snr": nz, "shots": s, "n_seeds": seeds}
            for key in ("dir_acc_pct", "cos_sim_median", "mag_mse",
                        "mag_psnr", "accuracy_pct", "f1_score",
                        "post_selection_mean"):
                vals = [r[key] for r in point_records]
                agg[f"{key}_mean"] = float(np.mean(vals))
                agg[f"{key}_std"] = float(np.std(vals))
            # store the classical-vs-GT reference (shot-independent) so
            # downstream scripts can read it straight from the xlsx
            if cls_ref is not None:
                for mk, mv in cls_ref.items():
                    agg[f"classical_{mk}"] = mv
            append_xlsx(xlsx_path, {"summary": [agg]})

        # per-noise shot-sweep line plots (in the noise subfolder)
        if do_sweep and len(shots_list) > 1:
            tag = f"{stem}_{args.operator}_snr{nz_tag}" + \
                ("_dcrm" if args.remove_dc else "")
            plot_sweep(pd.DataFrame([r for r in all_records
                                     if r["noise_snr"] == nz]),
                       sub, tag, pixel_data, classical_ref=cls_ref,
                       detection=args.detection)

    # noise-robustness line plots: metric vs input SNR, one figure per
    # shot count, classical and quantum curves overlaid. Only the
    # magnitude metrics (MSE/PSNR vs analytic GT) are shown, since
    # direction/cosine reference the classical result.
    if noise_mode and len(noise_list) > 1:
        ndir = outdir / "noise_robustness"
        ndir.mkdir(exist_ok=True)
        df_all = pd.DataFrame(all_records)
        tag = f"{stem}_{args.operator}"
        if args.detection == "binary":
            noise_metrics = [("accuracy_pct", "Accuracy (%, vs GT)", False),
                             ("f1_score", "F1 score (vs GT)", False)]
        else:
            noise_metrics = [("mag_mse", "Magnitude MSE (vs GT)", True),
                             ("mag_psnr", "Magnitude PSNR (dB, vs GT)", False)]
        for metric, ylab, logy in noise_metrics:
            if f"classical_{metric}" in df_all.columns:
                plot_noise_lines(df_all, metric, ylab, ndir, tag, logy=logy)


if __name__ == "__main__":
    main()