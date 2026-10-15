"""Overlay the shots-sweep metric curves of two backends (typically
'shots' vs 'noisy') that were run on the same image / operator /
detection settings, reading only the result xlsx files that demo.py
already wrote.

For each metric a single figure is produced with:
  - the 'shots' backend quantum curve (with seed error bars),
  - the 'noisy' backend quantum curve,
  - the classical-vs-GT reference as a horizontal line (read from the
    xlsx 'summary' sheet; identical for both backends, so drawn once).

The plot theme (fonts, log-x axis, annotations, y-range fitting) matches
demo.py. Figures are saved into <noisy_dir>/comparison/.

Which metrics are plotted is auto-detected from the summary columns:
  direction accuracy, cosine similarity, and either
  (MSE, PSNR)  [response / maximum detection]  or
  (accuracy, F1) [binary detection].

Usage
-----
    python compare_backends.py \
        results_synthetic_kirsch_shots_binary_otsu \
        results_synthetic_kirsch_noisy_binary_otsu

    # explicit metric list / output dir
    python compare_backends.py <shots_dir> <noisy_dir> \
        --metrics dir_acc_pct accuracy_pct f1_score --outdir figs
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# metric key -> (y-axis label, filename stem, value format, log-y)
METRIC_INFO = {
    "dir_acc_pct": ("Accuracy of direction (%)", "diracc", "{:.1f}", False),
    "cos_sim_median": ("Cosine similarity", "cossim", "{:.4f}", False),
    "mag_mse": ("Magnitude MSE", "mse", "{:.2f}", False),
    "mag_psnr": ("Magnitude PSNR (dB)", "psnr", "{:.2f}", False),
    "accuracy_pct": ("Binary edge accuracy (%)", "acc", "{:.1f}", False),
    "f1_score": ("Binary edge F1 score", "f1", "{:.4f}", False),
}


def find_summary_xlsx(folder: Path):
    """Return the results xlsx in `folder` (the file demo.py named
    qced_results_<image>_<backend>.xlsx)."""
    cands = sorted(folder.glob("qced_results_*.xlsx"))
    if not cands:
        raise SystemExit(f"no qced_results_*.xlsx found in {folder}")
    return cands[0]


def load_summary(folder: Path):
    """Load the 'summary' sheet; fall back to aggregating 'raw' if the
    summary sheet is missing."""
    xlsx = find_summary_xlsx(folder)
    sheets = pd.read_excel(xlsx, sheet_name=None)
    if "summary" in sheets and len(sheets["summary"]):
        df = sheets["summary"].copy()
    else:
        raw = sheets["raw"]
        # aggregate raw into per-shots mean/std to mimic the summary sheet
        keys = [c for c in METRIC_INFO if c in raw.columns]
        grp = raw.groupby("shots")
        df = grp[keys].mean().add_suffix("_mean")
        df = df.join(grp[keys].std().fillna(0).add_suffix("_std"))
        df = df.reset_index()
        for c in ("backend",):
            if c in raw.columns:
                df[c] = raw[c].iloc[0]
    return df.sort_values("shots").reset_index(drop=True)


def available_metrics(df):
    """Metrics present as <key>_mean columns, in a sensible order."""
    order = ["dir_acc_pct", "cos_sim_median",
             "mag_mse", "mag_psnr", "accuracy_pct", "f1_score"]
    return [k for k in order if f"{k}_mean" in df.columns
            and df[f"{k}_mean"].notna().any()]


def classical_value(df, metric):
    """Classical-vs-GT reference for `metric`, if stored in the summary
    (column 'classical_<metric>'); else None."""
    col = f"classical_{metric}"
    if col in df.columns and df[col].notna().any():
        return float(df[col].dropna().iloc[0])
    return None


def plot_metric(metric, shots_df, noisy_df, outdir, tag):
    """One overlaid figure for `metric`: shots curve, noisy curve, and
    the classical reference line. Matches demo.py's plot_sweep theme."""
    ylabel, name, vfmt, logy = METRIC_INFO[metric]
    mcol, scol = f"{metric}_mean", f"{metric}_std"

    plt.rcParams.update({"font.size": 13, "axes.labelsize": 15,
                         "lines.linewidth": 2.2, "lines.markersize": 7})
    fig, ax = plt.subplots(figsize=(5, 4))

    series = [("shots", shots_df, "C0", "o"),
              ("noisy", noisy_df, "C3", "s")]
    all_y = []
    for label, df, color, marker in series:
        if df is None or mcol not in df.columns:
            continue
        xs = df["shots"].to_numpy()
        ym = df[mcol].to_numpy()
        ye = df[scol].to_numpy() if scol in df.columns else np.zeros_like(ym)
        ax.errorbar(xs, ym, yerr=ye, marker=marker, capsize=4,
                    color=color, zorder=2, label=label)
        finite = ym[np.isfinite(ym)]
        fe = ye[np.isfinite(ym)] if np.isfinite(ym).any() else np.array([0.0])
        if finite.size:
            all_y.append(finite - fe)
            all_y.append(finite + fe)
        # annotate each point (skip non-finite)
        for x, m, e in zip(xs, ym, ye):
            if not np.isfinite(m):
                continue
            ax.annotate(vfmt.format(m), (x, m + e),
                        textcoords="offset points", xytext=(0, 8),
                        ha="center", va="bottom", fontsize=9, color=color,
                        zorder=4, bbox=dict(boxstyle="round,pad=0.15",
                                            fc="white", ec="none", alpha=0.7))

    # classical reference (shot-independent) — read from whichever df has it
    cval = classical_value(shots_df, metric)
    if cval is None and noisy_df is not None:
        cval = classical_value(noisy_df, metric)
    if cval is not None and np.isfinite(cval):
        ax.axhline(cval, color="black", ls="--", lw=1.8, zorder=3,
                   label="classical")
        all_y.append(np.array([cval]))

    ax.set_xscale("log", base=2)
    if logy:
        ax.set_yscale("log")
    else:
        flat = np.concatenate(all_y) if all_y else np.array([0.0, 1.0])
        flat = flat[np.isfinite(flat)]
        if flat.size:
            lo, hi = float(flat.min()), float(flat.max())
            pad = (hi - lo) * 0.08 or abs(hi) * 0.01 or 0.01
            ax.set_ylim(lo - pad, hi + 3 * pad)

    ax.set_xlabel("Shots")
    ax.set_ylabel(ylabel)
    ax.grid(True, ls="--", alpha=0.5)
    ax.legend(fontsize=10, loc="best", framealpha=0.7)
    plt.tight_layout()
    fname = outdir / f"cmp_{name}_{tag}.png"
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {fname}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("shots_dir", help="results folder of the 'shots' backend")
    ap.add_argument("noisy_dir", help="results folder of the 'noisy' backend")
    ap.add_argument("--metrics", nargs="*", default=None,
                    help="metric keys to plot (default: auto-detect)")
    ap.add_argument("--outdir", default=None,
                    help="output dir (default: <noisy_dir>/comparison)")
    args = ap.parse_args()

    shots_dir = Path(args.shots_dir)
    noisy_dir = Path(args.noisy_dir)
    shots_df = load_summary(shots_dir)
    noisy_df = load_summary(noisy_dir)

    outdir = Path(args.outdir) if args.outdir else (noisy_dir / "comparison")
    outdir.mkdir(parents=True, exist_ok=True)

    metrics = args.metrics or available_metrics(shots_df)
    metrics = [m for m in metrics if m in METRIC_INFO]
    if not metrics:
        raise SystemExit("no plottable metrics found in the summaries")

    # tag from folder name (strip the leading 'results_' if present)
    stem = shots_dir.name
    if stem.startswith("results_"):
        stem = stem[len("results_"):]
    stem = stem.replace("_shots_", "_").replace("_noisy_", "_")

    magnorm = "magnorm" in shots_dir.name
    print(f"shots : {find_summary_xlsx(shots_dir).name}")
    print(f"noisy : {find_summary_xlsx(noisy_dir).name}")
    print(f"metrics: {metrics}")
    print(f"mag-norm: {magnorm}  (magnitudes on {'[0,255]' if magnorm else 'raw'} scale)")
    print(f"output: {outdir}/")
    for metric in metrics:
        plot_metric(metric, shots_df, noisy_df, outdir, stem)


if __name__ == "__main__":
    main()