"""Dual-axis performance plots from a QCED noisy-sim results xlsx.

Reads one results workbook (one setting: image, detection, mag-norm; the
``summary`` sheet produced by demo.py) and draws, per input-SNR level, two
twin-axis figures against the shot count:

  Figure 1  left y = magnitude MSE        right y = magnitude PSNR [dB]
  Figure 2  left y = cosine similarity    right y = direction accuracy [%]

Both figures share the same style: x = shots (log2), two y-axes distinguished
by their y-label and line colour, and every data point annotated with its
value. When the workbook carries a classical reference (classical_mag_mse /
classical_mag_psnr) it is drawn on Figure 1 as a dashed horizontal line.

Binary detection: the MSE stored by demo.py is a {0,1} edge-map error rate
(MAX_I = 1). It is rescaled to the 0-255 intensity scale by multiplying by
255**2 before plotting. PSNR is unchanged: for a pure rescale the error and
MAX_I**2 both grow by 255**2, so 10*log10(MAX**2/MSE) is scale-invariant and
the stored value already equals the 0-255 PSNR.

Usage:
    python plot_dual_axis.py results/qced_results_grating_s0p0_noisy.xlsx
    python plot_dual_axis.py <xlsx> --snr 20 --outdir figs --no-classical
    python plot_dual_axis.py <xlsx> --errorbars        # add +/-1 std bars
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Use a Times-like serif. True "Times New Roman" is not bundled with
# matplotlib; this chain picks it when installed (e.g. msttcorefonts) and
# otherwise falls back to Liberation Serif, a metric-identical Times clone,
# then DejaVu Serif. To revert to the default sans-serif, delete this block.
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Liberation Serif",
                              "Nimbus Roman", "DejaVu Serif"]
plt.rcParams["mathtext.fontset"] = "stix"   # serif-matching math glyphs
# A user matplotlibrc with autolayout / constrained_layout enabled would
# override our fig.subplots_adjust(), clipping the outside legend and changing
# the layout from machine to machine. Force them off for deterministic output.
plt.rcParams["figure.autolayout"] = False
plt.rcParams["figure.constrained_layout.use"] = False
# A user rc with 'savefig.bbox: tight' would crop every save (version-
# dependent in mpl 3.11) even though we never pass bbox_inches. Force standard.
plt.rcParams["savefig.bbox"] = "standard"
plt.rcParams["savefig.pad_inches"] = 0.1

# left / right axis colours, shared by both figures
_CL = "#1f77b4"   # left  axis (MSE,  cosine)
_CR = "#d62728"   # right axis (PSNR, direction accuracy)
_BINARY_SCALE = 255.0 * 255.0


def _load_summary(xlsx_path: Path) -> pd.DataFrame:
    """Load the per-(noise_snr, shots) aggregated rows. Prefer the
    ``summary`` sheet; fall back to aggregating ``raw`` over seeds."""
    sheets = pd.read_excel(xlsx_path, sheet_name=None)
    if "summary" in sheets and not sheets["summary"].empty:
        df = sheets["summary"].copy()
    elif "raw" in sheets and not sheets["raw"].empty:
        raw = sheets["raw"].copy()
        keys = ["noise_snr", "shots"]
        metric = ["dir_acc_pct", "cos_sim_median", "mag_mse", "mag_psnr"]
        present = [m for m in metric if m in raw.columns]
        agg = raw.groupby(keys, dropna=False)[present].mean().reset_index()
        agg = agg.rename(columns={m: f"{m}_mean" for m in present})
        for meta in ("detection", "image", "operator", "backend"):
            if meta in raw.columns:
                agg[meta] = raw[meta].iloc[0]
        for c in raw.columns:
            if c.startswith("classical_"):
                # classical ref is shot-independent; carry one value per snr
                agg = agg.merge(
                    raw.groupby("noise_snr", dropna=False)[c].first()
                    .reset_index(), on="noise_snr", how="left")
        df = agg
    else:
        raise ValueError(f"{xlsx_path} has neither a 'summary' nor 'raw' sheet.")

    # normalise noise_snr: inf / 'inf' / 'clean' -> +inf
    def _to_snr(v):
        if isinstance(v, str):
            s = v.strip().lower()
            if s in ("inf", "clean", "+inf", "none", ""):
                return math.inf
            try:
                return float(s)
            except ValueError:
                return math.inf
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return math.inf
        return float(v)

    if "noise_snr" not in df.columns:
        df["noise_snr"] = math.inf
    df["noise_snr"] = df["noise_snr"].map(_to_snr)
    df["shots"] = pd.to_numeric(df["shots"], errors="coerce")
    df = df.dropna(subset=["shots"])
    df["shots"] = df["shots"].astype(int)
    # a re-run appends new rows; keep the last per (snr, shots)
    df = df.drop_duplicates(subset=["noise_snr", "shots"], keep="last")
    return df


def _snr_tag(snr: float) -> str:
    return "clean" if math.isinf(snr) else f"{snr:g}dB"


def _dual_axis_figure(x, yL, yR, labelL, labelR, fmtL, fmtR, title,
                      out_path, eL=None, eR=None, refL=None, refR=None,
                      refL_label=None, refR_label=None):
    """Draw one twin-axis figure: left series (yL) and right series (yR)
    vs x (log2 shots), each point annotated, optional std bars and
    horizontal reference lines."""
    fig, axL = plt.subplots(figsize=(4.5,2.7))
    fig.set_layout_engine("none")   # ignore any rc autolayout/constrained
    axR = axL.twinx()

    kwL = dict(color=_CL, marker="o", ls="-", lw=1.8, ms=6, zorder=3)
    kwR = dict(color=_CR, marker="s", ls="--", lw=1.8, ms=6, zorder=3)
    if eL is not None:
        axL.errorbar(x, yL, yerr=eL, ecolor=_CL, elinewidth=1,
                     capsize=3, **kwL)
    else:
        axL.plot(x, yL, **kwL)
    if eR is not None:
        axR.errorbar(x, yR, yerr=eR, ecolor=_CR, elinewidth=1,
                     capsize=3, **kwR)
    else:
        axR.plot(x, yR, **kwR)

    axL.set_xscale("log", base=2)
    axL.set_xticks(x)
    axL.set_xticklabels([str(int(s)) for s in x])
    axL.set_xlabel("shots", fontweight="bold")
    axL.minorticks_off()

    axL.set_ylabel(labelL, color=_CL, fontweight="bold")
    axR.set_ylabel(labelR, color=_CR, fontweight="bold")
    axL.tick_params(axis="y", labelcolor=_CL)
    axR.tick_params(axis="y", labelcolor=_CR)
    axL.spines["left"].set_color(_CL)
    axL.spines["right"].set_color(_CR)
    axR.spines["left"].set_color(_CL)
    axR.spines["right"].set_color(_CR)

    # reference lines first, so the y-range below accounts for them
    handles = [plt.Line2D([], [], **kwL), plt.Line2D([], [], **kwR)]
    labels = [labelL, labelR]
    if refL is not None and np.isfinite(refL):
        axL.axhline(refL, color=_CL, ls=":", lw=1.3, alpha=0.7)
        handles.append(plt.Line2D([], [], color=_CL, ls=":", lw=1.3))
        labels.append(refL_label or "classical (left)")
    if refR is not None and np.isfinite(refR):
        axR.axhline(refR, color=_CR, ls=":", lw=1.3, alpha=0.7)
        handles.append(plt.Line2D([], [], color=_CR, ls=":", lw=1.3))
        labels.append(refR_label or "classical (right)")

    # Expand each y-axis so the above/below value labels stay inside the
    # plotting area (the legend now lives OUTSIDE, above the axes, so the top
    # pad only has to fit the upper labels).
    def _pad_ylim(ax, vals, ref, top=0.18, bot=0.18):
        v = [t for t in vals if np.isfinite(t)]
        if ref is not None and np.isfinite(ref):
            v.append(ref)
        if not v:
            return
        lo, hi = min(v), max(v)
        rng = (hi - lo) or (abs(hi) or 1.0)
        ax.set_ylim(lo - bot * rng, hi + top * rng)
    _pad_ylim(axL, yL, refL)
    _pad_ylim(axR, yR, refR)

    # Drop y-ticks that fall outside the (padded) view. matplotlib otherwise
    # draws a label for the next round number just past the top/bottom of the
    # axis; such an out-of-range label is NOT counted by get_tightbbox, so it
    # would clip the figure edge. Pruning keeps the axis clean and makes the
    # margin measurement below exact.
    def _prune_ticks(ax):
        lo, hi = sorted(ax.get_ylim())
        ax.set_yticks([t for t in ax.get_yticks() if lo <= t <= hi])
    _prune_ticks(axL)
    _prune_ticks(axR)

    # Value labels: left series ALWAYS above its point, right series ALWAYS
    # below. Opposite fixed sides never overlap each other -- even where the
    # two curves coincide (which they always do on fig1, PSNR being inversely
    # related to MSE). This is fully deterministic: it does not depend on text
    # metrics, the renderer, or the matplotlib version, so it renders the same
    # everywhere. First/last labels are aligned inward to stay in the x-range.
    def _ha(xi):
        return "left" if xi == x[0] else ("right" if xi == x[-1] else "center")

    for xi, yi in zip(x, yL):
        if np.isfinite(yi):
            axL.annotate(fmtL(yi), (xi, yi), textcoords="offset points",
                         xytext=(0, 6), ha=_ha(xi), va="bottom", fontsize=7.5,
                         color=_CL, clip_on=False, zorder=7)
    for xi, yi in zip(x, yR):
        if np.isfinite(yi):
            axR.annotate(fmtR(yi), (xi, yi), textcoords="offset points",
                         xytext=(0, -6), ha=_ha(xi), va="top", fontsize=7.5,
                         color=_CR, clip_on=False, zorder=7)

    # Legend ABOVE the axes (outside the data area) so it can never cover a
    # value label -- deterministic, no overlap test needed.
    ncol = min(len(handles), 2)
    nrow = int(math.ceil(len(handles) / ncol))
    leg = axR.legend(handles, labels, loc="lower center",
                     bbox_to_anchor=(0.5, 1.01), ncol=ncol, fontsize=8,
                     framealpha=0.7, columnspacing=1.0, handletextpad=0.5,
                     borderaxespad=0.0)
    leg.set_zorder(6)
    #axL.set_title(title, fontsize=10, fontweight="bold")
    axL.grid(True, which="major", axis="x", ls=":", alpha=0.4)

    # Fit each margin to exactly what the decorations need: measure (once) how
    # far the labels, tick labels, value annotations and the legend stick out
    # past the data-area spines, then set the subplot margins to that overhang
    # plus a small uniform pad. This fits wide y-tick labels / a large rc font
    # without clipping AND leaves no excess whitespace above the legend (its
    # height is measured, not a fixed fraction). It only repositions the
    # subplot inside the FIXED canvas, so nothing is cropped and the output
    # size stays constant and deterministic across machines.
    W, H = fig.bbox.width, fig.bbox.height
    pad = 4.0
    # Resizing the subplot makes matplotlib recompute tick locations, which
    # changes the overhang, so a single measurement would still clip the top
    # tick. Iterate measure->adjust a few times; it converges in 2-3 passes.
    for _ in range(4):
        fig.canvas.draw()
        rend = fig.canvas.get_renderer()
        spine = axL.get_window_extent(rend)         # data-area box (both axes)
        tb = [b for b in (axL.get_tightbbox(rend), axR.get_tightbbox(rend))
              if b is not None]
        left_over = max([spine.x0 - b.x0 for b in tb] + [0.0])
        right_over = max([b.x1 - spine.x1 for b in tb] + [0.0])
        bot_over = max([spine.y0 - b.y0 for b in tb] + [0.0])
        top_over = max([b.y1 - spine.y1 for b in tb] + [0.0])   # incl. legend
        fig.subplots_adjust(
            left=(left_over + pad) / W, right=1.0 - (right_over + pad) / W,
            bottom=(bot_over + pad) / H, top=1.0 - (top_over + pad) / H)
    fig.savefig(out_path, dpi=800)
    plt.close(fig)
    return out_path


def _col(df, name):
    return df[name].to_numpy(dtype=float) if name in df.columns else None


def make_plots(xlsx_path, outdir=None, snr_filter=None,
               classical=True, errorbars=False):
    xlsx_path = Path(xlsx_path)
    df = _load_summary(xlsx_path)
    detection = (str(df["detection"].iloc[0])
                 if "detection" in df.columns else "unknown")
    image = str(df["image"].iloc[0]) if "image" in df.columns else xlsx_path.stem
    operator = str(df["operator"].iloc[0]) if "operator" in df.columns else "?"
    backend = str(df["backend"].iloc[0]) if "backend" in df.columns else "?"
    magnorm = "magnorm" in str(xlsx_path).lower()

    outdir = Path(outdir) if outdir else xlsx_path.parent
    outdir.mkdir(parents=True, exist_ok=True)
    stem = xlsx_path.stem

    snrs = sorted(df["noise_snr"].unique(),
                  key=lambda v: (math.inf if math.isinf(v) else v))
    if snr_filter is not None:
        want = (math.inf if str(snr_filter).lower() in ("inf", "clean")
                else float(snr_filter))
        snrs = [s for s in snrs if (math.isinf(s) and math.isinf(want))
                or (not math.isinf(s) and abs(s - want) < 1e-9)]
        if not snrs:
            raise ValueError(f"No rows for --snr {snr_filter}. "
                             f"Available: {[_snr_tag(s) for s in df['noise_snr'].unique()]}")

    binary = (detection == "binary")
    base_title = (f"{image} | {operator} | {backend} | {detection}"
                  + (" | mag-norm" if magnorm else ""))
    written = []

    for snr in snrs:
        sub = df[df["noise_snr"] == snr].sort_values("shots")
        if sub.empty:
            continue
        x = sub["shots"].to_numpy(dtype=float)
        tag = _snr_tag(snr)
        suffix = "" if (len(snrs) == 1 and math.isinf(snr)) else f"_snr{tag}"
        title = base_title + ("" if math.isinf(snr) else f" | SNR {tag}")

        # ---- Figure 1: MSE (left) / PSNR (right) ----
        mse = _col(sub, "mag_mse_mean")
        psnr = _col(sub, "mag_psnr_mean")
        mse_e = _col(sub, "mag_mse_std") if errorbars else None
        psnr_e = _col(sub, "mag_psnr_std") if errorbars else None
        ref_mse = (float(sub["classical_mag_mse"].iloc[0])
                   if classical and "classical_mag_mse" in sub.columns else None)
        ref_psnr = (float(sub["classical_mag_psnr"].iloc[0])
                    if classical and "classical_mag_psnr" in sub.columns else None)
        mse_label = "MSE"
        if binary and mse is not None:
            mse = mse * _BINARY_SCALE
            if mse_e is not None:
                mse_e = mse_e * _BINARY_SCALE
            if ref_mse is not None:
                ref_mse = ref_mse * _BINARY_SCALE
            mse_label = "MSE (0-255 scale)"
        if mse is not None and psnr is not None:
            p = outdir / f"{stem}_fig1_mse_psnr{suffix}.png"
            _dual_axis_figure(
                x, mse, psnr, mse_label, "PSNR (dB)",
                lambda v: f"{v:.3g}", lambda v: f"{v:.1f}",
                title, p, eL=mse_e, eR=psnr_e,
                refL=ref_mse, refR=ref_psnr,
                refL_label="classical MSE", refR_label="classical PSNR")
            written.append(p)

        # ---- Figure 2: cosine (left) / direction accuracy (right) ----
        cos = _col(sub, "cos_sim_median_mean")
        dacc = _col(sub, "dir_acc_pct_mean")
        cos_e = _col(sub, "cos_sim_median_std") if errorbars else None
        dacc_e = _col(sub, "dir_acc_pct_std") if errorbars else None
        if cos is not None and dacc is not None:
            p = outdir / f"{stem}_fig2_cos_diracc{suffix}.png"
            _dual_axis_figure(
                x, cos, dacc, "Cosine similarity", "Direction accuracy (%)",
                lambda v: f"{v:.3f}", lambda v: f"{v:.1f}",
                title, p, eL=cos_e, eR=dacc_e)
            written.append(p)

    return written


def main():
    ap = argparse.ArgumentParser(
        description="Dual-axis MSE/PSNR and Cos/DirAcc plots vs shots "
                    "from a QCED results xlsx.")
    ap.add_argument("xlsx", help="path to qced_results_<image>_<backend>.xlsx")
    ap.add_argument("--snr", default=None,
                    help="only this input SNR (e.g. 20, or 'clean'); "
                         "default: one figure pair per SNR present")
    ap.add_argument("--outdir", default=None,
                    help="output directory (default: alongside the xlsx)")
    ap.add_argument("--no-classical", action="store_true",
                    help="do not draw the classical reference lines")
    ap.add_argument("--errorbars", action="store_true",
                    help="draw +/-1 std error bars (uses the *_std columns)")
    args = ap.parse_args()

    written = make_plots(args.xlsx, outdir=args.outdir, snr_filter=args.snr,
                         classical=not args.no_classical,
                         errorbars=args.errorbars)
    if not written:
        print("No figures written (required metric columns missing).")
    for p in written:
        print(f"wrote {p}")


if __name__ == "__main__":
    main()