import argparse
import json
import os
from typing import Dict, Tuple

import numpy as np
from scipy.stats import spearmanr

from knf_core import utils
from knf_core.multiwfn import run_multiwfn
from knf_core.nci_torch.engine import NCIConfig, run_nci_engine
from knf_core.nci_torch.molden import ANGSTROM_TO_BOHR, parse_molden
from knf_core.nci_torch.types import GridSpec


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a, b)[0, 1])


def _subset_metrics(
    sl2_ref: np.ndarray,
    rdg_ref: np.ndarray,
    sl2_pred: np.ndarray,
    rdg_pred: np.ndarray,
    mask: np.ndarray,
) -> Dict[str, float]:
    a1 = sl2_ref[mask]
    b1 = sl2_pred[mask]
    a2 = rdg_ref[mask]
    b2 = rdg_pred[mask]
    return {
        "n": int(mask.sum()),
        "pearson_sl2rho": _pearson(a1, b1),
        "pearson_rdg": _pearson(a2, b2),
        "spearman_sl2rho": float(spearmanr(a1, b1).correlation),
        "spearman_rdg": float(spearmanr(a2, b2).correlation),
        "mae_sl2rho": float(np.mean(np.abs(a1 - b1))),
        "mae_rdg": float(np.mean(np.abs(a2 - b2))),
        "rmse_sl2rho": float(np.sqrt(np.mean((a1 - b1) ** 2))),
        "rmse_rdg": float(np.sqrt(np.mean((a2 - b2) ** 2))),
    }


def _coords_to_bohr(xyz: np.ndarray, unit: str) -> np.ndarray:
    if unit == "bohr":
        return xyz
    if unit == "angstrom":
        return xyz * ANGSTROM_TO_BOHR
    raise ValueError("Unsupported coord unit. Use 'angstrom' or 'bohr'.")


def _build_grid_from_points(xyz_bohr: np.ndarray) -> Tuple[GridSpec, np.ndarray, np.ndarray, np.ndarray]:
    x = np.unique(xyz_bohr[:, 0])
    y = np.unique(xyz_bohr[:, 1])
    z = np.unique(xyz_bohr[:, 2])
    if x.size * y.size * z.size != xyz_bohr.shape[0]:
        raise RuntimeError("Multiwfn output does not form a complete regular Cartesian grid.")
    spacing = float(np.median(np.diff(x))) if x.size > 1 else 1.0
    return GridSpec(x_bohr=x, y_bohr=y, z_bohr=z, spacing_bohr=spacing), x, y, z


def _make_plots(
    outdir: str,
    sl2_ref: np.ndarray,
    rdg_ref: np.ndarray,
    sl2_pred: np.ndarray,
    rdg_pred: np.ndarray,
    sample_size: int,
    seed: int,
) -> Dict[str, str]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return {}

    n = sl2_ref.shape[0]
    if sample_size > 0 and n > sample_size:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=sample_size, replace=False)
    else:
        idx = np.arange(n)

    sl2_ref_s = sl2_ref[idx]
    rdg_ref_s = rdg_ref[idx]
    sl2_pred_s = sl2_pred[idx]
    rdg_pred_s = rdg_pred[idx]

    overlay_path = os.path.join(outdir, "scatter_overlay_rdg_vs_sl2rho.png")
    fig, ax = plt.subplots(figsize=(9, 7), dpi=150)
    ax.scatter(sl2_ref_s, rdg_ref_s, s=1, alpha=0.2, label="Multiwfn")
    ax.scatter(sl2_pred_s, rdg_pred_s, s=1, alpha=0.2, label="Custom Torch")
    ax.set_xlabel("sign(lambda2)rho")
    ax.set_ylabel("RDG")
    ax.set_title("NCI Scatter Overlay")
    ax.legend(markerscale=6)
    fig.tight_layout()
    fig.savefig(overlay_path)
    plt.close(fig)

    parity_path = os.path.join(outdir, "parity_multiwfn_vs_custom.png")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150)
    axes[0].hexbin(sl2_ref_s, sl2_pred_s, gridsize=120, bins="log")
    axes[0].set_xlabel("Multiwfn sign(lambda2)rho")
    axes[0].set_ylabel("Custom sign(lambda2)rho")
    axes[0].set_title("Parity: sign(lambda2)rho")
    axes[1].hexbin(rdg_ref_s, rdg_pred_s, gridsize=120, bins="log")
    axes[1].set_xlabel("Multiwfn RDG")
    axes[1].set_ylabel("Custom RDG")
    axes[1].set_title("Parity: RDG")
    fig.tight_layout()
    fig.savefig(parity_path)
    plt.close(fig)

    # Focused scatter in low-RDG regime where NCI interpretation is usually done.
    low_mask = (rdg_ref_s <= 2.0) & (rdg_pred_s <= 2.0)
    low_path = os.path.join(outdir, "scatter_overlay_low_rdg_le_2.png")
    if np.any(low_mask):
        fig, ax = plt.subplots(figsize=(9, 7), dpi=150)
        ax.scatter(sl2_ref_s[low_mask], rdg_ref_s[low_mask], s=2, alpha=0.35, label="Multiwfn")
        ax.scatter(sl2_pred_s[low_mask], rdg_pred_s[low_mask], s=2, alpha=0.35, label="Custom Torch")
        ax.set_xlabel("sign(lambda2)rho")
        ax.set_ylabel("RDG")
        ax.set_title("NCI Scatter Overlay (RDG <= 2)")
        ax.legend(markerscale=5)
        fig.tight_layout()
        fig.savefig(low_path)
        plt.close(fig)
    else:
        low_path = ""

    # Quantile-clipped parity to reduce outlier domination.
    q_path = os.path.join(outdir, "parity_quantile_clipped_99_9.png")
    q = 0.999
    smax_ref = np.quantile(np.abs(sl2_ref_s), q)
    smax_pred = np.quantile(np.abs(sl2_pred_s), q)
    rmax_ref = np.quantile(rdg_ref_s, q)
    rmax_pred = np.quantile(rdg_pred_s, q)
    qmask = (
        (np.abs(sl2_ref_s) <= smax_ref)
        & (np.abs(sl2_pred_s) <= smax_pred)
        & (rdg_ref_s <= rmax_ref)
        & (rdg_pred_s <= rmax_pred)
    )
    if np.any(qmask):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150)
        axes[0].hexbin(sl2_ref_s[qmask], sl2_pred_s[qmask], gridsize=120, bins="log")
        axes[0].set_xlabel("Multiwfn sign(lambda2)rho")
        axes[0].set_ylabel("Custom sign(lambda2)rho")
        axes[0].set_title("Parity: sign(lambda2)rho (99.9% clipped)")
        axes[1].hexbin(rdg_ref_s[qmask], rdg_pred_s[qmask], gridsize=120, bins="log")
        axes[1].set_xlabel("Multiwfn RDG")
        axes[1].set_ylabel("Custom RDG")
        axes[1].set_title("Parity: RDG (99.9% clipped)")
        fig.tight_layout()
        fig.savefig(q_path)
        plt.close(fig)
    else:
        q_path = ""

    return {
        "overlay_plot": overlay_path,
        "parity_plot": parity_path,
        "overlay_low_rdg_plot": low_path,
        "parity_quantile_clipped_plot": q_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Multiwfn NCI output against custom torch NCI engine.")
    parser.add_argument("--molden", default="molden.input", help="Path to molden.input")
    parser.add_argument("--outdir", default="nci_compare", help="Output directory for compare artifacts")
    parser.add_argument("--skip-multiwfn", action="store_true", help="Skip Multiwfn execution and reuse existing output")
    parser.add_argument(
        "--multiwfn-output-name",
        default="output.txt",
        help="Expected Multiwfn output grid filename inside outdir",
    )
    parser.add_argument(
        "--custom-output-name",
        default="custom_torch_output_units_fixed.txt",
        help="Filename for custom torch output in outdir",
    )
    parser.add_argument(
        "--mw-coord-unit",
        choices=["angstrom", "bohr"],
        default="angstrom",
        help="Coordinate unit used by Multiwfn output grid",
    )
    parser.add_argument("--device", default="auto", help="Torch device: auto/cuda/cpu")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--batch-size", type=int, default=120000)
    parser.add_argument("--rho-floor", type=float, default=1e-12)
    parser.add_argument(
        "--apply-primitive-norm",
        action="store_true",
        help="Apply primitive normalization in Molden parser (default: off)",
    )
    parser.add_argument("--rdg-low-threshold", type=float, default=2.0)
    parser.add_argument("--trim-quantile", type=float, default=0.999)
    parser.add_argument("--plot-sample-size", type=int, default=250000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    molden_path = os.path.abspath(args.molden)
    outdir = os.path.abspath(args.outdir)
    os.makedirs(outdir, exist_ok=True)
    mw_path = os.path.join(outdir, args.multiwfn_output_name)
    custom_path = os.path.join(outdir, args.custom_output_name)
    summary_path = os.path.join(outdir, "correlation_summary.json")
    subsets_path = os.path.join(outdir, "correlation_subsets.json")

    utils.ensure_multiwfn_in_path()
    if not args.skip_multiwfn:
        run_multiwfn(molden_path, outdir)
        if not os.path.exists(mw_path):
            raise RuntimeError(f"Multiwfn completed but expected output not found: {mw_path}")
    elif not os.path.exists(mw_path):
        raise RuntimeError(f"--skip-multiwfn set but output file missing: {mw_path}")

    mw = np.loadtxt(mw_path)
    xyz = mw[:, :3]
    sl2_mw = mw[:, 3]
    rdg_mw = mw[:, 4]

    xyz_bohr = _coords_to_bohr(xyz, args.mw_coord_unit)
    grid, x, y, z = _build_grid_from_points(xyz_bohr)

    wf = parse_molden(
        molden_path,
        apply_primitive_normalization=args.apply_primitive_norm,
    )
    fields, device = run_nci_engine(
        wavefunction=wf,
        grid=grid,
        config=NCIConfig(
            device=args.device,
            dtype=args.dtype,
            batch_size=args.batch_size,
            rho_floor=args.rho_floor,
        ),
    )
    sl2_custom = fields.sign_lambda2_rho.detach().cpu().numpy()
    rdg_custom = fields.rdg.detach().cpu().numpy()

    ix = np.searchsorted(x, xyz_bohr[:, 0])
    iy = np.searchsorted(y, xyz_bohr[:, 1])
    iz = np.searchsorted(z, xyz_bohr[:, 2])
    sl2_pred = sl2_custom[ix, iy, iz]
    rdg_pred = rdg_custom[ix, iy, iz]

    out = np.column_stack([xyz, sl2_pred, rdg_pred])
    np.savetxt(custom_path, out, fmt="%.8f %.8f %.8f %.10e %.10e")

    all_mask = np.ones(sl2_mw.shape[0], dtype=bool)
    attractive_mask = sl2_mw < 0
    trim_q = float(args.trim_quantile)
    rdg_q = np.quantile(rdg_mw, trim_q)
    sl2_q = np.quantile(np.abs(sl2_mw), trim_q)
    trimmed_mask = (rdg_mw <= rdg_q) & (np.abs(sl2_mw) <= sl2_q)
    low_rdg_mask = rdg_mw <= float(args.rdg_low_threshold)

    subsets = {
        "all_points": _subset_metrics(sl2_mw, rdg_mw, sl2_pred, rdg_pred, all_mask),
        "attractive_points_sl2rho_lt_0": _subset_metrics(
            sl2_mw, rdg_mw, sl2_pred, rdg_pred, attractive_mask
        ),
        "trimmed": _subset_metrics(sl2_mw, rdg_mw, sl2_pred, rdg_pred, trimmed_mask),
        "low_rdg": _subset_metrics(sl2_mw, rdg_mw, sl2_pred, rdg_pred, low_rdg_mask),
    }
    subsets["trimmed"]["quantile"] = trim_q
    subsets["low_rdg"]["threshold"] = float(args.rdg_low_threshold)

    summary = {
        "n_points": int(xyz.shape[0]),
        "grid_shape": [int(x.size), int(y.size), int(z.size)],
        "device_used": str(device),
        "assumption": f"Multiwfn coordinates treated as {args.mw_coord_unit}",
        "apply_primitive_normalization": bool(args.apply_primitive_norm),
        "files": {
            "multiwfn_output": mw_path,
            "custom_output": custom_path,
        },
        "all_points": subsets["all_points"],
    }
    summary["plots"] = _make_plots(
        outdir=outdir,
        sl2_ref=sl2_mw,
        rdg_ref=rdg_mw,
        sl2_pred=sl2_pred,
        rdg_pred=rdg_pred,
        sample_size=args.plot_sample_size,
        seed=args.seed,
    )

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(subsets_path, "w", encoding="utf-8") as f:
        json.dump(subsets, f, indent=2)

    print(json.dumps(summary, indent=2))
    print("Detailed subsets written to:", subsets_path)


if __name__ == "__main__":
    main()
