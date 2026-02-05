"""Error metrics, slope estimation, sweep helpers, and plotting."""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt


def op_norm(A: np.ndarray) -> float:
    """Operator (spectral) norm."""
    return float(np.linalg.norm(A, 2))


def fit_loglog_slope(xs, ys, drop_edges: int = 3) -> float:
    """Fit slope of log(ys) vs log(xs), dropping a few edge points."""
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    lo, hi = int(drop_edges), len(xs) - int(drop_edges)
    x = np.log(xs[lo:hi])
    y = np.log(np.maximum(ys[lo:hi], 1e-300))
    a, _ = np.polyfit(x, y, 1)
    return float(a)


def sweep_error_operator_norm(seq, durs, Ts, Ufree_0, Ufree_J, pulses, unitary_fn) -> np.ndarray:
    """Compute ||U(T)-U0(T)|| for each T in Ts."""
    Ts = np.asarray(Ts, dtype=float)
    errs = np.empty_like(Ts)
    for i, T in enumerate(Ts):
        U0 = unitary_fn(seq, durs, float(T), Ufree_0, pulses)
        U  = unitary_fn(seq, durs, float(T), Ufree_J, pulses)
        errs[i] = op_norm(U - U0)
    return errs


def reference_power_line(errs, Ts, power: int, anchor_index: int | None = None) -> np.ndarray:
    """Reference ~ T^power line anchored at anchor_index (default: midpoint)."""
    errs = np.asarray(errs, dtype=float)
    Ts = np.asarray(Ts, dtype=float)
    if anchor_index is None:
        anchor_index = len(Ts) // 2
    c = errs[anchor_index] / (Ts[anchor_index] ** power)
    return c * (Ts ** power)


def plot_errors(Ts, errs_by_k: dict[int, np.ndarray], ref_by_k: dict[int, np.ndarray] | None = None,
                k_list: list[int] | None = None, figsize=(6.5, 4.5)) -> None:
    """Log-log plot of errors; optionally overlay reference lines."""
    if k_list is None:
        k_list = sorted(errs_by_k.keys())

    plt.figure(figsize=figsize)
    for k in k_list:
        plt.loglog(Ts, errs_by_k[k], marker="o", label=rf"$K={k}$")
        if ref_by_k is not None and k in ref_by_k:
            plt.loglog(Ts, ref_by_k[k], alpha=0.25)

    plt.xlabel(r"$T$")
    plt.ylabel(r"$||U(T)-U_0(T)||$")
    plt.legend(loc="lower right", ncol=2, frameon=False)
    plt.tight_layout()
