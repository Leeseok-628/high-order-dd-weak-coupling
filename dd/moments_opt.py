"""Moment-cancellation residuals and least-squares timing optimization."""

from __future__ import annotations
import numpy as np
from scipy.optimize import least_squares

from .pauli_frames import FRAME_TO_IDX, FRAME_SIGNS, softmax


def moment_residuals(theta: np.ndarray, k: int, T: float, s_mat: np.ndarray) -> np.ndarray:
    """
    Residual vector in R^{3k}. Durations are positive and sum to T via softmax(theta).

    Parameters
    ----------
    theta : (L,) unconstrained parameters
    k     : cancellation order K (enforce moments m=0..K-1)
    T     : total duration
    s_mat : (L,3) sign vectors for each segment

    Returns
    -------
    r : (3k,) residuals stacked as (m=1..k, alpha=x,y,z).
    """
    durs = float(T) * softmax(theta)  # (L,)
    t = np.empty(durs.size + 1, dtype=float)
    t[0] = 0.0
    np.cumsum(durs, out=t[1:])
    t[-1] = float(T)  # guard exact endpoint

    # powers 1..k (constant scaling 1/p omitted since constraints target 0)
    powers = np.arange(1, k + 1, dtype=float)  # (k,)
    t_pows = t[:, None] ** powers[None, :]     # (L+1, k)
    delta = t_pows[1:] - t_pows[:-1]           # (L, k)
    acc = delta.T @ s_mat                      # (k, 3)

    return acc.ravel()


def solve_durations_for_frames(
    *,
    k: int,
    T: float,
    frames: list[str],
    restarts: int = 6,
    seed: int = 0,
    max_nfev: int = 40000,
    tol: float = 1e-14,
) -> tuple[np.ndarray, float]:
    """
    For a fixed frame pattern, optimize durations to solve 3k moment equations.

    Returns
    -------
    durs_frac : (L,) durations as fractions of T (sum to 1)
    best_resn : residual 2-norm at the best restart
    """
    L = len(frames)
    frames_idx = np.fromiter((FRAME_TO_IDX[f] for f in frames), dtype=int, count=L)
    s_mat = FRAME_SIGNS[frames_idx]  # (L,3)

    res_fun = lambda th: moment_residuals(th, k, T, s_mat)

    rng = np.random.default_rng(seed)
    best_x = None
    best_resn = np.inf

    for _ in range(int(restarts)):
        th0 = rng.standard_normal(L) * 0.5
        sol = least_squares(
            res_fun,
            th0,
            method="trf",
            ftol=tol, xtol=tol, gtol=tol,
            max_nfev=int(max_nfev),
        )
        resn = float(np.linalg.norm(res_fun(sol.x)))
        if resn < best_resn:
            best_resn = resn
            best_x = sol.x

    durs_frac = softmax(best_x)
    return durs_frac, best_resn


def build_solutions(
    *,
    k_min: int = 1,
    k_max: int = 8,
    T_norm: float = 1.0,
    frame_cycle: tuple[str, ...] = ("I", "X", "Y", "Z"),
    restarts: int = 8,
    seed_base: int = 100,
    max_nfev: int = 100000,
    compile_sequence_fn=None,
    pattern_frames_fn=None,
    verbose: bool = True,
) -> dict[int, dict]:
    """
    Helper to generate solutions for a range of K.

    You can pass compile_sequence_fn / pattern_frames_fn to avoid circular imports.
    Expected signature:
      compile_sequence_fn(frames)->seqdict
      pattern_frames_fn(L, cycle)->frameslist
    """
    if compile_sequence_fn is None or pattern_frames_fn is None:
        raise ValueError("Please pass compile_sequence_fn and pattern_frames_fn")

    sols: dict[int, dict] = {}
    for k in range(int(k_min), int(k_max) + 1):
        L = 3 * k + 1
        frames = pattern_frames_fn(L, frame_cycle)
        durs_frac, resn = solve_durations_for_frames(
            k=k, T=T_norm, frames=frames,
            restarts=restarts, seed=seed_base + k, max_nfev=max_nfev
        )
        sols[k] = {
            "k": k,
            "L": L,
            "frames": frames,
            "seq": compile_sequence_fn(frames),
            "durs_frac": durs_frac,
            "residual_norm": resn,
        }
        if verbose:
            print(f"[k={k}] L={L}, residual_norm={resn:.3e}, sum(durs)={durs_frac.sum():.12f}")
    return sols
