"""Benchmark helpers for comparing DD sequences via subsystem trace distance."""

from __future__ import annotations
import numpy as np

from .basics import random_product_state, partial_trace_over_bath, trace_distance


def trace_distance_error(
    *,
    U0: np.ndarray,
    U: np.ndarray,
    rho0: np.ndarray,
    dS: int,
    dB: int,
) -> float:
    """Compute trace distance between reduced system states for ideal vs DD evolution."""
    rho_id = U0 @ rho0 @ U0.conj().T
    rho_dd = U  @ rho0 @ U.conj().T
    rhoS_id = partial_trace_over_bath(rho_id, dS=dS, dB=dB)
    rhoS_dd = partial_trace_over_bath(rho_dd, dS=dS, dB=dB)
    return trace_distance(rhoS_dd, rhoS_id)


def run_monte_carlo_compare(
    *,
    Ts: np.ndarray,
    num_samples: int,
    rng: np.random.Generator,
    dS: int,
    dB: int,
    unitary_our,     # callable(T, is_ideal: bool, rng)->U  (uses J inside closure)
    unitary_qdd,     # callable(T, is_ideal: bool, rng)->U
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Monte Carlo over random product pure initial states.

    Returns (mean_our, std_our, mean_qdd, std_qdd) each with shape (len(Ts),).
    """
    Ts = np.asarray(Ts, dtype=float)
    errs_our = np.empty((len(Ts), int(num_samples)), dtype=float)
    errs_qdd = np.empty((len(Ts), int(num_samples)), dtype=float)

    for i, T in enumerate(Ts):
        for s in range(int(num_samples)):
            rho0 = random_product_state(dS=dS, dB=dB, rng=rng)

            U0_our = unitary_our(float(T), True)
            U_our  = unitary_our(float(T), False)
            errs_our[i, s] = trace_distance_error(U0=U0_our, U=U_our, rho0=rho0, dS=dS, dB=dB)

            U0_qdd = unitary_qdd(float(T), True)
            U_qdd  = unitary_qdd(float(T), False)
            errs_qdd[i, s] = trace_distance_error(U0=U0_qdd, U=U_qdd, rho0=rho0, dS=dS, dB=dB)

    mean_our = errs_our.mean(axis=1)
    std_our  = errs_our.std(axis=1)
    mean_qdd = errs_qdd.mean(axis=1)
    std_qdd  = errs_qdd.std(axis=1)
    return mean_our, std_our, mean_qdd, std_qdd
