"""UDD/QDD pulse-time builders and a compiled QDD propagator."""

from __future__ import annotations
import numpy as np


def udd_relative_times(N: int) -> np.ndarray:
    """UDD relative pulse times in (0,1): t_j/T = sin^2(pi j/(2N+2))."""
    N = int(N)
    if N <= 0:
        return np.array([], dtype=float)
    j = np.arange(1, N + 1, dtype=float)
    return np.sin(np.pi * j / (2 * N + 2)) ** 2


def build_qdd_pulse_list_rel(Nx: int, Nz: int) -> list[tuple[float, str]]:
    """Return sorted list of (t_rel, axis) for QDD in normalized time [0,1]."""
    Nx = int(Nx); Nz = int(Nz)
    tZ_rel = udd_relative_times(Nz)  # Z pulses in (0,1)
    pulses = [(float(t), 'Z') for t in tZ_rel]

    boundaries = np.concatenate(([0.0], tZ_rel, [1.0]))  # intervals between Z pulses
    tX_rel = udd_relative_times(Nx)

    for a, b in zip(boundaries[:-1], boundaries[1:]):
        dt = float(b - a)
        for u in tX_rel:
            pulses.append((float(a + float(u) * dt), 'X'))

    pulses.sort(key=lambda x: x[0])
    return pulses


def propagator_qdd_compiled(
    *,
    Ufree,             # callable(dt)->U
    pulseX: np.ndarray,
    pulseZ: np.ndarray,
    T: float,
    pulse_list_rel: list[tuple[float, str]],
) -> np.ndarray:
    """
    QDD propagator under ideal instantaneous pulses.
    Evolves with constant lab Hamiltonian via Ufree(dt) between pulse times.
    """
    d = pulseX.shape[0]
    U = np.eye(d, dtype=complex)
    last = 0.0
    T = float(T)

    for t_rel, axis in pulse_list_rel:
        tp = float(t_rel) * T
        dt = tp - last
        if dt > 0:
            U = Ufree(dt) @ U
        U = (pulseX if axis == 'X' else pulseZ) @ U
        last = tp

    dt_final = T - last
    if dt_final > 0:
        U = Ufree(dt_final) @ U

    return U
