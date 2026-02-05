"""Pulse compilation and unitary simulation under instantaneous π-pulses."""

from __future__ import annotations
import numpy as np

from .pauli_frames import FRAME_TO_IDX, IDX_TO_FRAME, FRAME_SIGNS, X, Y, Z, kron


def build_pulses(dimB: int) -> list[np.ndarray]:
    """π-pulses on the system: X⊗I, Y⊗I, Z⊗I."""
    IB = np.eye(int(dimB), dtype=complex)
    return [
        kron(X, IB),  # x
        kron(Y, IB),  # y
        kron(Z, IB),  # z
    ]


def build_transition_axis() -> np.ndarray:
    """
    Axis codes: -1 means no pulse, 0->x, 1->y, 2->z.
    Between different frames, exactly two signs flip; pulse is about the remaining axis.
    """
    trans = -np.ones((4, 4), dtype=int)
    for a in range(4):
        for b in range(4):
            if a == b:
                trans[a, b] = -1
                continue
            flips = (FRAME_SIGNS[a] != FRAME_SIGNS[b])
            if flips.sum() != 2:
                raise ValueError(f"Invalid transition {IDX_TO_FRAME[a]}->{IDX_TO_FRAME[b]}")
            axis = int(np.where(~flips)[0][0])
            trans[a, b] = axis
    return trans


TRANS_AXIS = build_transition_axis()


def compile_sequence(frames: list[str]) -> dict:
    """Precompute boundary pulse axes for a given frame list."""
    frames_idx = np.fromiter((FRAME_TO_IDX[f] for f in frames), dtype=int, count=len(frames))

    init_axis = TRANS_AXIS[0, frames_idx[0]] if (frames_idx.size and frames_idx[0] != 0) else -1
    between_axes = np.array(
        [TRANS_AXIS[frames_idx[i], frames_idx[i + 1]] for i in range(frames_idx.size - 1)],
        dtype=int
    ) if frames_idx.size > 1 else np.array([], dtype=int)
    final_axis = TRANS_AXIS[frames_idx[-1], 0] if (frames_idx.size and frames_idx[-1] != 0) else -1

    return {
        "frames_idx": frames_idx,
        "init_axis": int(init_axis),
        "between_axes": between_axes,
        "final_axis": int(final_axis),
    }


def unitary_sequence_pulse_compiled(
    seq: dict,
    durs_frac: np.ndarray,
    T: float,
    free_evolver,
    pulses: list[np.ndarray],
) -> np.ndarray:
    """
    Apply boundary π-pulses (instantaneous), and free-evolve under fixed lab H.
    durs_frac sums to 1; each segment uses dt = durs_frac[i] * T.
    """
    d = pulses[0].shape[0]
    U = np.eye(d, dtype=complex)

    ax0 = int(seq["init_axis"])
    if ax0 != -1:
        U = pulses[ax0] @ U

    between_axes = seq["between_axes"]
    for i, frac in enumerate(durs_frac):
        dt = float(frac) * float(T)
        if dt > 0:
            U = free_evolver(dt) @ U
        if i < between_axes.size:
            ax = int(between_axes[i])
            if ax != -1:
                U = pulses[ax] @ U

    axf = int(seq["final_axis"])
    if axf != -1:
        U = pulses[axf] @ U

    return U
