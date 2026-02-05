"""Pauli-frame definitions and small utilities for least-squares DD timing optimization."""

from __future__ import annotations
import numpy as np

# -----------------------------
# Pauli matrices (system)
# -----------------------------
I2 = np.eye(2, dtype=complex)
X  = np.array([[0, 1], [1, 0]], dtype=complex)
Y  = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z  = np.array([[1, 0], [0, -1]], dtype=complex)


def kron(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Kronecker product."""
    return np.kron(A, B)


# -----------------------------
# Frames and signs (I, X, Y, Z)
# -----------------------------
FRAME_TO_IDX = {"I": 0, "X": 1, "Y": 2, "Z": 3}
IDX_TO_FRAME = ["I", "X", "Y", "Z"]

# Rows correspond to I, X, Y, Z (same convention as your original notebook)
# Columns correspond to (x, y, z) toggling signs.
FRAME_SIGNS = np.array(
    [
        [ 1.0,  1.0,  1.0],  # I
        [ 1.0, -1.0, -1.0],  # X
        [-1.0,  1.0, -1.0],  # Y
        [-1.0, -1.0,  1.0],  # Z
    ],
    dtype=float,
)


def softmax(z: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    z = np.asarray(z, dtype=float)
    z = z - np.max(z)
    e = np.exp(z)
    return e / np.sum(e)


def pattern_frames(L: int, cycle: tuple[str, ...] = ("I", "X", "Y", "Z")) -> list[str]:
    """Convenience: produce a repeating frame pattern of length L."""
    cyc = list(cycle)
    return [cyc[i % len(cyc)] for i in range(L)]
