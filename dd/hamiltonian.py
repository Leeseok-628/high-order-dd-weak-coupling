"""Hamiltonian model and fast matrix exponential evolver."""

from __future__ import annotations
import numpy as np

from .pauli_frames import I2, X, Y, Z, kron


def rand_hermitian(n: int, rng: np.random.Generator) -> np.ndarray:
    """Random Hermitian normalized to spectral radius 1."""
    A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    H = (A + A.conj().T) / 2.0
    w = np.linalg.eigvalsh(H)
    return H / np.max(np.abs(w))


def H_lab(J: float, HB: np.ndarray, Bx: np.ndarray, By: np.ndarray, Bz: np.ndarray) -> np.ndarray:
    """System-bath lab Hamiltonian: I⊗HB + J( X⊗Bx + Y⊗By + Z⊗Bz )."""
    return kron(I2, HB) + float(J) * (kron(X, Bx) + kron(Y, By) + kron(Z, Bz))


def make_free_evolver(H: np.ndarray):
    """
    Returns U(dt) = exp(-i H dt) using a single eigh decomposition.
    H must be Hermitian.
    """
    w, V = np.linalg.eigh(H)
    Vh = V.conj().T

    def U(dt: float) -> np.ndarray:
        ph = np.exp(-1j * w * float(dt))
        return (V * ph[None, :]) @ Vh

    return U
