"""Small quantum-information utilities (states, partial trace, trace distance)."""

from __future__ import annotations
import numpy as np


def random_pure_state(dim: int, rng: np.random.Generator) -> np.ndarray:
    """Haar-random pure state |psi> in C^dim."""
    v = rng.normal(size=dim) + 1j * rng.normal(size=dim)
    v = v / np.linalg.norm(v)
    return v


def random_product_state(dS: int, dB: int, rng: np.random.Generator) -> np.ndarray:
    """|psi>_S ⊗ |phi>_B as a density matrix rho_SB."""
    psi = random_pure_state(dS, rng)
    phi = random_pure_state(dB, rng)
    psi_SB = np.kron(psi, phi)
    rho = np.outer(psi_SB, psi_SB.conj())
    return rho


def initial_product_state_zero(dS: int, dB: int) -> np.ndarray:
    """|0>_S ⊗ |0>_B as a density matrix rho_SB."""
    ketS = np.zeros(dS, dtype=complex); ketS[0] = 1.0
    ketB = np.zeros(dB, dtype=complex); ketB[0] = 1.0
    ket = np.kron(ketS, ketB)
    return np.outer(ket, ket.conj())


def partial_trace_over_bath(rhoSB: np.ndarray, dS: int, dB: int) -> np.ndarray:
    """Trace out the bath from rhoSB of shape (dS*dB, dS*dB)."""
    rhoSB = np.asarray(rhoSB, dtype=complex)
    if rhoSB.shape != (dS * dB, dS * dB):
        raise ValueError(f"rhoSB has shape {rhoSB.shape}, expected {(dS*dB, dS*dB)}")
    # reshape indices: (s,b; s',b') -> (s,b,s',b')
    rho = rhoSB.reshape(dS, dB, dS, dB)
    # trace over bath: sum_b rho[s,b,s',b]
    return np.einsum("abcb->ac", rho)


def trace_distance(rho: np.ndarray, sigma: np.ndarray) -> float:
    """Trace distance D(rho,sigma) = 1/2 ||rho - sigma||_1."""
    rho = np.asarray(rho, dtype=complex)
    sigma = np.asarray(sigma, dtype=complex)
    delta = rho - sigma
    # delta should be Hermitian if rho,sigma are Hermitian; symmetrize for numerical stability
    deltaH = (delta + delta.conj().T) / 2.0
    evals = np.linalg.eigvalsh(deltaH)
    return 0.5 * float(np.sum(np.abs(evals)))
