"""
grover.py - Grover's quantum search algorithm (Nielsen & Chuang, Chapter 6).

Problem: among N = 2^n items, find the M "marked" ones. Classically this takes
about N/2 guesses on average; Grover needs about (pi/4) * sqrt(N/M) queries.

Algorithm:
  1. Start in |0...0> and apply H to every qubit -> uniform superposition |s>.
  2. Repeat k times (one "Grover iteration" G = D . O):
       O (oracle):    flip the sign of every marked basis state.
       D (diffusion): 2|s><s| - I, an "inversion about the mean".
  3. Measure. A marked item comes out with probability sin^2((2k+1) theta),
     where sin(theta) = sqrt(M/N).
"""

import numpy as np

from multi_qubit import Register, kron_all, H


def to_index(item):
    """Accept either an int or a bit string like '101'."""
    return int(item, 2) if isinstance(item, str) else int(item)


def oracle(n, marked):
    """Diagonal matrix: -1 on marked basis states, +1 everywhere else."""
    diag = np.ones(2 ** n, dtype=complex)
    for m in marked:
        diag[to_index(m)] = -1
    return np.diag(diag)


def diffusion(n):
    """D = 2|s><s| - I, built the way a circuit would: H^n (2|0><0| - I) H^n."""
    Hn = kron_all([H] * n)
    reflect_zero = -np.eye(2 ** n, dtype=complex)
    reflect_zero[0, 0] = 1                      # 2|0><0| - I
    return Hn @ reflect_zero @ Hn


def theta(n, num_marked):
    return np.arcsin(np.sqrt(num_marked / 2 ** n))


def optimal_iterations(n, num_marked=1):
    """Number of rounds that gets closest to success probability 1:
    k = round(pi / (4 theta) - 1/2)."""
    return max(0, int(np.round(np.pi / (4 * theta(n, num_marked)) - 0.5)))


def theoretical_success(n, num_marked, k):
    """Exact success probability after k rounds: sin^2((2k+1) theta)."""
    return float(np.sin((2 * k + 1) * theta(n, num_marked)) ** 2)


def ideal_state(n, marked, k):
    """The textbook state after k rounds, written down directly (no simulation):
    sin((2k+1)theta)|good> + cos((2k+1)theta)|bad>, where |good> and |bad> are
    uniform superpositions of the marked and unmarked states."""
    N, idx = 2 ** n, [to_index(m) for m in marked]
    M, t = len(idx), theta(n, len(idx))
    good = np.zeros(N, dtype=complex)
    good[idx] = 1 / np.sqrt(M)
    bad = np.full(N, 1 / np.sqrt(N - M), dtype=complex)
    bad[idx] = 0
    return np.sin((2 * k + 1) * t) * good + np.cos((2 * k + 1) * t) * bad


def grover(n, marked, iterations=None, rng=None, record=False):
    """Run Grover's search on n qubits. Returns the final Register; if
    record=True, also returns the success probability after each round."""
    marked = list(marked)
    k = optimal_iterations(n, len(marked)) if iterations is None else iterations
    G = diffusion(n) @ oracle(n, marked)        # one Grover iteration

    reg = Register(n, rng=rng).h_all()
    idx = [to_index(m) for m in marked]
    history = [float(reg.probabilities()[idx].sum())]
    for _ in range(k):
        reg.apply(G)
        history.append(float(reg.probabilities()[idx].sum()))
    return (reg, history) if record else reg
