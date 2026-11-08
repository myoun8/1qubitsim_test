"""
multi_qubit.py - an n-qubit state-vector simulator built with NumPy.

An n-qubit state is a complex vector of length 2^n. Basis states are written
|q0 q1 ... q(n-1)>, with qubit 0 as the LEFTMOST bit (the textbook convention
in Nielsen & Chuang). Example for n = 2: index 0 = |00>, 1 = |01>, 2 = |10>, 3 = |11>.

A single-qubit gate U acting on qubit k is lifted to the whole register with
tensor (Kronecker) products:   I x ... x U x ... x I   (U in position k).
"""

from functools import reduce

import numpy as np

TOL = 1e-10  # numerical tolerance for floating-point checks

# Single-qubit gates (2x2 unitary matrices) used to build n-qubit operations
I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)                # bit flip (NOT)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)  # Hadamard


def kron_all(mats):
    """Tensor product of a list of matrices or vectors, left to right."""
    return reduce(np.kron, mats)


def lift(U, target, n):
    """Build the 2^n x 2^n matrix that applies 1-qubit gate U to qubit `target`."""
    return kron_all([U if k == target else I for k in range(n)])


def is_unitary(U, tol=TOL):
    U = np.asarray(U, dtype=complex)
    return np.allclose(U.conj().T @ U, np.eye(U.shape[0]), atol=tol)


class Register:
    """n qubits stored as one normalized complex vector of length 2^n."""

    def __init__(self, n, state=None, rng=None):
        self.n = n
        self.dim = 2 ** n
        if state is None:
            state = np.zeros(self.dim, dtype=complex)
            state[0] = 1                       # start in |00...0>
        state = np.asarray(state, dtype=complex).reshape(self.dim)
        self.state = state / np.linalg.norm(state)
        self.rng = rng if rng is not None else np.random.default_rng()

    @classmethod
    def from_bits(cls, bits, **kw):
        """Register in the basis state given by a bit string, e.g. '101'."""
        reg = cls(len(bits), **kw)
        reg.state[:] = 0
        reg.state[int(bits, 2)] = 1
        return reg

    # ---- evolution -------------------------------------------------------
    def apply(self, U, target=None):
        """Apply a 1-qubit gate to qubit `target`, or, if target is None,
        a full 2^n x 2^n unitary to the whole register."""
        M = U if target is None else lift(U, target, self.n)
        if M.shape != (self.dim, self.dim) or not is_unitary(M):
            raise ValueError("Gate must be unitary with the right size.")
        self.state = M @ self.state
        return self

    def h_all(self):
        """Hadamard on every qubit: H x H x ... x H."""
        return self.apply(kron_all([H] * self.n))

    # ---- measurement -----------------------------------------------------
    def label(self, index):
        return format(index, f"0{self.n}b")

    def probabilities(self):
        return np.abs(self.state) ** 2

    def prob_dict(self, threshold=1e-12):
        """{'00': p, '01': p, ...}, leaving out outcomes with ~zero probability."""
        return {self.label(i): float(p)
                for i, p in enumerate(self.probabilities()) if p > threshold}

    def sample(self, shots=1000):
        """Counts from repeated measurement of all qubits (state not changed)."""
        idx = self.rng.choice(self.dim, size=shots, p=self.probabilities())
        values, counts = np.unique(idx, return_counts=True)
        return {self.label(int(v)): int(c) for v, c in zip(values, counts)}

    def __repr__(self):
        terms = [f"({a:.4f})|{self.label(i)}>"
                 for i, a in enumerate(self.state) if abs(a) > 1e-9]
        return " + ".join(terms)


def fidelity(psi, phi):
    """F = |<psi|phi>|^2 for pure states. 1 means identical (up to global phase).
    Accepts state vectors or Registers."""
    a = psi.state if isinstance(psi, Register) else np.asarray(psi, dtype=complex)
    b = phi.state if isinstance(phi, Register) else np.asarray(phi, dtype=complex)
    return float(np.abs(np.vdot(a, b)) ** 2)
