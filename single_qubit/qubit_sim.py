"""
qubit_sim.py - a 1-qubit quantum simulator built with NumPy.

A qubit state is a complex vector |psi> = a|0> + b|1> with |a|^2 + |b|^2 = 1.
A gate is a 2x2 unitary matrix U, and applying it means |psi'> = U |psi>.
Measuring in the computational basis gives 0 with probability |a|^2
and 1 with probability |b|^2 (the Born rule).
"""

import numpy as np

TOL = 1e-10  # numerical tolerance for floating-point checks


# --------------------------------------------------------------------------
# Basis states
# --------------------------------------------------------------------------
KET_0 = np.array([1, 0], dtype=complex)
KET_1 = np.array([0, 1], dtype=complex)
KET_PLUS = np.array([1, 1], dtype=complex) / np.sqrt(2)
KET_MINUS = np.array([1, -1], dtype=complex) / np.sqrt(2)


# --------------------------------------------------------------------------
# Gates (2x2 unitary matrices)
# --------------------------------------------------------------------------
I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)              # bit flip (NOT)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)             # phase flip
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)  # Hadamard
S = np.array([[1, 0], [0, 1j]], dtype=complex)             # sqrt(Z)
T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)  # sqrt(S)


def Rx(theta):
    """Rotation by angle theta about the x-axis of the Bloch sphere."""
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


def Ry(theta):
    """Rotation by angle theta about the y-axis of the Bloch sphere."""
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def Rz(theta):
    """Rotation by angle theta about the z-axis of the Bloch sphere."""
    return np.array([[np.exp(-1j * theta / 2), 0],
                     [0, np.exp(1j * theta / 2)]], dtype=complex)


def is_unitary(U, tol=TOL):
    """Check that U^dagger U = I, i.e. the gate preserves total probability."""
    U = np.asarray(U, dtype=complex)
    return U.shape == (2, 2) and np.allclose(U.conj().T @ U, I, atol=tol)


# --------------------------------------------------------------------------
# The qubit
# --------------------------------------------------------------------------
class Qubit:
    """A single qubit stored as a normalized complex 2-vector."""

    def __init__(self, state=KET_0, rng=None):
        state = np.asarray(state, dtype=complex).reshape(2)
        norm = np.linalg.norm(state)
        if norm < TOL:
            raise ValueError("State vector cannot be zero.")
        self.state = state / norm          # always keep the state normalized
        self.history = []                  # names of gates applied so far
        self.rng = rng if rng is not None else np.random.default_rng()

    # ---- evolution -------------------------------------------------------
    def apply(self, U, name="U"):
        """Apply a unitary gate. Returns self so calls can be chained."""
        if not is_unitary(U):
            raise ValueError(f"Gate {name} is not unitary.")
        self.state = U @ self.state
        self.history.append(name)
        return self

    def x(self):  return self.apply(X, "X")
    def y(self):  return self.apply(Y, "Y")
    def z(self):  return self.apply(Z, "Z")
    def h(self):  return self.apply(H, "H")
    def s(self):  return self.apply(S, "S")
    def t(self):  return self.apply(T, "T")
    def rx(self, th): return self.apply(Rx(th), f"Rx({th:.3f})")
    def ry(self, th): return self.apply(Ry(th), f"Ry({th:.3f})")
    def rz(self, th): return self.apply(Rz(th), f"Rz({th:.3f})")

    # ---- measurement -----------------------------------------------------
    def probabilities(self):
        """Exact Born-rule probabilities [P(0), P(1)]."""
        return np.abs(self.state) ** 2

    def sample(self, shots=1000):
        """Simulate repeated measurements WITHOUT disturbing the state.
        Returns counts {'0': n0, '1': n1}, like a real quantum computer."""
        outcomes = self.rng.choice([0, 1], size=shots, p=self.probabilities())
        n1 = int(outcomes.sum())
        return {"0": shots - n1, "1": n1}

    def measure(self):
        """One projective measurement: returns 0 or 1 and COLLAPSES the state."""
        result = int(self.rng.choice([0, 1], p=self.probabilities()))
        self.state = KET_0.copy() if result == 0 else KET_1.copy()
        self.history.append(f"M->{result}")
        return result

    # ---- analysis --------------------------------------------------------
    def bloch_vector(self):
        """(x, y, z) coordinates on the Bloch sphere: the expectation values
        <X>, <Y>, <Z>. Pure states lie on the unit sphere."""
        psi = self.state
        return np.real([psi.conj() @ X @ psi,
                        psi.conj() @ Y @ psi,
                        psi.conj() @ Z @ psi])

    def density_matrix(self):
        """rho = |psi><psi|."""
        return np.outer(self.state, self.state.conj())

    def __repr__(self):
        a, b = self.state
        return f"({a:.4f})|0> + ({b:.4f})|1>"


# --------------------------------------------------------------------------
# Accuracy metrics (used to compare a simulated state with the ideal one)
# --------------------------------------------------------------------------
def _vec(s):
    return s.state if isinstance(s, Qubit) else np.asarray(s, dtype=complex)


def fidelity(psi, phi):
    """F = |<psi|phi>|^2 for pure states. 1 means identical (up to global phase)."""
    return float(np.abs(np.vdot(_vec(psi), _vec(phi))) ** 2)


def trace_distance(psi, phi):
    """D = (1/2) * sum |eigenvalues of (rho - sigma)|. 0 means identical.
    For pure states this equals sqrt(1 - F)."""
    a, b = _vec(psi), _vec(phi)
    diff = np.outer(a, a.conj()) - np.outer(b, b.conj())
    return float(0.5 * np.sum(np.abs(np.linalg.eigvalsh(diff))))
