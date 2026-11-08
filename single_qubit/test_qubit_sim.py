"""Checks the simulator against results worked out by hand. Run: python test_qubit_sim.py"""

import numpy as np
from qubit_sim import (Qubit, KET_0, KET_1, KET_PLUS, KET_MINUS,
                       I, X, Y, Z, H, S, T, Rx, Ry, Rz,
                       is_unitary, fidelity, trace_distance)


def close(a, b):
    return np.allclose(a, b, atol=1e-9)


def test_all_gates_unitary():
    for U in [I, X, Y, Z, H, S, T, Rx(0.7), Ry(1.3), Rz(2.1)]:
        assert is_unitary(U)
    assert not is_unitary(np.array([[1, 1], [0, 1]]))


def test_basic_gate_actions():
    assert close(Qubit().x().state, KET_1)                 # X|0> = |1>
    assert close(Qubit().h().state, KET_PLUS)              # H|0> = |+>
    assert close(Qubit(KET_1).h().state, KET_MINUS)        # H|1> = |->
    assert close(Qubit().h().h().state, KET_0)             # H H = I
    assert fidelity(Qubit(KET_PLUS).z(), KET_MINUS) > 1 - 1e-9  # Z|+> = |->


def test_gate_identities():
    assert close(S @ S, Z)                                 # S^2 = Z
    assert close(T @ T, S)                                 # T^2 = S
    assert close(H @ X @ H, Z)                             # HXH = Z
    # Rx(pi) equals X up to a global phase of -i
    assert close(Rx(np.pi), -1j * X)


def test_probabilities():
    assert close(Qubit().probabilities(), [1, 0])
    assert close(Qubit().h().probabilities(), [0.5, 0.5])
    # Ry(theta)|0> gives P(1) = sin^2(theta/2)
    th = 1.0
    assert close(Qubit().ry(th).probabilities()[1], np.sin(th / 2) ** 2)


def test_sampling_matches_probabilities():
    q = Qubit(rng=np.random.default_rng(42)).ry(np.pi / 3)   # P(1) = 0.25
    counts = q.sample(shots=20000)
    assert abs(counts["1"] / 20000 - 0.25) < 0.02


def test_measure_collapses():
    q = Qubit(rng=np.random.default_rng(0)).h()
    r = q.measure()
    assert close(q.state, KET_0 if r == 0 else KET_1)
    assert q.measure() == r                                # repeated measurement agrees


def test_bloch_vectors():
    assert close(Qubit().bloch_vector(), [0, 0, 1])        # |0> = north pole
    assert close(Qubit(KET_1).bloch_vector(), [0, 0, -1])
    assert close(Qubit().h().bloch_vector(), [1, 0, 0])    # |+> on +x axis
    assert close(Qubit().h().s().bloch_vector(), [0, 1, 0])  # |+i> on +y axis


def test_metrics():
    assert abs(fidelity(KET_0, KET_0) - 1) < 1e-12
    assert abs(fidelity(KET_0, KET_1)) < 1e-12
    assert abs(fidelity(KET_0, KET_PLUS) - 0.5) < 1e-12
    assert abs(trace_distance(KET_0, KET_1) - 1) < 1e-12
    # global phase does not matter
    assert abs(fidelity(KET_PLUS, -1j * KET_PLUS) - 1) < 1e-12
    # pure-state relation D = sqrt(1 - F)
    a, b = Qubit().ry(0.4).state, Qubit().rx(1.1).state
    assert abs(trace_distance(a, b) - np.sqrt(1 - fidelity(a, b))) < 1e-9


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\nAll {len(tests)} tests passed.")
