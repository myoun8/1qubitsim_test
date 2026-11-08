"""Checks the multi-qubit layer and Grover's search. Run: python test_grover.py"""

import numpy as np
from multi_qubit import Register, lift, is_unitary, X, H, fidelity
from grover import (oracle, diffusion, optimal_iterations, theoretical_success,
                    ideal_state, grover)


def test_tensor_lift():
    # X on qubit 0 of |00> gives |10>; X on qubit 1 gives |01>
    assert Register(2).apply(X, 0).prob_dict() == {"10": 1.0}
    assert Register(2).apply(X, 1).prob_dict() == {"01": 1.0}
    assert lift(H, 1, 3).shape == (8, 8)


def test_uniform_superposition():
    p = Register(3).h_all().probabilities()
    assert np.allclose(p, 1 / 8)


def test_oracle_and_diffusion_unitary():
    for n in [2, 3, 4]:
        assert is_unitary(oracle(n, [1]))
        assert is_unitary(diffusion(n))


def test_two_qubits_one_round_is_exact():
    # For N = 4, a single Grover round finds the marked item with certainty
    for target in ["00", "01", "10", "11"]:
        reg = grover(2, [target])
        assert optimal_iterations(2) == 1
        assert abs(reg.prob_dict().get(target, 0) - 1) < 1e-12


def test_matches_theory_every_round():
    for n, marked in [(3, ["101"]), (4, ["0110"]), (5, ["10011", "00001"])]:
        _, hist = grover(n, marked, iterations=8, record=True)
        for k, p in enumerate(hist):
            assert abs(p - theoretical_success(n, len(marked), k)) < 1e-9


def test_fidelity_with_ideal_state():
    for n, marked in [(3, ["110"]), (4, ["0001", "1110"])]:
        for k in range(5):
            reg = grover(n, marked, iterations=k)
            assert fidelity(reg.state, ideal_state(n, marked, k)) > 1 - 1e-9


def test_finds_target_by_sampling():
    reg = grover(4, ["1011"], rng=np.random.default_rng(1))
    counts = reg.sample(2000)
    assert max(counts, key=counts.get) == "1011"


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\nAll {len(tests)} tests passed.")
