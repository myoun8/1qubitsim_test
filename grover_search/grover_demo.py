"""Walkthrough of Grover's search. Run: python grover_demo.py
Saves grover_success.png if matplotlib is installed."""

import numpy as np
from multi_qubit import fidelity
from grover import grover, optimal_iterations, theoretical_success, ideal_state

rng = np.random.default_rng(3)

print("1) Two qubits, searching for |10>")
reg = grover(2, ["10"], rng=rng)
print("   rounds used:  ", optimal_iterations(2))
print("   probabilities:", {k: round(v, 4) for k, v in reg.prob_dict().items()})

print("\n2) Bigger searches, single marked item")
print("   n   N    rounds  P(success)  classical avg guesses")
for n in range(2, 9):
    k = optimal_iterations(n)
    target = format(2 ** n - 1, f"0{n}b")
    p = grover(n, [target]).prob_dict().get(target, 0)
    print(f"   {n}  {2**n:4d}  {k:5d}   {p:9.4f}   {(2**n + 1) / 2:8.1f}")

print("\n3) 1000 measurements after searching 16 items for |1011>")
counts = grover(4, ["1011"], rng=rng).sample(1000)
top = sorted(counts.items(), key=lambda kv: -kv[1])[:4]
print("   most common outcomes:", top)

print("\n4) Too many rounds overshoots: success probability for n = 5, target 00111")
_, hist = grover(5, ["00111"], iterations=12, record=True)
for k, p in enumerate(hist):
    bar = "#" * int(round(p * 40))
    print(f"   k={k:2d}  {p:.4f}  {bar}")

print("\n5) Accuracy: simulated state vs. textbook state (fidelity)")
for k in range(4):
    sim = grover(4, ["0110"], iterations=k).state
    print(f"   k={k}: fidelity = {fidelity(sim, ideal_state(4, ['0110'], k)):.12f}")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    for n in [3, 4, 5, 6]:
        target = format(1, f"0{n}b")
        _, hist = grover(n, [target], iterations=15, record=True)
        ks = np.arange(len(hist))
        line, = ax.plot(ks, hist, "o", ms=4, label=f"n = {n} (N = {2**n})")
        fine = np.linspace(0, 15, 300)
        ax.plot(fine, [theoretical_success(n, 1, x) for x in fine],
                color=line.get_color(), lw=1, alpha=0.5)
    ax.set_xlabel("Grover iterations k")
    ax.set_ylabel("P(measure the marked item)")
    ax.set_title("Grover's search: simulation (dots) vs. sin²((2k+1)θ) (lines)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("grover_success.png", dpi=150)
    print("\nSaved plot to grover_success.png")
except ImportError:
    print("\n(matplotlib not installed, so no plot was saved)")
