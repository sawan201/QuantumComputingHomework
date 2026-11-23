"""PennyLane experiments for 'Permutation symmetry in quantum circuits'.

Python 3.12. Reproduce Table 1 with
    python -m pip install pennylane==0.45.1 pennylane-lightning==0.45.0 numpy==2.3.5
    python symmetric_circuits.py

X holds the logical input; Y is the ordered workspace on which the ordinary
circuit acts. Two reversible selectors implement the same controlled routing.
The small-instance weight-one test below is expanded into positive-control
gates. The paper counts a reversible threshold gate of any arity as one gate.
"""

import itertools
import math
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import pennylane as qml


def toggle_weight_one(wires, flag):
    """XOR [weight = 1] into flag, using ANDs with a common target.

    On an input of weight k>0 there are 2**(k-1) odd-sized subsets of
    the active wires. Their parity is one precisely when k=1.
    """
    for size in range(1, len(wires) + 1, 2):
        for subset in itertools.combinations(wires, size):
            qml.MultiControlledX(wires=[*subset, flag])


def prepare_choice(q, available, r, coin, flag):
    """Uniformly select one of the r available wires. None means all available."""
    if r == 1:
        if available is None:
            qml.PauliX(q[0])
        else:
            for control, target in zip(available, q):
                qml.CNOT([control, target])
        return

    p = (1 - 1 / r) ** (r - 1)
    theta = 2 * math.asin(1 / math.sqrt(r))
    coin_angle = 2 * math.asin(1 / (2 * math.sqrt(p)))

    def rotations():
        for i, target in enumerate(q):
            if available is None:
                qml.RY(theta, wires=target)
            else:
                qml.ctrl(qml.RY, control=available[i])(theta, wires=target)
        qml.RY(coin_angle, wires=coin)

    rotations()
    # First change the sign exactly when the selector has weight one and coin=1.
    toggle_weight_one(q, flag)
    qml.CZ([coin, flag])
    qml.adjoint(toggle_weight_one)(q, flag)
    # Reflect about the prepared state, up to a common phase. The inverse preparation cancels it.
    qml.adjoint(rotations)()
    controls = [*q, coin]
    for wire in controls:
        qml.PauliX(wire)
    qml.MultiControlledX(wires=[*controls, flag])
    qml.PauliZ(flag)
    qml.MultiControlledX(wires=[*controls, flag])
    for wire in controls:
        qml.PauliX(wire)
    rotations()
    qml.PauliX(coin)


def swap_selected(flags, rows, target):
    """On a one-hot branch swap the active row wire with the ordered target."""
    for flag, row in zip(flags, rows):
        qml.Toffoli([flag, row, target])
    for flag, row in zip(flags, rows):
        qml.Toffoli([flag, target, row])
    for flag, row in zip(flags, rows):
        qml.Toffoli([flag, row, target])


def toggle_equal(labels, value, flags):
    """XOR each [label_i == value] into its local flag. Bit 0 is least significant."""
    for label, flag in zip(labels, flags):
        pattern = [(value >> bit) & 1 for bit in range(len(label))]
        qml.MultiControlledX(wires=[*label, flag], control_values=pattern)


def group_selector(n, group):
    """One frame qubit per group element. Permutations are zero-based tuples."""
    x = list(range(n))
    y = list(range(n, 2 * n))
    q = list(range(2 * n, 2 * n + len(group)))
    coin, flag = q[-1] + 1, q[-1] + 2

    def prepare():
        prepare_choice(q, None, len(q), coin, flag)

    def route():
        for j in range(n):
            swap_selected(q, [x[g[j]] for g in group], y[j])

    return flag + 1, prepare, route


def binary_selector(n):
    """One b-bit register per row. Label zero means unused; labels 1,...,n are ranks."""
    x = list(range(n))
    y = list(range(n, 2 * n))
    b = n.bit_length()
    labels = [list(range(2 * n + i * b, 2 * n + (i + 1) * b)) for i in range(n)]
    start = 2 * n + n * b
    q = list(range(start, start + n))
    flags = list(range(start + n, start + 2 * n))
    coin, flag = start + 2 * n, start + 2 * n + 1

    def prepare():
        for j in range(1, n + 1):
            toggle_equal(labels, 0, flags)
            prepare_choice(q, flags, n - j + 1, coin, flag)
            toggle_equal(labels, 0, flags)
            for row, selected in zip(labels, q):
                for bit, wire in enumerate(row):
                    if (j >> bit) & 1:
                        qml.CNOT([selected, wire])
            toggle_equal(labels, j, q)  # Only the newly assigned row has label j.

    def route():
        for j, target in enumerate(y, start=1):
            toggle_equal(labels, j, flags)
            swap_selected(flags, x, target)
            toggle_equal(labels, j, flags)

    return flag + 1, prepare, route


THETA, PHI = 0.37, 0.29


def spin_pulse(wires):
    """Uniform x pulse followed by commuting all-pairs Ising interactions."""
    for wire in wires:
        qml.RX(THETA, wires=wire)
    for i, j in itertools.combinations(wires, 2):
        qml.CNOT([i, j])
        qml.RZ(PHI, wires=j)
        qml.CNOT([i, j])


def spin_reference(n):
    """Independent dense target, using sum_{i<j} z_i z_j = ((sum_i z_i)^2-n)/2."""
    rx = np.cos(THETA / 2) * np.eye(2) - 1j * np.sin(THETA / 2) * np.array([[0, 1], [1, 0]])
    pulse = np.array([[1]])
    for _ in range(n):
        pulse = np.kron(pulse, rx)
    weights = np.array([x.bit_count() for x in range(2**n)])
    phase = np.exp(-0.25j * PHI * ((n - 2 * weights) ** 2 - n))
    return phase[:, None] * pulse


def translation(wires):
    """Translate three sites around a ring: |abc> goes to |cab>."""
    a, b, c = wires
    for left, right in [(a, b), (a, c)]:
        qml.CNOT([left, right])
        qml.CNOT([right, left])
        qml.CNOT([left, right])


def make_circuit(n, selector, target):
    width, prepare, route = selector
    device = qml.device("lightning.qubit", wires=width)
    ordered_wires = list(range(n, 2 * n))

    @qml.qnode(device, interface=None, diff_method=None)
    def circuit(basis):
        for i in range(n):
            if (basis >> (n - 1 - i)) & 1:
                qml.PauliX(i)
        prepare()
        route()
        target(ordered_wires)
        qml.adjoint(route)()
        qml.adjoint(prepare)()
        return qml.state()

    return circuit


def check_case(name, n, selector, target, reference):
    """Build the full clean-input isometry column by column; compare amplitudes."""
    width = selector[0]
    circuit = make_circuit(n, selector, target)
    dim = 2**n
    errors = np.empty((dim, 2**width), dtype=np.complex128)
    gram = np.empty((dim, dim), dtype=np.complex128)
    leakage = 0.0
    for basis in range(dim):
        errors[basis] = circuit(basis)
        state = errors[basis].reshape(dim, -1)  # data | workspace; column zero is clean.
        leakage = max(leakage, float(np.linalg.norm(state[:, 1:]) ** 2))
        state[:, 0] -= reference[:, basis]
        for other in range(basis + 1):
            value = np.vdot(errors[other], errors[basis])
            gram[other, basis] = value
            gram[basis, other] = value.conjugate()
    error = math.sqrt(max(0.0, float(np.linalg.eigvalsh(gram)[-1])))
    print(f"{name:24s} {width:3d} {dim:5d} {error:13.5e} {leakage:13.5e}")
    assert error < 1e-11 and leakage < 1e-20


def main():
    print(f"PennyLane {qml.__version__}; NumPy {np.__version__}; lightning.qubit; no shots")
    print("Group / target / labels    wires bases     iso.error      max.leak")
    for n in (2, 3):
        group = list(itertools.permutations(range(n)))
        reference = spin_reference(n)
        check_case(f"S{n} / spin / group", n, group_selector(n, group), spin_pulse, reference)
        check_case(f"S{n} / spin / binary", n, binary_selector(n), spin_pulse, reference)

    cycle = [(0, 1, 2), (1, 2, 0), (2, 0, 1)]
    shift = np.zeros((8, 8))
    for x in range(8):
        shift[((x & 1) << 2) | (x >> 1), x] = 1
    check_case("C3 / shift / group", 3, group_selector(3, cycle), translation, shift)

    # Negative control: flipping one particular spin violates the swap symmetry.
    selector = group_selector(2, [(0, 1), (1, 0)])
    circuit = make_circuit(2, selector, lambda wires: qml.PauliX(wires[0]))
    state = np.asarray(circuit(0)).reshape(4, -1)
    leakage = float(np.linalg.norm(state[:, 1:]) ** 2)
    print(f"S2 / one-site flip on |00>: workspace leakage {leakage:.12f}")
    assert abs(leakage - 0.5) < 1e-11


if __name__ == "__main__":
    main()



