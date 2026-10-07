#Codercise 1.6.1
dev = qp.device("default.qubit", wires=1)


@qp.qnode(dev)
def apply_rx_pi(state):
    """Apply an RX gate with an angle of \pi to a particular basis state.

    Args:
        state (int): Either 0 or 1. If 1, initialize the qubit to state |1>
            before applying other operations.

    Returns:
        np.array[complex]: The state of the qubit after the operations.
    """
    if state == 1:
        qp.PauliX(wires=0)

    qp.RX(np.pi, wires=0)

    return qp.state()


print(apply_rx_pi(0))
print(apply_rx_pi(1))

#Codercise 1.6.2
dev = qp.device("default.qubit", wires=1)


@qp.qnode(dev)
def apply_rx(theta, state):
    """Apply an RX gate with an angle of theta to a particular basis state.

    Args:
        theta (float): A rotation angle.
        state (int): Either 0 or 1. If 1, initialize the qubit to state |1>
            before applying other operations.

    Returns:
        np.array[complex]: The state of the qubit after the operations.
    """
    if state == 1:
        qp.PauliX(wires=0)

    qp.RX(theta, wires=0)

    return qp.state()


# Code for plotting
angles = np.linspace(0, 4 * np.pi, 200)
output_states = np.array([apply_rx(t, 0) for t in angles])

plot = plotter(angles, output_states)

#Codercise 1.6.3
dev = qp.device("default.qubit", wires=1)


@qp.qnode(dev)
def apply_ry(theta, state):
    """Apply an RY gate with an angle of theta to a particular basis state.

    Args:
        theta (float): A rotation angle.
        state (int): Either 0 or 1. If 1, initialize the qubit to state |1>
            before applying other operations.

    Returns:
        np.array[complex]: The state of the qubit after the operations.
    """
    if state == 1:
        qp.PauliX(wires=0)

    qp.RY(theta, wires=0)

    return qp.state()


# Code for plotting
angles = np.linspace(0, 4 * np.pi, 200)
output_states = np.array([apply_ry(t, 0) for t in angles])

plot = plotter(angles, output_states)

#Codercise 1.7.1
dev = qp.device("default.qubit", wires=1)

phi, theta, omega = np.pi / 2, np.pi / 2, np.pi / 2


@qp.qnode(dev)
def hadamard_with_rz_rx():
    qp.RZ(phi, wires=0)
    qp.RX(theta, wires=0)
    qp.RZ(omega, wires=0)
    return qp.state()

#Codercise 1.7.2
dev = qp.device("default.qubit", wires=1)


@qp.qnode(dev)
def convert_to_rz_rx():
    qp.RZ(np.pi / 2, wires=0)
    qp.RX(-np.pi / 2, wires=0)
    qp.RZ(np.pi / 4, wires=0)

    return qp.state()

#Codercise 1.7.3
dev = qp.device("default.qubit", wires=1)


@qp.qnode(dev)
def unitary_with_h_and_t():
    qp.Hadamard(wires=0)
    qp.T(wires=0)
    qp.Hadamard(wires=0)
    qp.T(wires=0)
    qp.T(wires=0)
    qp.Hadamard(wires=0)

    return qp.state()

#Codercise 1.8.1
dev = qp.device("default.qubit", wires=1)


@qp.qnode(dev)
def prepare_state():
    qp.Hadamard(wires=0)
    qp.RZ(5 * np.pi / 4, wires=0)

    return qp.state()

#Codercise 1.8.2
dev = qp.device("default.qubit", wires=1)


@qp.qnode(dev)
def prepare_state():
    qp.RX(np.pi / 3, wires=0)

    return qp.state()

#Codercise 1.8.3
v = np.array([0.52889389 - 0.14956775j, 0.67262317 + 0.49545818j])

dev = qp.device("default.qubit", wires=1)


@qp.qnode(dev)
def prepare_state(state=v):
    qp.MottonenStatePreparation(state_vector=state, wires=0)

    return qp.state()


# This will draw the quantum circuit and allow you to inspect the output gates
print(prepare_state(v))
print()
print(qp.draw(prepare_state, level="device")(v))

#Codercise 1.10.1
dev = qp.device("default.qubit", wires=1)


@qp.qnode(dev)
def circuit():
    qp.RX(np.pi / 4, wires=0)
    qp.Hadamard(wires=0)
    qp.PauliZ(wires=0)

    return qp.expval(qp.PauliY(0))


print(circuit())

#Codercise 1.10.2
shot_results = []

shot_values = [100, 1000, 10000, 100000, 1000000]

for shots in shot_values:
    dev = qp.device("default.qubit", wires=1, shots=shots)

    @qp.qnode(dev)
    def circuit():
        qp.RX(np.pi / 4, wires=0)
        qp.Hadamard(wires=0)
        qp.PauliZ(wires=0)
        return qp.expval(qp.PauliY(0))

    shot_results.append(circuit())

print(qp.math.unwrap(shot_results))

#Codercise 1.10.3
dev = qp.device("default.qubit", wires=1, shots=100000)


@qp.qnode(dev)
def circuit():
    qp.RX(np.pi / 4, wires=0)
    qp.Hadamard(wires=0)
    qp.PauliZ(wires=0)

    return qp.sample(qp.PauliY(0))


def compute_expval_from_samples(samples):
    """Compute the expectation value of an observable given a set of
    sample outputs. You can assume that there are two possible outcomes,
    1 and -1.

    Args:
        samples (np.array[float]): 100000 samples representing the results of
            running the above circuit.

    Returns:
        float: the expectation value computed based on samples.
    """

    estimated_expval = np.mean(samples)

    return estimated_expval


samples = circuit()
print(compute_expval_from_samples(samples))

#Codercise 1.10.4
#Codercise 1.10.4
def variance_experiment(n_shots):
    """Run an experiment to determine the variance in an expectation
    value computed with a given number of shots.

    Args:
        n_shots (int): The number of shots

    Returns:
        float: The variance in expectation value we obtain running the
        circuit 100 times with n_shots shots each.
    """

    n_trials = 100

    dev = qp.device("default.qubit", wires=1, shots=n_shots)

    @qp.qnode(dev)
    def circuit():
        qp.Hadamard(wires=0)
        return qp.expval(qp.PauliZ(wires=0))

    results = [circuit() for _ in range(n_trials)]

    return np.var(results)


def variance_scaling(n_shots):
    """Once you have determined how the variance in expectation value scales
    with the number of shots, complete this function to programmatically
    represent the relationship.

    Args:
        n_shots (int): The number of shots

    Returns:
        float: The variance in expectation value we expect to see when we run
        an experiment with n_shots shots.
    """

    estimated_variance = 1 / n_shots

    return estimated_variance


# Various numbers of shots; you can change this
shot_vals = [10, 20, 40, 100, 200, 400, 1000, 2000, 4000]

# Used to plot your results
results_experiment = [variance_experiment(shots) for shots in shot_vals]
results_scaling = [variance_scaling(shots) for shots in shot_vals]
plot = plotter(shot_vals, results_experiment, results_scaling)

#Codercise 1.11.1
num_wires = 3
dev = qp.device("default.qubit", wires=num_wires)


@qp.qnode(dev)
def make_basis_state(basis_id):
    """Produce the 3-qubit basis state corresponding to |basis_id>.

    Note that the system starts in |000>.

    Args:
        basis_id (int): An integer value identifying the basis state to construct.

    Returns:
        np.array[complex]: The computational basis state |basis_id>.
    """

    bits = np.array([int(b) for b in np.binary_repr(basis_id, width=num_wires)])
    qp.BasisState(bits, wires=range(num_wires))

    return qp.state()


basis_id = 3
print(f"Output state = {make_basis_state(basis_id)}")

#Codercise 1.11.2
dev = qp.device("default.qubit", wires=2)


@qp.qnode(dev)
def two_qubit_circuit():
    qp.Hadamard(wires=0)
    qp.PauliX(wires=1)

    return qp.expval(qp.PauliY(0)), qp.expval(qp.PauliZ(1))

#Codercise 1.11.3
dev = qp.device("default.qubit", wires=2)


@qp.qnode(dev)
def create_one_minus():
    qp.PauliX(wires=0)
    qp.PauliX(wires=1)
    qp.Hadamard(wires=1)

    return qp.expval(qp.PauliZ(0) @ qp.PauliX(1))


print(create_one_minus())

#Codercise 1.11.4
#Codercise 1.11.4
dev = qp.device("default.qubit", wires=2)


@qp.qnode(dev)
def circuit_1(theta):
    """Implement the circuit and measure Z I and I Z.

    Args:
        theta (float): a rotation angle.

    Returns:
        float, float: The expectation values of the observables Z I, and I Z
    """
    qp.RX(theta, wires=0)
    qp.RY(2 * theta, wires=1)

    return qp.expval(qp.PauliZ(0)), qp.expval(qp.PauliZ(1))


@qp.qnode(dev)
def circuit_2(theta):
    """Implement the circuit and measure Z Z.

    Args:
        theta (float): a rotation angle.

    Returns:
        float: The expectation value of the observable Z Z
    """
    qp.RX(theta, wires=0)
    qp.RY(2 * theta, wires=1)

    return qp.expval(qp.PauliZ(0) @ qp.PauliZ(1))


def zi_iz_combination(ZI_results, IZ_results):
    """Implement a function that acts on the ZI and IZ results to
    produce the ZZ results. How do you think they should combine?

    Args:
        ZI_results (np.array[float]): Results from the expectation value of
            ZI in circuit_1.
        IZ_results (np.array[float]): Results from the expectation value of
            IZ in circuit_2.

    Returns:
        np.array[float]: A combination of ZI_results and IZ_results that
        produces results equivalent to measuring ZZ.
    """

    combined_results = ZI_results * IZ_results

    return combined_results


theta = np.linspace(0, 2 * np.pi, 100)

# Run circuit 1, and process the results
circuit_1_results = np.array([circuit_1(t) for t in theta])

ZI_results = circuit_1_results[:, 0]
IZ_results = circuit_1_results[:, 1]
combined_results = zi_iz_combination(ZI_results, IZ_results)

# Run circuit 2
ZZ_results = np.array([circuit_2(t) for t in theta])

# Plot your results
plot = plotter(theta, ZI_results, IZ_results, ZZ_results, combined_results)

#Codercise 1.9.1
dev = qp.device("default.qubit", wires=1)


@qp.qnode(dev)
def apply_h_and_measure(state):
    """Complete the function such that we apply the Hadamard gate
    and measure in the computational basis.

    Args:
        state (int): Either 0 or 1. If 1, prepare the qubit in state |1>,
            otherwise leave it in state 0.

    Returns:
        np.array[float]: The measurement outcome probabilities.
    """
    if state == 1:
        qp.PauliX(wires=0)

    qp.Hadamard(wires=0)

    return qp.probs(wires=0)


print(apply_h_and_measure(0))
print(apply_h_and_measure(1))

#Codercise 1.9.2
def prepare_psi():
    qp.RX(-2 * np.pi / 3, wires=0)


def y_basis_rotation():
    qp.Hadamard(wires=0)
    qp.S(wires=0)

#Codercise 1.9.3
dev = qp.device("default.qubit", wires=1)


@qp.qnode(dev)
def measure_in_y_basis():
    prepare_psi()
    qp.adjoint(y_basis_rotation)()

    return qp.probs(wires=0)


print(measure_in_y_basis())
