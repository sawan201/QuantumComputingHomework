#Codercise 1.2.1
def my_circuit(theta, phi):
    qp.CNOT(wires=[0, 1])
    qp.RX(theta, wires=2)
    qp.Hadamard(wires=0)
    qp.CNOT(wires=[2, 0])
    qp.RY(phi, wires=1)

    return qp.probs(wires=[0, 1, 2])

#Codercise 1.2.2
dev = qp.device("default.qubit", wires=3)


def my_circuit(theta, phi, omega):

    qp.RX(theta, wires=0)
    qp.RY(phi, wires=1)
    qp.RZ(omega, wires=2)

    qp.CNOT(wires=[0, 1])
    qp.CNOT(wires=[1, 2])
    qp.CNOT(wires=[2, 0])

    return qp.probs(wires=[0, 1, 2])


# This creates a QNode, binding the function and device
my_qnode = qp.QNode(my_circuit, dev)

# We set up some values for the input parameters
theta, phi, omega = 0.1, 0.2, 0.3

# Execute the QNode
my_qnode(theta, phi, omega)

#Codercise 1.2.3
dev = qp.device("default.qubit", wires=3)

##################
# YOUR CODE HERE #
##################

@qp.qnode(dev)


def my_circuit(theta, phi, omega):
    qp.RX(theta, wires=0)
    qp.RY(phi, wires=1)
    qp.RZ(omega, wires=2)
    qp.CNOT(wires=[0, 1])
    qp.CNOT(wires=[1, 2])
    qp.CNOT(wires=[2, 0])
    return qp.probs(wires=[0, 1, 2])


theta, phi, omega = 0.1, 0.2, 0.3

##################
# YOUR CODE HERE #
##################

my_circuit(theta, phi, omega)

#Codercise 1.2.4

dev = qp.device("default.qubit", wires=3)


@qp.qnode(dev)
def my_circuit(theta, phi, omega):
    qp.RX(theta, wires=0)
    qp.RY(phi, wires=1)
    qp.RZ(omega, wires=2)
    qp.CNOT(wires=[0, 1])
    qp.CNOT(wires=[1, 2])
    qp.CNOT(wires=[2, 0])
    return qp.probs(wires=[0, 1, 2])


##################
# YOUR CODE HERE #
##################

# FILL IN THE CORRECT CIRCUIT DEPTH
depth = 4
#Codercise 1.3.1

dev = qp.device("default.qubit", wires=1)

U = np.array([[1, 1], [1, -1]]) / np.sqrt(2)


@qp.qnode(dev)
def apply_u():

    qp.QubitUnitary(U, wires=0)

    # Return the state
    return qp.state()

#Codercise 1.3.2
dev = qp.device("default.qubit", wires=1)


@qp.qnode(dev)
def apply_u_as_rot(phi, theta, omega):

    qp.Rot(phi, theta, omega, wires=0)

    return qp.state()

#Codercise 1.4.1
dev = qp.device("default.qubit", wires=1)

U = np.array([[1, 1], [1, -1]]) / np.sqrt(2)


@qp.qnode(dev)
def varied_initial_state(state):
    """Complete the function such that we can apply the operation U to
    either |0> or |1> depending on the input argument flag.
    """
    
    if state == 1:
        qp.PauliX(wires=0)

    qp.QubitUnitary(U, wires=0)

    return qp.state()

#Codercise 1.4.2
dev = qp.device("default.qubit", wires=1)


@qp.qnode(dev)
def apply_hadamard():

    qp.Hadamard(wires=0)

    return qp.state()

#codercise 1.4.3
dev = qp.device("default.qubit", wires=1)


@qp.qnode(dev)
def apply_hadamard_to_state(state):
    """Complete the function such that we can apply the Hadamard to
    either |0> or |1> depending on the input argument flag.
    """

    if state == 1:
        qp.PauliX(wires=0)

    qp.Hadamard(wires=0)

    return qp.state()


print(apply_hadamard_to_state(0))
print(apply_hadamard_to_state(1))

#Codercise 1.4.4
##################
# YOUR CODE HERE #
##################

# CREATE A DEVICE
dev = qp.device("default.qubit", wires=1)


# CREATE A QNODE CALLED apply_hxh THAT APPLIES THE CIRCUIT ABOVE
@qp.qnode(dev)
def apply_hxh(state):

    if state == 1:
        qp.PauliX(wires=0)

    qp.Hadamard(wires=0)
    qp.PauliX(wires=0)
    qp.Hadamard(wires=0)

    return qp.state()


# Print your results
print(apply_hxh(0))
print(apply_hxh(1))

#Codercise 1.5.1
dev = qp.device("default.qubit", wires=1)


@qp.qnode(dev)
def apply_z_to_plus():
    """Write a circuit that applies PauliZ to the |+> state and returns
    the state.
    """

    # CREATE THE |+> STATE
    qp.Hadamard(wires=0)

    # APPLY PAULI Z
    qp.PauliZ(wires=0)

    # RETURN THE STATE
    return qp.state()


print(apply_z_to_plus())

#Codercise 1.5.2
dev = qp.device("default.qubit", wires=1)


@qp.qnode(dev)
def fake_z():
    """Use RZ to produce the same action as Pauli Z on the |+> state.

    Returns:
        np.array[complex]: The state of the qubit after the operations.
    """

    # CREATE THE |+> STATE
    qp.Hadamard(wires=0)

    # APPLY RZ
    qp.RZ(np.pi, wires=0)

    # RETURN THE STATE
    return qp.state()

#Codercise 1.5.3
dev = qp.device("default.qubit", wires=1)


@qp.qnode(dev)
def many_rotations():
    """Implement the circuit depicted above and return the quantum state.

    Returns:
        np.array[complex]: The state of the qubit after the operations.
    """

    qp.Hadamard(wires=0)
    qp.S(wires=0)
    qp.adjoint(qp.T)(wires=0)
    qp.RZ(0.3, wires=0)
    qp.adjoint(qp.S)(wires=0)

    return qp.state()
