import numpy as np

# Codercise 1
ket_0 = np.array([0,1])
ket_1 = np.array([1,0])

def normalize(alpha, beta):

    '''
    To normalize, we divide Psi by its norm which gives us Psi'
    '''

    psi = np.array([alpha, beta])
    norm = np.sqrt(abs(alpha)**2 + abs(beta)**2)

    return psi/norm

#Codercise 2
def inner_product(state_1, state_2):

    bra_state_1 = np.conjugate(state_1).T
    prod = bra_state_1 @ state_2

    return prod

print(f"<0|0> = {inner_product(ket_0, ket_0)}")
print(f"<0|1> = {inner_product(ket_0, ket_1)}")
print(f"<1|0> = {inner_product(ket_1, ket_0)}")
print(f"<1|1> = {inner_product(ket_1, ket_1)}")


#Codercise 3
def measure_state(state, num_meas):
    """Simulate a quantum measurement process."""


    probabilities = np.abs(state)**2


    outcomes = np.random.choice(
        [0, 1],
        size=num_meas,
        p=probabilities
    )

    return outcomes

#Codercise 4

U = np.array([[1, 1], [1, -1]]) / np.sqrt(2)


def apply_u(state):

    return U @ state

#Codercise 5
U = np.array([[1, 1], [1, -1]]) / np.sqrt(2)


def initialize_state():
    return np.array([1, 0])


def apply_u(state):
   
    return np.dot(U, state)


def measure_state(state, num_meas):
   
    p_alpha = np.abs(state[0]) ** 2
    p_beta = np.abs(state[1]) ** 2
    meas_outcome = np.random.choice([0, 1], p=[p_alpha, p_beta], size=num_meas)
    return meas_outcome


def quantum_algorithm():

    state = initialize_state()
    state = apply_u(state)
    return measure_state(state, 100)
