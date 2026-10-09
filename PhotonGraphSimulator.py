from collections import defaultdict
from itertools import combinations, product
from math import isclose, sqrt


def perfect_matchings(vertices, weights):
    """Pair the first remaining vertex with each available neighbor."""
    if not vertices:
        yield ()
        return
    i = vertices[0]
    for j in vertices[1:]:
        if (i, j) in weights:
            rest = tuple(v for v in vertices if v != i and v != j)
            for matching in perfect_matchings(rest, weights):
                yield ((i, j),) + matching


def simulate(n, d, weights):
    """Return unnormalized amplitudes and the contributing matchings.

    State keys are tuples (c_0,...,c_{n-1}). Zero-based color labels
    correspond to the draft's labels 1,...,d. Complex amplitudes are
    added before computing probabilities, retaining interference.
    """
    if not isinstance(n, int) or n < 2 or n % 2:
        raise ValueError("n must be a positive even integer of at least 2")
    if not isinstance(d, int) or d < 2:
        raise ValueError("d must be an integer of at least 2")
    for (i, j), matrix in weights.items():
        if not 0 <= i < j < n:
            raise ValueError("Each edge must satisfy 0 <= i < j < n")
        if len(matrix) != d or any(len(row) != d for row in matrix):
            raise ValueError("Each edge matrix must have shape d by d")
    matchings = list(perfect_matchings(tuple(range(n)), weights))
    state = defaultdict(complex)
    for matching in matchings:
        # Each option assigns endpoint colors and an amplitude to one pair.
        options = [
            [(a, b, complex(weights[edge][a][b]))
             for a in range(d) for b in range(d)
             if weights[edge][a][b] != 0]
            for edge in matching
        ]
        for assignment in product(*options):
            colors = [0] * n
            amplitude = 1 + 0j
            for (i, j), (a, b, value) in zip(matching, assignment):
                colors[i], colors[j] = a, b
                amplitude *= value
            state[tuple(colors)] += amplitude
    return dict(state), matchings


def statistics(state, n, d):
    """Normalize and compare with sum_c |c,...,c>/sqrt(d)."""
    norm2 = sum(abs(a) ** 2 for a in state.values())
    if norm2 == 0:
        raise ValueError("The postselected state is zero")
    psi = {colors: a / sqrt(norm2) for colors, a in state.items()}
    overlap = sum(psi.get((c,) * n, 0j) for c in range(d)) / sqrt(d)
    leakage = sum(abs(a) ** 2 for colors, a in psi.items()
                  if len(set(colors)) > 1)
    return psi, abs(overlap) ** 2, leakage


def color_edge(d, c):
    """A source that emits both photons in the single color c."""
    return [[int(a == c and b == c) for b in range(d)]
            for a in range(d)]


def examples():
    # The three matchings of K4 receive three different colors.
    k4 = {}
    for c, matching in enumerate([
        [(0, 1), (2, 3)], [(0, 2), (1, 3)], [(0, 3), (1, 2)]
    ]):
        for edge in matching:
            k4[edge] = color_edge(3, c)

    # The two alternating matchings of a six-cycle receive colors 0 and 1.
    cycle = {edge: color_edge(2, c)
             for c, matching in enumerate([
                 [(0, 1), (2, 3), (4, 5)],
                 [(0, 5), (1, 2), (3, 4)]
             ]) for edge in matching}

    # Equal-color pair sources on every edge of K6.
    identity = [[int(a == b) for b in range(3)] for a in range(3)]
    k6 = {edge: identity for edge in combinations(range(6), 2)}
    return [("K4 with three colors", 4, 3, k4),
            ("C6 with two colors", 6, 2, cycle),
            ("K6 with three colors and W = I", 6, 3, k6)]


def verify():
    """Compare with the coloring formula using complex asymmetric sources."""
    weights = {
        edge: [[1 + 1j * (i + 1), -0.3j], [0.7, j - 0.2j]]
        for edge in combinations(range(4), 2) for i, j in [edge]
    }
    state, _ = simulate(4, 2, weights)
    explicit_matchings = [((0, 1), (2, 3)), ((0, 2), (1, 3)),
                          ((0, 3), (1, 2))]
    for colors in product(range(2), repeat=4):
        expected = sum(weights[i][colors[i[0]]][colors[i[1]]] *
                       weights[j][colors[j[0]]][colors[j[1]]]
                       for i, j in explicit_matchings)
        assert abs(state.get(colors, 0j) - expected) < 1e-10
    # Two matchings producing the same pattern cancel coherently.
    cancel = {edge: color_edge(2, 0) for edge in
              [(0, 1), (2, 3), (0, 2), (1, 3)]}
    cancel[(0, 2)][0][0] = -1
    cancelled, _ = simulate(4, 2, cancel)
    assert sum(abs(a) ** 2 for a in cancelled.values()) == 0
    for index, (_, n, d, weights) in enumerate(examples()):
        state, matchings = simulate(n, d, weights)
        _, fidelity, leakage = statistics(state, n, d)
        expected = [(3, 1.0, 0.0), (2, 1.0, 0.0), (15, 3/7, 4/7)]
        count, target_fidelity, target_leakage = expected[index]
        assert len(matchings) == count
        assert isclose(fidelity, target_fidelity, abs_tol=1e-12)
        assert isclose(leakage, target_leakage, abs_tol=1e-12)


if __name__ == "__main__":
    verify()
    print("All validation checks passed")
    for title, n, d, weights in examples():
        state, matchings = simulate(n, d, weights)
        psi, fidelity, leakage = statistics(state, n, d)
        print(f"\n{title}")
        print(f"Perfect matchings: {len(matchings)}")
        print(f"GHZ fidelity: {fidelity:.6f}")
        print(f"Nonconstant-pattern probability: {leakage:.6f}")
        print("Most likely measurement outcomes:")
        for colors, amplitude in sorted(
            psi.items(), key=lambda item: -abs(item[1]) ** 2
        )[:6]:
            ket = "".join(map(str, colors))
            print(f"  |{ket}> amplitude={amplitude:.6g} "
                  f"probability={abs(amplitude) ** 2:.6f}")
