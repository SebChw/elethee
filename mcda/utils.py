import numpy as np


def get_srf(Z: int, ordered_criteria: list[str], blank_cards: list[int]):
    """Calculate weights using SRF method

    Args:
        Z (int): ratio: weight_of_most_important/weight_of_least_important
        ordered_criteria (list[str]): criteria ordered from the worst to the best.
        If two criteria are on the same level pass them IN A TUPLE like this: ["g1", ("g3", "g2")]
        blank_cards (list[int]): number of blank cards between 2 consecutive criteria

    Returns:
        dict: name of criterion as key and it's weight as a value
    """
    num_of_criteria = len(ordered_criteria)
    assert len(blank_cards) == num_of_criteria-1

    r2 = np.arange(num_of_criteria) + np.cumsum([0] + blank_cards) + 1
    w = 1 + (Z-1) * (r2 - 1)/(r2[-1] - 1)
    w /= w.sum()

    weights_dict = {}
    for criterion, weight in zip(ordered_criteria, w):
        if isinstance(criterion, tuple):
            for crit in criterion:
                weights_dict[crit] = weight
        else:
            weights_dict[criterion] = weight

    return weights_dict


def infer_lambda(weights: np.ndarray, criteria_to_be_compared: list[list]):
    weights_sum = []
    for criteria in criteria_to_be_compared:
        weights_sum.append(weights[criteria].sum())

    return min(weights_sum)

def create_rank_matrics(rank):
    """I know that we don't consider indifference, sorry for that."""
    n_alternatives = len(rank)
    ranking = np.zeros((n_alternatives, n_alternatives))
    for i in range(n_alternatives):
        better_alternative = rank[i]
        for j in range(i, n_alternatives):
            worse_alternative = rank[j]
            ranking[better_alternative, worse_alternative] = 1

    return ranking

def kendalls_tau(rank1, rank2):
    matrix1 = create_rank_matrics(rank1)
    matrix2 = create_rank_matrics(rank2)

    m = len(rank1)
    k_distance = np.sum(np.abs(matrix1 - matrix2))/2

    return 1 - 4 * k_distance/ (m * (m-1))

if __name__ == "__main__":
    print(get_srf(10, ["g1", "g2", "g3", "g4", "g5"], [2, 0, 3, 1]))
    print(get_srf(8, ["g4", "g3", "g2", "g1"], [3, 0, 1]))
    print(get_srf(8, ["g4", ("g3", "g2"), "g1"], [3, 1]))
