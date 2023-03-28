from mcda.electretrib import ElectreTriB
from mcda.electretrib import Relation as R
import numpy as np


def test_electre():
    """Example taken from lecture slides"""
    alternatives = np.array(
        [
            [90, 86, 46, 30],
            [40, 90, 14, 48],
            [94, 100, 40, 36],
            [78, 76, 30, 50],
            [60, 60, 30, 30],
            [64, 72, 12, 46],
            [62, 88, 22, 48],
            [70, 30, 12, 12],
        ]
    )

    profiles = np.array(
        [
            [64, 61, 32, 32],
            [86, 84, 43, 43],
        ]
    )

    thresholds = {
        "q": np.array(
            [
                [2, 2, 0, 0],
                [3, 2, 0, 0]
            ]),
        "p": np.array(
            [
                [6, 5, 2, 2],
                [7, 8, 2, 2]
            ]),
        "v": np.array([
            [20, 24, np.nan, np.nan],
            [20, 25, np.nan, np.nan]
        ])
    }

    preference_types = np.array([1, 1, 1, 1])  # 1 for gain and -1 for cost
    weights = np.array([0.4, 0.3, 0.25, 0.05])
    cred_th = 0.65
    solver = ElectreTriB()

    assert all(solver.solve(alternatives, preference_types,
                            thresholds, profiles, weights, cred_th, "pessimistic") == [3, 1, 3, 2, 1, 2, 2, 1])

    assert all(solver.solve(alternatives, preference_types,
                            thresholds, profiles, weights, cred_th, "optimistic") == [3, 2, 3, 2, 1, 2, 2, 2])

    assert np.array_equal(solver.marginal_c_a_b, np.array([
        [
            [1, 1, 1, 0],
            [1, 1, 1, 0],
        ],
        [
            [0, 1, 0, 1],
            [0, 1, 0, 1],
        ],
        [
            [1, 1, 1, 1],
            [1, 1, 0, 0],
        ],
        [
            [1, 1, 0, 1],
            [0, 0, 0, 1],
        ],
        [
            [0.5, 1, 0, 0],
            [0, 0, 0, 0],
        ],
        [
            [1, 1, 0, 1],
            [0, 0, 0, 1],
        ],
        [
            [1, 1, 0, 1],
            [0, 1, 0, 1],
        ],
        [
            [1, 0, 0, 0],
            [0, 0, 0, 0],
        ]]))

    #! I compared results with presentation results. In one place there is an error
    #! slide 19 0.33 -> 0.4 and 0.66 -> 0.8
    assert np.allclose(solver.marginal_c_b_a, np.array([
        [
            [0, 0, 0, 1],
            [0.75, 1, 0, 1],
        ],
        [
            [1, 0, 1, 0],
            [1, 0.33333333, 1, 0],
        ],
        [
            [0, 0, 0, 0],
            [0, 0, 1, 1],
        ],
        [
            [0, 0, 1, 0],
            [1, 1, 1, 0],
        ],
        [
            [1, 1, 1, 1],
            [1, 1, 1, 1],
        ],
        [
            [1, 0, 1, 0],
            [1, 1, 1, 0],
        ],
        [
            [1, 0, 1, 0],
            [1, 0.66666667, 1, 0],
        ],
        [
            [0, 1, 1, 1],
            [1, 1, 1, 1],
        ]]))
    np.allclose(solver.marginal_d_a_b, np.array([
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        [
            [1, 0, 0, 0],
            [1, 0, 0, 0],
        ],
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        [
            [0, 0, 0, 0],
            [0.07, 0, 0, 0],
        ],
        [
            [0, 0, 0, 0],
            [1, 0.94, 0, 0],
        ],
        [
            [0, 0, 0, 0],
            [1, 0.23, 0, 0],
        ],
        [
            [0, 0, 0, 0],
            [1, 0, 0, 0],
        ],
        [
            [0, 1, 0, 0],
            [0.69, 1, 0, 0],
        ],
    ]))
    np.allclose(solver.marginal_d_b_a, np.array([
        [
            [1, 1, 0, 0],
            [0, 0, 0, 0],
        ],
        [
            [0, 1, 0, 0],
            [0, 0, 0, 0],
        ],
        [
            [1, 1, 0, 0],
            [0.07, 0.47, 0, 0],
        ],
        [
            [0.57, 0.52, 0, 0],
            [0, 0, 0, 0],
        ],
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        [
            [0, 0.31, 0, 0],
            [0, 0, 0, 0],
        ],
        [
            [0, 1, 0, 0],
            [0, 0, 0, 0],
        ],
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
    ]))

    assert solver.comp_C_a_b[4, 1] == 0
    assert solver.comp_C_b_a[4, 0] == 1
    assert solver.comp_C_a_b[4, 0] == 0.5
    # ! Error from slide propagated here
    assert solver.comp_C_b_a[1, 1] == 0.75

    assert np.allclose(solver.C_b_a, np.array([
        [0, 0.65],
        [0, 0.75],
        [0, 0.22689076],
        [0.09022556, 0.95],
        [1, 1],
        [0.65, 0.95],
        [0, 0.85],
        [0.6, 1.0],
    ]))

    assert np.array_equal(solver.S_a_b, np.array([
        [1, 1],
        [0, 0],
        [1, 1],
        [1, 0],
        [0, 0],
        [1, 0],
        [1, 0],
        [0, 0],
    ]))

    assert np.array_equal(solver.p_structure, np.array([
        [R.PREF_A_TO_B.value, R.INDIFF.value],
        [R.INCOMP.value, R.PREF_B_TO_A.value],
        [R.PREF_A_TO_B.value, R.PREF_A_TO_B.value],
        [R.PREF_A_TO_B.value, R.PREF_B_TO_A.value],
        [R.PREF_B_TO_A.value, R.PREF_B_TO_A.value],
        [R.INDIFF.value, R.PREF_B_TO_A.value],
        [R.PREF_A_TO_B.value, R.PREF_B_TO_A.value],
        [R.INCOMP.value, R.PREF_B_TO_A.value],
    ]))


def test_cost_type_con_and_disc():
    """On slides there is lack of cost type criteria. So we took example from exercise sheet"""
    alternatives = np.array(
        [
            [40],
            [20],
            [10],
            [3],
            [-10]
        ]
    )

    profiles = np.array(
        [
            [15]
        ]
    )

    thresholds = {
        "q": np.array(
            [
                [0]
            ]),
        "p": np.array(
            [
                [10]
            ]),
        "v": np.array([
            [20]
        ])
    }

    preference_types = np.array([-1])  # 1 for gain and -1 for cost
    weights = np.array([1])
    cred_th = 0.65
    solver = ElectreTriB()

    solver.solve(alternatives, preference_types,
                 thresholds, profiles, weights, cred_th, "optimistic")

    np.array_equal(solver.marginal_c_a_b, np.array([
        [[0], [0.5], [1], [1], [1]]
    ]))
    np.array_equal(solver.marginal_c_b_a, np.array([
        [[1], [1], [0.5], [0], [0]]
    ]))

    np.array_equal(solver.marginal_d_a_b, np.array([
        [[1], [0], [0], [0], [0]]
    ]))
    np.array_equal(solver.marginal_d_b_a, np.array([
        [[0], [0], [0], [0.2], [1]]
    ]))
