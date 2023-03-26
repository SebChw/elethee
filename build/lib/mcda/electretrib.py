import pandas as pd
import numpy as np
from enum import Enum

Data = pd.DataFrame | np.ndarray


class Relation(Enum):
    PREF_A_TO_B = 0
    PREF_B_TO_A = 1
    INDIFF = 2
    INCOMP = 3

#! I compared results with presentation results. In one place there is an error
#! slide 19 0.33 -> 0.4 and 0.66 -> 0.8
#! Some test comparing all these values should be written for sure


class ElectreTriB:
    def solve(
        self,
        alternatives: Data,
        # attributes_ranges: list[tuple],
        preference_types: np.ndarray,
        thresholds: dict[str, np.ndarray],
        profiles: Data,
        weights: np.array,
        credibility_th: float,
        assignment_type: str
    ):
        # expand dimensions so that it is broadcastable with profiles
        self.alternatives = alternatives[:, None, :]
        # self.attribute_ranges = attributes_ranges
        self.preference_types = preference_types
        self.thresholds = {k: arr[None, :, :] for k, arr in thresholds.items()}
        self.profiles = profiles[None, :, :]
        self.weights = weights
        self.credibility_th = credibility_th
        self.assignment_type = assignment_type

        self.n_alternatives = alternatives.shape[0]
        self.n_profiles = profiles.shape[0]
        self.n_criteria = alternatives.shape[1]

        # sign is used to model cost vs gain situation
        self.sign = self.preference_types[None, None, :]
        self.ga_min_gb = self.alternatives - self.profiles

        # We will later select only applicable thresholds
        self.thresholds['v'] = np.where(
            np.isnan(self.thresholds['v']), -1, self.thresholds['v'])

        self.marginal_c_a_b = self._marginal_concordance()
        self.marginal_c_b_a = self._marginal_concordance(
            profile_to_alternative=True)
        self.marginal_d_a_b = self._marginal_discordance()
        self.marginal_d_b_a = self._marginal_discordance(
            profile_to_alternative=True)

        self._aggregate()
        self._threshold_credibility()
        self._preference_structure()
        {"pessimistic": self.pessimistic_assignment,
            "optimistic": self.optimistic_assignment}[self.assignment_type]()

        return self.assignment

    def _fill_the_matrix(self, should_be_1, should_be_0, partial):
        m_sth_alt_to_prof = np.zeros(
            (self.n_alternatives, self.n_profiles, self.n_criteria))

        m_sth_alt_to_prof[should_be_1] = 1
        mask_partial_c = ~(should_be_1 | should_be_0)

        return np.where(
            mask_partial_c, partial, m_sth_alt_to_prof)

    def _marginal_concordance(self, profile_to_alternative=False):
        sign = self.sign * (-1) if profile_to_alternative else self.sign
        should_be_1 = sign*self.ga_min_gb >= -self.thresholds['q']
        should_be_0 = sign*self.ga_min_gb < -self.thresholds['p']

        # Probably not the most efficient way but the cleanest one

        partial_c = (self.thresholds['p'] - sign*(self.profiles - self.alternatives)
                     ) / (self.thresholds['p'] - self.thresholds['q'])

        m_conc = self._fill_the_matrix(
            should_be_1, should_be_0, partial_c)

        # print(m_conc_alt_to_prof)
        return m_conc

    def _marginal_discordance(self, profile_to_alternative=False):
        sign = self.sign * (-1) if profile_to_alternative else self.sign

        should_be_1 = sign*self.ga_min_gb <= -self.thresholds['v']
        should_be_0 = sign*self.ga_min_gb >= -self.thresholds['p']

        partial_d = (sign*(self.profiles - self.alternatives) - self.thresholds['p']
                     ) / (self.thresholds['v'] - self.thresholds['p'])

        m_disc = self._fill_the_matrix(
            should_be_1, should_be_0, partial_d)

        # it won't influence the Credibility and still be broadcastable
        m_disc[np.repeat(
            self.thresholds['v'] == -1, self.n_alternatives, axis=0)] = 0
        # print(m_disc)
        return m_disc

    def _credibility(self, C, d):
        division = np.divide(
            (1 - d), (1-C), out=np.zeros_like(d), where=C != 1)
        return C[:, :, 0] * np.prod(np.where(C < d, division, 1), axis=-1)

    def _aggregate(self):
        self.C_a_b = self.marginal_c_a_b @ self.weights / self.weights.sum()
        self.C_b_a = self.marginal_c_b_a @ self.weights / self.weights.sum()
        # print(self.C_a_b, self.C_b_a)

        self.C_a_b = self._credibility(
            self.C_a_b[:, :, None], self.marginal_d_a_b)
        self.C_b_a = self._credibility(
            self.C_b_a[:, :, None], self.marginal_d_b_a)

        # print(self.C_a_b, self.C_b_a)

    def _threshold_credibility(self):
        self.S_a_b = self.C_a_b >= self.credibility_th
        self.S_b_a = self.C_b_a >= self.credibility_th

        # print(self.S_a_b, self.S_b_a)

    def _preference_structure(self):
        self.p_structure = np.full(
            (self.n_alternatives, self.n_profiles), Relation.PREF_A_TO_B.value)
        self.p_structure[~self.S_a_b & self.S_b_a] = Relation.PREF_B_TO_A.value
        self.p_structure[self.S_a_b & self.S_b_a] = Relation.INDIFF.value
        self.p_structure[~self.S_a_b & ~self.S_b_a] = Relation.INCOMP.value

        # print(self.p_structure)

    def pessimistic_assignment(self):
        self.assignment = self.S_a_b.sum(axis=1) + 1

    def optimistic_assignment(self):
        self.assignment = self.n_profiles - \
            (self.p_structure == Relation.PREF_B_TO_A.value).sum(axis=1) + 1


if __name__ == "__main__":
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
    print(solver.solve(alternatives, preference_types,
                       thresholds, profiles, weights, cred_th, "pessimistic"))

    print(solver.solve(alternatives, preference_types,
                       thresholds, profiles, weights, cred_th, "optimistic"))
