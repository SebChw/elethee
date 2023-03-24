import pandas as pd
import numpy as np

Data = pd.DataFrame | np.ndarray


class ElectreTriB:
    def solve(
        self,
        alternatives: Data,
        # attributes_ranges: list[tuple],
        preference_types: np.ndarray,
        thresholds: dict[str, np.ndarray],
        profiles: Data,
    ):
        # expand dimensions so that it is broadcastable with profiles
        self.alternatives = alternatives[:, None, :]
        # self.attribute_ranges = attributes_ranges
        self.preference_types = preference_types
        self.thresholds = {k: arr[None, :, :] for k, arr in thresholds.items()}
        self.profiles = profiles[None, :, :]

        self.n_alternatives = alternatives.shape[0]
        self.n_profiles = profiles.shape[0]
        self.n_criteria = alternatives.shape[1]

        self._marginal_concordance()

    def _marginal_concordance(self):
        sign = self.preference_types[:, :, None]
        preffer_x = (self.profiles - sign*self.thresholds['p'])
        indiff_x = (self.profiles - sign*self.thresholds['q'])

        m_conc_alt_to_prof = np.zeros(
            (self.n_alternatives, self.n_profiles, self.n_criteria))

        m_conc_alt_to_prof[self.alternatives
                           >= indiff_x] = 1

        # Probably not the most efficient way but the cleanest one
        mask_partial_c = (self.alternatives > preffer_x) & (
            self.alternatives < indiff_x)
        partial_c = (self.thresholds['p'] - sign*(self.profiles - self.alternatives)
                     ) / (self.thresholds['p'] - self.thresholds['q'])
        m_conc_alt_to_prof = np.where(
            mask_partial_c, partial_c, m_conc_alt_to_prof)

        print(m_conc_alt_to_prof)
        return m_conc_alt_to_prof


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
    }

    preference_types = np.ndarray([1, 1, 1, 1])  # 1 for gain and -1 for cost

    solver = ElectreTriB()
    solver.solve(alternatives, preference_types, thresholds, profiles)
