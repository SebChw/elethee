import pandas as pd
import numpy as np
from enum import Enum
from pulp import LpProblem, LpVariable, LpMinimize, lpSum, LpStatus, value
from abc import ABC, abstractmethod


class RelationUTA(Enum):
    PREFFERENCE = 0
    INDIFFERENCE = 1


class PreferenceType(Enum):
    COST = -1
    GAIN = 1


Data = pd.DataFrame | np.ndarray
pref_information_type = tuple[int, int, RelationUTA]


class UTA(ABC):
    """In this class thera are common parts to both Ordinal Regression, and Inconsistency resolver"""

    def solve(self,
              alternatives: Data,
              pref_informations: list[pref_information_type],
              preference_types: list[PreferenceType],
              num_breaks: list[int],
              problem_name: str = "UTA"):
        """solver for UTA type problems

        Args:
            alternatives (Data): matrix with alternatives on rows and criteria on columns
            pref_informations (list[pref_information_type]): List of tuples with a form (alternative1, alternative2, relation_between_them)
            preference_types (list[PreferenceType]): List indicating whether attribute is cost or gain type
            num_breaks (list[int]): how many breaks every marginal value function should have. Can be custom for every function
            problem_name (str, optional): Problem name. Defaults to "UTA".

        Raises:
            ValueError: If some inconsistency in shapes is detected.
        """

        self.alternatives = alternatives
        self.pref_informations = pref_informations
        self.preference_types = preference_types
        self.num_breaks = num_breaks
        self.problem_name = problem_name
        self.num_criteria = self.alternatives.shape[1]

        if self.alternatives.shape[1] != len(self.num_breaks):
            raise ValueError(
                "You should give number of breaks of every utility function created.")

        self.problem = LpProblem(self.problem_name, LpMinimize)

        self._create_breakpoints_variables()
        self._create_objective()
        self._add_comparison_constraints()
        self._add_normalization_constraints()
        self._add_monotonicity_constraints()

        print(self.problem)
        self.problem.solve()
        print("Status:", LpStatus[self.problem.status])
        for v in self.problem.variables():
            print(v.name, "=", v.varValue)

        print("Total error ", value(self.problem.objective))

    def _create_breakpoints_variables(self):
        """Probably the most confusing function. The goal is to create at the end have 
        self.breakpoints_variables[i]: list[LpVariables] - variables connected to ith marginal utility function
        self.all_break_points[i] : list[float] - values of particular breakpoints needed for linear interpolation calculation
        self.bins_of_u : np.array - for every gi(a) we want to indicate between which 2 breakpoints of marginal utility function it lies.
        """
        ranges_min = self.alternatives.min(axis=0)
        ranges_max = self.alternatives.max(axis=0)

        DEFAULT_NUM_BREAKS = 2
        self.breakpoints_variables = []
        self.all_break_points = []
        # Variable indicating to which bin the evaluation of given attribute of given sample belongs
        self.bins_of_u = np.zeros_like(self.alternatives, dtype=np.uint8)
        for i, num_breaks in enumerate(self.num_breaks):
            break_points = np.linspace(
                ranges_min[i], ranges_max[i], num_breaks + DEFAULT_NUM_BREAKS)
            self.all_break_points.append(break_points)

            self.bins_of_u[:, i] = np.digitize(
                self.alternatives[:, i], break_points)
            # THe biggest possile value is assumed to be in the last bin not in the last + 1 as np.digitize suggests
            self.bins_of_u[:, i][self.bins_of_u[:, i]
                                 == len(break_points)] -= 1

            self.breakpoints_variables.append([])
            for break_point in break_points:
                self.breakpoints_variables[i].append(
                    LpVariable(f"U_{i}({break_point})", lowBound=0, upBound=1))

    def _add_normalization_constraints(self,):
        best_values = []
        for pref_type, breakpoint_variables in zip(self.preference_types, self.breakpoints_variables):
            if pref_type == PreferenceType.GAIN:
                best_values.append(breakpoint_variables[-1])
                self.problem += breakpoint_variables[0] == 0
            else:
                best_values.append(breakpoint_variables[0])
                self.problem += breakpoint_variables[-1] == 0

        self.problem += lpSum(best_values) == 1

    def _add_monotonicity_constraints(self):
        for pref_type, breakpoint_variables in zip(self.preference_types, self.breakpoints_variables):
            for i in range(1, len(breakpoint_variables)):
                if pref_type == PreferenceType.GAIN:
                    self.problem += breakpoint_variables[i] >= breakpoint_variables[i-1]
                else:
                    self.problem += breakpoint_variables[i -
                                                         1] >= breakpoint_variables[i]

    def _create_initial_LHS_RHS(self, a, b):
        """
        U(a) = u1(g1(a)) + u2(g2(a))
        and if gi(a) belongs to [xi, xi_1] => ui(gi(a)) = ui(xi) + (gi(a) - xi) / (xi_1 - xi) * (ui(xi_1) - ui(xi))
        """
        LHS = None
        RHS = None
        for alt in [a, b]:
            for i in range(self.num_criteria):
                bin_a = self.bins_of_u[alt, i]
                x_i, x_j, x_j_1 = self.alternatives[alt,
                                                    i], self.all_break_points[i][bin_a-1], self.all_break_points[i][bin_a]
                u_x_j, u_x_j_1 = self.breakpoints_variables[i][bin_a -
                                                               1], self.breakpoints_variables[i][bin_a]

                if alt == a:
                    LHS += u_x_j + (x_i - x_j)/(x_j_1 - x_j) * \
                        (u_x_j_1 - u_x_j)
                else:
                    RHS += u_x_j + (x_i - x_j)/(x_j_1 - x_j) * \
                        (u_x_j_1 - u_x_j)

        return LHS, RHS

    @abstractmethod
    def _create_objective(self):
        pass

    @abstractmethod
    def _add_comparison_constraints(self):
        pass


class OrdinalRegression(UTA):
    def _create_objective(self):
        """
            Create objective as sum of positive and negative errors for every U(a)
        """
        self.error_variables = {}
        self.comparison_constraints = []
        error_vars = []
        for a, b, relation in self.pref_informations:
            for alt in [a, b]:
                if alt not in self.error_variables:
                    pos = LpVariable(f"p_e_{alt}", lowBound=0)
                    neg = LpVariable(f"n_e_{alt}", lowBound=0)
                    self.error_variables[alt] = {"pos": pos, "neg": neg}
                    error_vars.extend([pos, neg])

        self.problem += lpSum(error_vars)

    def _add_comparison_constraints(self):
        for a, b, relation in self.pref_informations:
            LHS, RHS = self._create_initial_LHS_RHS(a, b)

            LHS = LHS - \
                self.error_variables[a]['pos'] + self.error_variables[a]['neg']
            RHS = RHS - \
                self.error_variables[b]['pos'] + self.error_variables[b]['neg']

            if relation == RelationUTA.PREFFERENCE:
                self.problem += LHS >= (RHS + 0.00001)
            else:
                self.problem += LHS == RHS


class UTAInconsistency(UTA):
    # TODO
    pass


if __name__ == "__main__":
    alternatives = np.array([
        [5, 0, 10],
        [10, 5, 15],
        [0, 10, 12.5],
    ])

    pref_informations = [(1, 0, RelationUTA.PREFFERENCE),
                         (0, 2, RelationUTA.PREFFERENCE)]
    preference_types = [PreferenceType.GAIN,
                        PreferenceType.GAIN, PreferenceType.COST]
    num_breaks = [1, 1, 1]

    OrdinalRegression().solve(alternatives, pref_informations,
                              preference_types, num_breaks)
