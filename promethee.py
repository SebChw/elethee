import pandas as pd
import numpy as np
from enum import Enum

Data = pd.DataFrame | np.ndarray


class Promethe:
    def solve(
        self,
        alternatives: Data,
        preference_types: np.ndarray,
        thresholds: dict[str, np.ndarray],
        weights: np.array,
        alternatives_names: list[str]
    ):
        self.alternatives = alternatives
        self.preference_types = preference_types
        self.thresholds = thresholds
        self.weights = weights
        self.alternatives_names = alternatives_names
        self.partial_pi = np.zeros((len(self.preference_types), len(
            self.alternatives), len(self.alternatives)))

        for g, g_type in enumerate(self.preference_types):
            p = self.thresholds['p'][g]
            q = self.thresholds['q'][g]
            for i, a in enumerate(self.alternatives):
                for j, b in enumerate(self.alternatives):
                    if i == j:
                        continue
                    d = None
                    if g_type == 1:
                        d = a[g] - b[g]
                    else:
                        d = b[g] - a[g]
                    if d > p:
                        self.partial_pi[g, i, j] = 1
                    elif d <= q:
                        self.partial_pi[g, i, j] = 0
                    else:
                        self.partial_pi[g, i, j] = (d - q) / (p - q)

        self.pi = np.tensordot(self.weights, self.partial_pi, axes=1)
        self.pi /= np.sum(self.weights)
        self.positive_flows = np.sum(self.pi, axis=1)
        self.negative_flows = np.sum(self.pi, axis=0)
        self.net_flows = self.positive_flows - self.negative_flows
        return self.ranking()

    def ranking(self):
        pass


class PrometheI(Promethe):
    def ranking(self):
        n = self.pi.shape[0]
        self.matrix = np.zeros(self.pi.shape, dtype=str)
        for a in range(n):
            for b in range(n):
                if self.positive_flows[a] == self.positive_flows[b] and self.negative_flows[a] == self.negative_flows[b]:
                    self.matrix[a][b] = "I"
                elif (self.positive_flows[a] > self.positive_flows[b] and self.negative_flows[a] < self.negative_flows[b]) or \
                    (self.positive_flows[a] == self.positive_flows[b] and self.negative_flows[a] < self.negative_flows[b]) or \
                        (self.positive_flows[a] > self.positive_flows[b] and self.negative_flows[a] == self.negative_flows[b]):
                    self.matrix[a][b] = "P"
                elif (self.positive_flows[a] > self.positive_flows[b] and self.negative_flows[a] > self.negative_flows[b]) or \
                        (self.positive_flows[a] < self.positive_flows[b] and self.negative_flows[a] < self.negative_flows[b]):
                    self.matrix[a][b] = "R"
                else:
                    self.matrix[a][b] = "N"
        # Now no other idea but to use Copeland method

        self.scores = np.zeros(n)

        for i in range(n):
            for j in range(n):
                if self.matrix[i,j] == 'P':
                    self.scores[i] += 1
                elif self.matrix[i,j] == 'N':
                    self.scores[i] -= 1

        if np.any(self.scores < 0):
            self.scores -= np.min(self.scores)
        self.scores = self.scores.astype(int)
        self.idx = np.arange(n)
        counts = [[] for _ in range(np.max(self.scores)+1)]
        for i in range(n):
            counts[self.scores[i]].append(i)
        self.final_ranking = []
        for i in range(len(counts)):
            if len(counts[i]) > 0:
                self.final_ranking.append(counts[i])
        self.final_ranking = self.final_ranking[::-1]
        self.final_ranking_names = []
        for i in range(len(self.final_ranking)):
            self.final_ranking_names.append([self.alternatives_names[j] for j in self.final_ranking[i]])
        
        return self.final_ranking_names
    
    


class PrometheII(Promethe):
    def ranking(self):
        self.ranking = np.arange(self.pi.shape[0])
        self.ranking = self.ranking[np.argsort(self.net_flows)]
        self.ranking = self.ranking[::-1]
        self.ranking_names = [self.alternatives_names[i] for i in self.ranking]
        return self.ranking_names


if __name__ == "__main__":
    alternatives = np.array(
        [
            [98, 8, 400],
            [58, 0, 800],
            [66, 5, 1000],
            [74, 3, 600],
            [80, 7, 200],
            [82, 10, 600],
        ]
    )
    alternatives_names = ["ITA", "BEL", "GER", "SWE", "AUT", "FRA"]

    thresholds = {
        "q": np.array([0, 0, 100]),
        "p": np.array([0, 2, 300])
    }

    preference_types = np.array([1, 1, -1])  # 1 for gain and -1 for cost
    weights = np.array([3, 2, 5])
    solver = PrometheI()
    print(solver.solve(alternatives, preference_types,
                       thresholds, weights, alternatives_names))
