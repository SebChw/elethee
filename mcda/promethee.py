import pandas as pd
import numpy as np
from enum import Enum
import copy
import matplotlib.pyplot as plt
import networkx as nx

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

    def create_graph(self):
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
    
    def create_graph(self):
        df = pd.DataFrame(self.matrix, columns=self.alternatives_names, index=self.alternatives_names)
        arrows = {x:[] for x in self.alternatives_names}
        reversed_arrows = {x:[] for x in self.alternatives_names}
        for i, level in enumerate(self.final_ranking_names[1:]):
            for alt in level:
                for prev_alt in self.final_ranking_names[i]:
                    if df[alt][prev_alt] == 'P':
                        arrows[prev_alt].append(alt)
                        reversed_arrows[alt].append(prev_alt)

        for alt in arrows: # append the alternatives that have no predecessors from previous step
            if len(arrows[alt]) == 0:
                for row in df:
                    if df[row][alt] == 'P':
                        arrows[alt].append(row)

        allPconnections = copy.deepcopy(arrows)
        for i, level in enumerate(self.final_ranking_names[::-1]):
            for alt in level:
                for pointing_alt in reversed_arrows[alt]:
                    for refs in reversed_arrows[pointing_alt]:
                        allPconnections[refs] += allPconnections[alt] + [alt]
                        allPconnections[refs] = list(set(allPconnections[refs]))

        idx = [x for x in self.alternatives_names]
        rest = []
        for i in idx:
            for j in idx:
                if df[j][i] == 'P' and j not in allPconnections[i]:
                    rest.append((i,j))

        for i,j in rest:
            for k, m in rest:
                if j == m and df[k][i] == 'P':
                    rest.remove((i,j))
        for i,j in rest:
            arrows[i].append(j)      
        G = nx.DiGraph()
        G.add_edges_from([(i,j) for i in arrows for j in arrows[i]])
        print(arrows)
        return G

    
    


class PrometheII(Promethe):
    def ranking(self):
        self.final_ranking = np.arange(self.pi.shape[0])
        self.final_ranking = self.final_ranking[np.argsort(self.net_flows)]
        self.final_ranking = self.final_ranking[::-1]
        self.ranking_names = [self.alternatives_names[i] for i in self.final_ranking]
        return self.ranking_names

    def create_graph(self):
        G = nx.DiGraph()
        if len(set(self.net_flows)) == len(self.net_flows):
            G.add_edges_from([(name, self.ranking_names[i+1]) for i, name in enumerate(self.ranking_names[:-1])])
        else:
            lefts = 0
            for i, name in enumerate(self.ranking_names[:-1]):
                if self.net_flows[self.final_ranking[i]] > self.net_flows[self.final_ranking[i+1]]:
                    G.add_edges_from([(name, self.ranking_names[i+1])])
                    while lefts > 0:
                        G.add_edges_from([(self.ranking_names[i-lefts], name)])
                        lefts -= 1
                elif i > 0:
                    G.add_edges_from([(self.ranking_names[i-1], self.ranking_names[i+1])])
                    lefts += 1
        return G