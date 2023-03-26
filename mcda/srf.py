import numpy as np

class SRF:
    def solve(self, Z: int, num: int, blank_cards: list[int]):
        assert len(blank_cards) == num-1

        weights = np.zeros(num)
        r = np.zeros(num +1)
        for i in range(1, num+1):
            r[i] = i + np.sum(blank_cards[:i-1])
        
        for i in range(1, num+1):
            weights[i-1] = 1 + (Z-1) * (r[i] - 1) / (r[num] - 1)
        weights = weights / np.sum(weights)
        weights = weights[::-1]
        print(weights)

if __name__ == "__main__":
    srf = SRF()
    srf.solve(10, 5, [2,0,3,1])