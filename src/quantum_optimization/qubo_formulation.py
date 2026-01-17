"""
qubo_formulation.py
QUBO Model for Optimal Site Selection

Purpose:
- Select best sites
- Respect distance, budget, and quantity constraints
- Ready for quantum / hybrid solvers
"""

import numpy as np
import pandas as pd
import json
import os


class QUBOFormer:
    """
    Convert site selection problem into QUBO
    """

    def __init__(self):
        self.df = None
        self.dist = None
        self.n = 0

    # ---------------- DATA ----------------

    def load_data(self,
                  sites='data/processed/scored_sites.csv',
                  dist='data/processed/distance_matrix.npy'):

        self.df = pd.read_csv(sites)
        self.n = len(self.df)
        self.dist = np.load(dist)

        print(f"Loaded {self.n} sites")
        return self.df

    # ---------------- QUBO BUILD ----------------

    def build_qubo(self,
                   k=5,
                   min_dist=5.0,
                   budget=None,
                   w=None):

        if w is None:
            w = {
                'k': 2.0,
                'dist': 1.5,
                'budget': 1.0
            }

        Q = np.zeros((self.n, self.n))

        # 1. Objective: maximize score
        for i in range(self.n):
            Q[i, i] -= self.df.iloc[i]['final_score']

        # 2. K constraint (select exactly k)
        for i in range(self.n):
            for j in range(self.n):
                Q[i, j] += w['k'] * (1 - 2*k if i == j else 2)

        # 3. Distance constraint
        for i in range(self.n):
            for j in range(i+1, self.n):
                if self.dist[i, j] < min_dist:
                    penalty = w['dist'] * (1 - self.dist[i, j]/min_dist)
                    Q[i, j] += penalty
                    Q[j, i] += penalty

        # 4. Budget constraint (optional)
        if budget:
            for i in range(self.n):
                cost = self.df.iloc[i]['estimated_cost_usd']
                Q[i, i] += w['budget'] * (cost**2 - 2*budget*cost)

        return Q

    # ---------------- UTILITIES ----------------

    def to_dict(self, Q):
        return {
            (i, j): float(Q[i, j])
            for i in range(self.n)
            for j in range(i, self.n)
            if abs(Q[i, j]) > 1e-9
        }

    def analyze(self, Q):
        return {
            'size': Q.shape[0],
            'density': round(np.count_nonzero(Q)/Q.size, 4),
            'min': round(Q.min(), 4),
            'max': round(Q.max(), 4),
            'memory_mb': round(Q.nbytes / 1024**2, 2)
        }

    def save(self, Q, path='results/qubo/qubo.npy'):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, Q)

        info = self.analyze(Q)
        with open(path.replace('.npy', '.json'), 'w') as f:
            json.dump(info, f, indent=2)

        print("QUBO saved")


def main():
    former = QUBOFormer()
    former.load_data()

    Q = former.build_qubo(
        k=5,
        min_dist=5,
        budget=500_000
    )

    print(former.analyze(Q))
    former.save(Q)


if __name__ == "__main__":
    main()
