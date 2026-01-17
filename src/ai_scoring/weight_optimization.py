"""
weight_optimization.py
AI-based Weight Optimization Engine

Goal:
- Find best scoring weights
- Balance fairness and performance
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution


class WeightOptimizer:
    """
    Evolutionary optimizer for scoring weights
    """

    def __init__(self, data_path='data/processed/scored_sites.csv'):
        self.data_path = data_path
        self.df = pd.read_csv(data_path) if os.path.exists(data_path) else None

        # Scoring factors
        self.factors = ['risk', 'access', 'priority', 'weather']

        # Weight limits
        self.bounds = [
            (0.1, 0.6),   # risk
            (0.1, 0.5),   # access
            (0.1, 0.5),   # priority
            (0.05, 0.2)   # weather
        ]

    # ---------------- CORE LOGIC ----------------

    def calculate_scores(self, w):
        """
        Compute scores using given weights
        """
        r, a, p, we = w
        return np.array([
            r * (1 - x['base_risk_score']) +
            a * x['base_access_score'] +
            p * x['base_priority_score'] +
            we * x.get('weather_score', 0.7)
            for _, x in self.df.iterrows()
        ])

    def objective(self, w):
        """
        Optimization target (lower is better)
        """
        w = w / w.sum()
        scores = self.calculate_scores(w)

        mean = scores.mean()
        std = scores.std()
        high_ratio = (scores > 0.7).mean()

        return (
            0.5 * (1 - mean) +
            0.3 * std +
            0.2 * (1 - high_ratio)
        )

    def optimize(self, pop=30, iters=50):
        """
        Run evolutionary optimization
        """
        result = differential_evolution(
            self.objective,
            self.bounds,
            popsize=pop,
            maxiter=iters,
            seed=42
        )

        w = result.x / result.x.sum()

        return {
            'risk': round(w[0], 4),
            'access': round(w[1], 4),
            'priority': round(w[2], 4),
            'weather': round(w[3], 4)
        }

    def evaluate(self, weights):
        """
        Simple evaluation stats
        """
        w = np.array(list(weights.values()))
        scores = self.calculate_scores(w)

        return {
            'mean': round(scores.mean(), 4),
            'std': round(scores.std(), 4),
            'high_ratio': round((scores > 0.7).mean(), 4)
        }

    def save(self, weights, path='config/optimized_weights.json'):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        json.dump(
            {'weights': weights},
            open(path, 'w', encoding='utf-8'),
            indent=2
        )

    def plot(self, weights):
        w = np.array(list(weights.values()))
        scores = self.calculate_scores(w)

        plt.hist(scores, bins=20)
        plt.title("Score Distribution")
        plt.xlabel("Score")
        plt.ylabel("Sites")
        plt.grid(alpha=0.3)

        os.makedirs('results', exist_ok=True)
        plt.savefig('results/score_distribution.png', dpi=300)
        plt.close()


def main():
    optimizer = WeightOptimizer()
    if optimizer.df is None:
        print("No data found")
        return

    default = {'risk':0.35,'access':0.25,'priority':0.3,'weather':0.1}
    print("Default:", optimizer.evaluate(default))

    best = optimizer.optimize()
    print("Optimized:", best)
    print("Optimized eval:", optimizer.evaluate(best))

    optimizer.save(best)
    optimizer.plot(best)


if __name__ == "__main__":
    main()
