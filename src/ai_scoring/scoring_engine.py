"""
scoring_engine.py
Smart Scoring Engine for Humanitarian Sites

Purpose:
- Score and rank sites
- Help decision makers choose best locations
"""

import os
import json
import pandas as pd


class ScoringEngine:
    """
    Multi-criteria scoring system
    """

    def __init__(self, config_path=None):
        # Default weights
        self.weights = {
            'risk': 0.35,
            'access': 0.25,
            'priority': 0.30,
            'weather': 0.10
        }

        # Load custom config if exists
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)

        # Site type impact
        self.site_type_modifiers = {
            'hospital': {'priority': 1.2, 'risk': 1.1},
            'school': {'priority': 1.1, 'risk': 0.9},
            'camp': {'priority': 1.3, 'risk': 1.0},
            'aid_center': {'priority': 1.0, 'risk': 0.8},
            'water_station': {'priority': 1.2, 'risk': 0.7}
        }

    def load_config(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            self.weights.update(json.load(f).get('weights', {}))

    # ---------------- CORE LOGIC ----------------

    def calculate_base_score(self, site):
        """
        Main score logic
        """
        risk = site.get('base_risk_score', 0.5)
        access = site.get('base_access_score', 0.5)
        priority = site.get('base_priority_score', 0.5)
        weather = site.get('weather_score', 0.7)

        site_type = site.get('site_type', 'aid_center')
        mod = self.site_type_modifiers.get(site_type, {'priority': 1, 'risk': 1})

        score = (
            self.weights['risk'] * (1 - risk * mod['risk']) +
            self.weights['access'] * access +
            self.weights['priority'] * priority * mod['priority'] +
            self.weights['weather'] * weather
        )

        return round(max(0, min(1, score)), 4)

    def score_site(self, site):
        """
        Score one site
        """
        base = self.calculate_base_score(site)

        cost_eff = site.get('population_served', 1000) / max(
            site.get('estimated_cost_usd', 1), 1
        )

        urgency = (
            site.get('base_priority_score', 0.5) * 0.6 +
            site.get('base_risk_score', 0.5) * 0.4
        )

        final = (
            base * 0.7 +
            cost_eff * 10000 * 0.2 +
            urgency * 0.1
        )

        return {
            'site_id': site.get('site_id'),
            'final_score': round(max(0, min(1, final)), 4)
        }

    def score_dataset(self, df):
        """
        Score all sites
        """
        scores = df.apply(lambda r: self.score_site(r.to_dict()), axis=1)
        scores_df = pd.DataFrame(scores.tolist())

        df = df.merge(scores_df, on='site_id', how='left')
        df['rank'] = df['final_score'].rank(ascending=False)

        return df
