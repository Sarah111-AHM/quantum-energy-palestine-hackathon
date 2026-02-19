"""
QUBO (Quadratic Unconstrained Binary Optimization) Formulation Module
for renewable energy site selection in Gaza Strip.

This module handles the mathematical encoding of the site selection problem
into QUBO format with multiple objectives and constraints.
"""

import numpy as np
import logging
from typing import Tuple, List, Dict, Optional, Union
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ObjectiveType(Enum):
    """Types of optimization objectives."""
    MAXIMIZE = "maximize"  # For benefits (solar, wind)
    MINIMIZE = "minimize"  # For costs (risk, distance)


class ConstraintType(Enum):
    """Types of constraints."""
    EXACT_K = "exact_k"  # Select exactly K sites
    AT_MOST_K = "at_most_k"  # Select at most K sites
    AT_LEAST_K = "at_least_k"  # Select at least K sites
    MUTUALLY_EXCLUSIVE = "mutually_exclusive"  # Cannot both be selected
    DEPENDENCY = "dependency"  # If A selected, B must be selected


@dataclass
class QUBOParameters:
    """Parameters for QUBO formulation."""
    # Primary weights
    solar_weight: float = 0.4
    wind_weight: float = 0.3
    risk_weight: float = 0.5
    grid_weight: float = 0.2
    diversity_weight: float = 0.1
    
    # Penalty strengths
    cardinality_penalty: float = 500.0
    risk_threshold: float = 7.0
    min_distance_km: float = 2.0
    
    # Scaling factors
    score_scale: float = 100.0
    penalty_scale: float = 1000.0
    
    # Advanced options
    use_quadratic_risk: bool = True
    use_distance_penalty: bool = True
    normalize_inputs: bool = True
    
    def validate(self):
        """Validate parameter values."""
        assert 0 <= self.solar_weight <= 1, "Solar weight must be between 0 and 1"
        assert 0 <= self.wind_weight <= 1, "Wind weight must be between 0 and 1"
        assert 0 <= self.risk_weight <= 1, "Risk weight must be between 0 and 1"
        assert 0 <= self.grid_weight <= 1, "Grid weight must be between 0 and 1"
        assert self.cardinality_penalty > 0, "Cardinality penalty must be positive"
        assert self.min_distance_km > 0, "Minimum distance must be positive"
        logger.info("QUBO parameters validated successfully")


class QUBOFormulation:
    """
    Enhanced QUBO formulation for site selection problem.
    
    Handles:
    - Multiple objectives with weighting
    - Complex constraints
    - Geographic diversity
    - Risk mitigation
    - Grid proximity optimization
    """
    
    def __init__(
        self,
        n_sites_to_select: int = 5,
        params: Optional[QUBOParameters] = None
    ):
        """
        Initialize QUBO formulation.
        
        Args:
            n_sites_to_select: Number of sites to select (K)
            params: QUBO parameters (uses defaults if None)
        """
        self.n_sites_to_select = n_sites_to_select
        self.params = params or QUBOParameters()
        self.params.validate()
        
        # Storage for intermediate calculations
        self.normalized_data = {}
        self.distance_matrix = None
        self.risk_matrix = None
        
        logger.info(f"Initialized QUBOFormulation for K={n_sites_to_select} sites")
    
    def build_qubo(
        self,
        solar_scores: np.ndarray,
        wind_scores: np.ndarray,
        risk_scores: np.ndarray,
        coordinates: np.ndarray,
        grid_distances: Optional[np.ndarray] = None,
        accessibility: Optional[np.ndarray] = None,
        constraints: Optional[Dict] = None
    ) -> np.ndarray:
        """
        Build complete QUBO matrix from input data.
        
        Args:
            solar_scores: Solar irradiance values
            wind_scores: Wind speed values
            risk_scores: Risk indices (0-10)
            coordinates: Geographic coordinates (lat, lon)
            grid_distances: Distance to grid infrastructure
            accessibility: Binary accessibility flags
            constraints: Additional constraints
            
        Returns:
            QUBO matrix (n x n)
        """
        n = len(solar_scores)
        logger.info(f"Building QUBO matrix for {n} sites")
        
        # Validate inputs
        self._validate_inputs(
            solar_scores, wind_scores, risk_scores,
            coordinates, grid_distances, accessibility, n
        )
        
        # Normalize inputs
        self._normalize_inputs(
            solar_scores, wind_scores, risk_scores,
            grid_distances, accessibility
        )
        
        # Calculate distance matrix
        self.distance_matrix = self._calculate_distance_matrix(coordinates)
        
        # Calculate risk correlation matrix
        self.risk_matrix = self._calculate_risk_matrix(risk_scores)
        
        # Initialize QUBO matrix
        Q = np.zeros((n, n))
        
        # ===== DIAGONAL TERMS (Site selection benefits/costs) =====
        Q = self._add_diagonal_terms(
            Q,
            self.normalized_data['solar'],
            self.normalized_data['wind'],
            self.normalized_data['risk'],
            self.normalized_data.get('grid'),
            self.normalized_data.get('accessibility')
        )
        
        # ===== OFF-DIAGONAL TERMS (Site interactions) =====
        Q = self._add_off_diagonal_terms(Q)
        
        # ===== CONSTRAINTS =====
        Q = self._add_cardinality_constraint(Q)
        
        if self.params.use_distance_penalty:
            Q = self._add_distance_constraints(Q)
        
        if constraints:
            Q = self._add_custom_constraints(Q, constraints)
        
        # Ensure symmetry
        Q = (Q + Q.T) / 2
        
        # Log matrix statistics
        self._log_matrix_stats(Q)
        
        return Q
    
    def _validate_inputs(
        self,
        solar: np.ndarray,
        wind: np.ndarray,
        risk: np.ndarray,
        coords: np.ndarray,
        grid: Optional[np.ndarray],
        access: Optional[np.ndarray],
        n: int
    ):
        """Validate input arrays."""
        assert len(solar) == n, "Solar scores length mismatch"
        assert len(wind) == n, "Wind scores length mismatch"
        assert len(risk) == n, "Risk scores length mismatch"
        assert coords.shape == (n, 2), "Coordinates shape mismatch"
        
        if grid is not None:
            assert len(grid) == n, "Grid distances length mismatch"
        
        if access is not None:
            assert len(access) == n, "Accessibility length mismatch"
            assert set(access).issubset({0, 1}), "Accessibility must be binary"
        
        logger.info("Input validation passed")
    
    def _normalize_inputs(
        self,
        solar: np.ndarray,
        wind: np.ndarray,
        risk: np.ndarray,
        grid: Optional[np.ndarray],
        access: Optional[np.ndarray]
    ):
        """Normalize all inputs to [0, 1] range."""
        self.normalized_data = {}
        
        # Solar (higher is better)
        self.normalized_data['solar'] = self._normalize(solar, maximize=True)
        
        # Wind (higher is better)
        self.normalized_data['wind'] = self._normalize(wind, maximize=True)
        
        # Risk (lower is better - invert)
        self.normalized_data['risk'] = 1 - self._normalize(risk, maximize=True)
        
        # Grid distance (lower is better - invert)
        if grid is not None:
            self.normalized_data['grid'] = 1 - self._normalize(grid, maximize=True)
        
        # Accessibility (already 0/1)
        if access is not None:
            self.normalized_data['accessibility'] = access.astype(float)
        
        logger.info("Input normalization complete")
    
    def _normalize(self, arr: np.ndarray, maximize: bool = True) -> np.ndarray:
        """
        Normalize array to [0, 1].
        
        Args:
            arr: Input array
            maximize: If True, higher values are better
                     If False, lower values are better
        """
        min_val = arr.min()
        max_val = arr.max()
        
        if max_val > min_val:
            normalized = (arr - min_val) / (max_val - min_val)
        else:
            normalized = np.zeros_like(arr)
        
        return normalized
    
    def _calculate_distance_matrix(self, coordinates: np.ndarray) -> np.ndarray:
        """
        Calculate Haversine distance matrix between all points.
        
        Returns:
            Distance matrix in kilometers
        """
        n = len(coordinates)
        distances = np.zeros((n, n))
        
        # Convert to radians
        coords_rad = np.radians(coordinates)
        
        for i in range(n):
            for j in range(i+1, n):
                lat1, lon1 = coords_rad[i]
                lat2, lon2 = coords_rad[j]
                
                # Haversine formula
                dlat = lat2 - lat1
                dlon = lon2 - lon1
                
                a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
                c = 2 * np.arcsin(np.sqrt(a))
                
                # Distance in kilometers (Earth radius = 6371 km)
                distance = c * 6371
                distances[i, j] = distances[j, i] = distance
        
        logger.info(f"Distance matrix calculated: mean={distances[distances>0].mean():.2f}km")
        return distances
    
    def _calculate_risk_matrix(self, risk_scores: np.ndarray) -> np.ndarray:
        """
        Calculate risk correlation matrix.
        Higher values where sites have similar high risk.
        """
        n = len(risk_scores)
        risk_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i+1, n):
                # Risk similarity penalty
                risk_diff = abs(risk_scores[i] - risk_scores[j])
                risk_sum = risk_scores[i] + risk_scores[j]
                
                if risk_sum > self.params.risk_threshold * 2:
                    # Both are high risk
                    similarity = 1 - (risk_diff / 10)
                    risk_matrix[i, j] = risk_matrix[j, i] = similarity * risk_sum / 20
        
        return risk_matrix
    
    def _add_diagonal_terms(
        self,
        Q: np.ndarray,
        solar: np.ndarray,
        wind: np.ndarray,
        risk: np.ndarray,
        grid: Optional[np.ndarray] = None,
        accessibility: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Add diagonal terms to QUBO matrix.
        """
        n = len(solar)
        
        for i in range(n):
            # Primary objective: maximize suitability
            suitability = (
                solar[i] * self.params.solar_weight +
                wind[i] * self.params.wind_weight
            )
            
            # Risk penalty (negative term since we minimize in QUBO)
            risk_penalty = risk[i] * self.params.risk_weight * self.params.score_scale
            
            # Grid proximity benefit
            grid_benefit = 0
            if grid is not None:
                grid_benefit = grid[i] * self.params.grid_weight * self.params.score_scale
            
            # Accessibility bonus
            access_bonus = 0
            if accessibility is not None and accessibility[i] > 0:
                access_bonus = 20  # Fixed bonus for accessible sites
            
            # Combine terms (negative because we minimize in QUBO)
            Q[i, i] = -suitability * self.params.score_scale + risk_penalty - grid_benefit - access_bonus
            
            # Add quadratic risk term if enabled
            if self.params.use_quadratic_risk and risk[i] > self.params.risk_threshold / 10:
                Q[i, i] += 50 * risk[i]  # Extra penalty for high-risk sites
        
        logger.info(f"Added diagonal terms: range=[{Q.diagonal().min():.2f}, {Q.diagonal().max():.2f}]")
        return Q
    
    def _add_off_diagonal_terms(self, Q: np.ndarray) -> np.ndarray:
        """
        Add off-diagonal terms for site interactions.
        """
        n = Q.shape[0]
        
        # Calculate adaptive distance threshold
        if self.distance_matrix is not None:
            distances = self.distance_matrix
            avg_distance = distances[distances > 0].mean()
            threshold = avg_distance * 0.3  # 30% of average distance
            
            for i in range(n):
                for j in range(i+1, n):
                    distance = distances[i, j]
                    
                    # Geographic diversity penalty
                    if distance < threshold:
                        # Progressive penalty based on closeness
                        closeness = 1 - (distance / threshold)
                        penalty = self.params.diversity_weight * self.params.penalty_scale * closeness
                        Q[i, j] = penalty
                    else:
                        # Small reward for good separation
                        reward = -self.params.diversity_weight * self.params.penalty_scale * 0.1
                        Q[i, j] = reward
                    
                    # Risk correlation penalty
                    if self.risk_matrix is not None:
                        Q[i, j] += self.risk_matrix[i, j] * self.params.penalty_scale * 0.3
            
            logger.info(f"Added off-diagonal terms: max={Q.max():.2f}")
        
        return Q
    
    def _add_cardinality_constraint(self, Q: np.ndarray) -> np.ndarray:
        """
        Add exact cardinality constraint: sum(x_i) = K.
        Uses penalty method: P * (sum(x_i) - K)^2
        """
        n = Q.shape[0]
        K = self.n_sites_to_select
        P = self.params.cardinality_penalty
        
        # Expand (sum(x_i) - K)^2 = sum(x_i^2) + 2*sum(x_i*x_j) - 2K*sum(x_i) + K^2
        # Note: x_i^2 = x_i for binary variables
        
        for i in range(n):
            # Linear term: -2K * x_i
            Q[i, i] += P * (1 - 2 * K)
            
            for j in range(i+1, n):
                # Quadratic term: 2 * x_i * x_j
                Q[i, j] += 2 * P
        
        # Constant term K^2 is ignored (doesn't affect optimization)
        
        logger.info(f"Added cardinality constraint: exactly {K} sites")
        return Q
    
    def _add_distance_constraints(self, Q: np.ndarray) -> np.ndarray:
        """
        Add minimum distance constraints.
        Penalizes selecting sites that are too close together.
        """
        if self.distance_matrix is None:
            return Q
        
        n = Q.shape[0]
        min_distance_rad = self.params.min_distance_km / 6371  # Convert to radians
        
        for i in range(n):
            for j in range(i+1, n):
                if self.distance_matrix[i, j] < self.params.min_distance_km:
                    # Very strong penalty for violating minimum distance
                    Q[i, j] += self.params.penalty_scale * 2
        
        logger.info(f"Added minimum distance constraint: {self.params.min_distance_km}km")
        return Q
    
    def _add_custom_constraints(self, Q: np.ndarray, constraints: Dict) -> np.ndarray:
        """
        Add custom constraints to QUBO.
        
        Args:
            Q: QUBO matrix
            constraints: Dictionary of constraints
                - 'mutually_exclusive': List of pairs that cannot both be selected
                - 'dependencies': List of (a, b) where a requires b
                - 'at_least_one': List of groups where at least one must be selected
        """
        n = Q.shape[0]
        P = self.params.cardinality_penalty / 2  # Weaker penalty for custom constraints
        
        # Mutually exclusive constraints
        if 'mutually_exclusive' in constraints:
            for i, j in constraints['mutually_exclusive']:
                # Penalty for selecting both: P * x_i * x_j
                Q[i, j] += P
                logger.info(f"Added mutually exclusive constraint: {i} and {j}")
        
        # Dependency constraints (if i then j)
        if 'dependencies' in constraints:
            for i, j in constraints['dependencies']:
                # Penalty for selecting i without j: P * x_i * (1 - x_j)
                # = P * x_i - P * x_i * x_j
                Q[i, i] += P
                Q[i, j] -= P
                logger.info(f"Added dependency constraint: {i} requires {j}")
        
        # At least one constraint
        if 'at_least_one' in constraints:
            for group in constraints['at_least_one']:
                # Penalty for selecting none: P * (1 - sum(x_i))^2
                # = P * (1 - 2*sum(x_i) + sum(x_i)^2)
                for i in group:
                    Q[i, i] += P * (1 - 2)
                    for j in group:
                        if j > i:
                            Q[i, j] += 2 * P
                logger.info(f"Added at-least-one constraint for group of size {len(group)}")
        
        return Q
    
    def _log_matrix_stats(self, Q: np.ndarray):
        """Log QUBO matrix statistics."""
        diag = Q.diagonal()
        off_diag = Q[~np.eye(Q.shape[0], dtype=bool)]
        
        logger.info("QUBO Matrix Statistics:")
        logger.info(f"  Shape: {Q.shape}")
        logger.info(f"  Diagonal: min={diag.min():.2f}, max={diag.max():.2f}, mean={diag.mean():.2f}")
        logger.info(f"  Off-diagonal: min={off_diag.min():.2f}, max={off_diag.max():.2f}, mean={off_diag.mean():.2f}")
        logger.info(f"  Non-zero elements: {np.count_nonzero(Q)}")
        logger.info(f"  Condition number: {np.linalg.cond(Q):.2f}")
    
    def qubo_to_ising(self, Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Convert QUBO to Ising model parameters.
        
        Returns:
            h: Local fields (single-qubit terms)
            J: Couplings (two-qubit terms)
            offset: Constant offset
        """
        n = Q.shape[0]
        h = np.zeros(n)
        J = np.zeros((n, n))
        offset = 0
        
        for i in range(n):
            h[i] = Q[i, i] / 2
            offset += Q[i, i] / 4
            for j in range(i+1, n):
                J[i, j] = Q[i, j] / 4
                h[i] += Q[i, j] / 4
                h[j] += Q[i, j] / 4
                offset += Q[i, j] / 4
        
        logger.info(f"Converted to Ising: h range=[{h.min():.2f}, {h.max():.2f}], "
                   f"J range=[{J[J!=0].min() if J.any() else 0:.2f}, {J.max():.2f}], "
                   f"offset={offset:.2f}")
        
        return h, J, offset
    
    def ising_to_qubo(self, h: np.ndarray, J: np.ndarray, offset: float) -> np.ndarray:
        """
        Convert Ising parameters back to QUBO.
        
        Args:
            h: Local fields
            J: Couplings
            offset: Constant offset
            
        Returns:
            QUBO matrix
        """
        n = len(h)
        Q = np.zeros((n, n))
        
        for i in range(n):
            Q[i, i] = 2 * h[i]
            for j in range(i+1, n):
                Q[i, j] = 4 * J[i, j]
                Q[j, i] = Q[i, j]
        
        # Adjust for offset (distribute evenly)
        Q += 2 * offset / n
        
        return Q
    
    def evaluate_solution(self, solution: List[int], Q: np.ndarray) -> float:
        """
        Evaluate a solution's energy.
        
        Args:
            solution: List of selected indices
            Q: QUBO matrix
            
        Returns:
            Energy value (lower is better)
        """
        n = Q.shape[0]
        x = np.zeros(n)
        x[solution] = 1
        
        # Compute x^T Q x
        energy = 0
        for i in range(n):
            if x[i] > 0:
                energy += Q[i, i]
                for j in range(i+1, n):
                    if x[j] > 0:
                        energy += 2 * Q[i, j]
        
        return energy
    
    def get_feasibility_report(self, solution: List[int]) -> Dict:
        """
        Check if solution satisfies all constraints.
        
        Returns:
            Dictionary with constraint satisfaction status
        """
        report = {
            'feasible': True,
            'cardinality': len(solution) == self.n_sites_to_select,
            'min_distance': True,
            'risk_threshold': True,
            'violations': []
        }
        
        # Check cardinality
        if len(solution) != self.n_sites_to_select:
            report['feasible'] = False
            report['violations'].append(f"Cardinality: {len(solution)} != {self.n_sites_to_select}")
        
        # Check minimum distance
        if self.distance_matrix is not None and len(solution) > 1:
            for i in range(len(solution)):
                for j in range(i+1, len(solution)):
                    dist = self.distance_matrix[solution[i], solution[j]]
                    if dist < self.params.min_distance_km:
                        report['feasible'] = False
                        report['min_distance'] = False
                        report['violations'].append(
                            f"Distance: {solution[i]}-{solution[j]} = {dist:.2f}km < {self.params.min_distance_km}km"
                        )
        
        return report


# Utility functions for quick QUBO building
def create_qubo_from_data(
    df,
    n_select: int = 5,
    solar_col: str = 'Solar_Irradiance',
    wind_col: str = 'Wind_Speed',
    risk_col: str = 'Risk_Score',
    lat_col: str = 'Latitude',
    lon_col: str = 'Longitude',
    grid_col: Optional[str] = 'Grid_Distance',
    access_col: Optional[str] = 'Accessibility'
) -> Tuple[np.ndarray, QUBOFormulation]:
    """
    Quick QUBO creation from DataFrame.
    
    Example:
        >>> Q, qubo = create_qubo_from_data(df, n_select=5)
    """
    # Extract data
    solar = df[solar_col].values
    wind = df[wind_col].values
    risk = df[risk_col].values
    coords = df[[lat_col, lon_col]].values
    
    grid = df[grid_col].values if grid_col in df.columns else None
    access = df[access_col].values if access_col in df.columns else None
    
    # Create QUBO
    qubo = QUBOFormulation(n_sites_to_select=n_select)
    Q = qubo.build_qubo(solar, wind, risk, coords, grid, access)
    
    return Q, qubo


# Example usage
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    n = 20
    
    solar = np.random.uniform(4.5, 6.0, n)
    wind = np.random.uniform(2.5, 6.5, n)
    risk = np.random.randint(2, 10, n)
    coords = np.random.uniform([31.25, 34.20], [31.58, 34.55], (n, 2))
    grid = np.random.randint(100, 5000, n)
    access = np.random.choice([0, 1], n, p=[0.2, 0.8])
    
    # Create QUBO
    qubo = QUBOFormulation(n_sites_to_select=5)
    Q = qubo.build_qubo(solar, wind, risk, coords, grid, access)
    
    print(f"QUBO matrix shape: {Q.shape}")
    print(f"Sample solution energy: {qubo.evaluate_solution([0,1,2,3,4], Q):.2f}")
    
    # Convert to Ising
    h, J, offset = qubo.qubo_to_ising(Q)
    print(f"Ising: {len(h)} qubits, offset={offset:.2f}")
