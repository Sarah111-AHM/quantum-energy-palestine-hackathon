"""
Classical Optimization Baselines for Site Selection Problem.
Provides classical algorithms to compare against quantum solutions.
"""

import numpy as np
import logging
from typing import List, Tuple, Dict, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import itertools
from scipy.optimize import minimize
from sklearn.cluster import KMeans

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClassicalMethod(Enum):
    """Types of classical optimization methods."""
    GREEDY = "greedy"
    EXACT = "exact"  # Brute force (only for small n)
    GENETIC = "genetic"
    SIMULATED_ANNEALING = "simulated_annealing"
    KMEANS = "kmeans"
    RANDOM = "random"
    MULTI_OBJECTIVE = "multi_objective"


@dataclass
class ClassicalResult:
    """Container for classical optimization results."""
    selected_indices: List[int]
    score: float
    method: ClassicalMethod
    execution_time: float
    iterations: int
    convergence_history: List[float]
    feasibility: bool
    metadata: Dict


class ClassicalBaselines:
    """
    Classical optimization methods for site selection.
    
    Provides multiple algorithms to:
    - Benchmark quantum advantage
    - Fallback when quantum fails
    - Compare solution quality
    - Validate quantum results
    """
    
    def __init__(self, n_sites_to_select: int = 5, random_seed: int = 42):
        """
        Initialize classical baselines.
        
        Args:
            n_sites_to_select: Number of sites to select (K)
            random_seed: Random seed for reproducibility
        """
        self.n_sites_to_select = n_sites_to_select
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
        logger.info(f"Initialized ClassicalBaselines with K={n_sites_to_select}")
    
    def greedy_selection(
        self,
        scores: np.ndarray,
        risks: np.ndarray,
        coords: np.ndarray,
        grid_distances: Optional[np.ndarray] = None,
        risk_weight: float = 0.5,
        min_distance_km: float = 2.0
    ) -> ClassicalResult:
        """
        Greedy selection algorithm.
        
        Strategy:
        1. Sort sites by score/risk ratio
        2. Select highest while maintaining constraints
        3. Enforce minimum distance
        
        Args:
            scores: Site suitability scores
            risks: Risk indices
            coords: Geographic coordinates
            grid_distances: Distance to grid
            risk_weight: Weight for risk penalty
            min_distance_km: Minimum distance between sites
            
        Returns:
            ClassicalResult with selected indices
        """
        import time
        start_time = time.time()
        
        n = len(scores)
        logger.info("Running greedy selection...")
        
        # Calculate adjusted scores
        adjusted_scores = scores.copy()
        
        # Apply risk penalty
        risk_penalty = 1 - (risks / 10) * risk_weight
        adjusted_scores *= risk_penalty
        
        # Apply grid proximity bonus
        if grid_distances is not None:
            grid_norm = 1 - (grid_distances / grid_distances.max())
            adjusted_scores *= (1 + 0.3 * grid_norm)
        
        # Sort by adjusted score
        sorted_indices = np.argsort(adjusted_scores)[::-1]
        
        # Greedy selection with distance constraint
        selected = []
        convergence = []
        
        for idx in sorted_indices:
            if len(selected) >= self.n_sites_to_select:
                break
            
            # Check distance constraint
            valid = True
            if min_distance_km > 0:
                for sel_idx in selected:
                    dist = self._haversine_distance(coords[idx], coords[sel_idx])
                    if dist < min_distance_km:
                        valid = False
                        break
            
            if valid:
                selected.append(idx)
                convergence.append(len(selected))
        
        # If we don't have enough, add best remaining regardless of distance
        if len(selected) < self.n_sites_to_select:
            remaining = [i for i in sorted_indices if i not in selected]
            selected.extend(remaining[:self.n_sites_to_select - len(selected)])
        
        # Calculate score
        final_score = self._evaluate_solution(selected, scores, risks, coords)
        
        execution_time = time.time() - start_time
        
        logger.info(f"Greedy selection complete: {len(selected)} sites, score={final_score:.2f}")
        
        return ClassicalResult(
            selected_indices=selected,
            score=final_score,
            method=ClassicalMethod.GREEDY,
            execution_time=execution_time,
            iterations=len(selected),
            convergence_history=convergence,
            feasibility=len(selected) == self.n_sites_to_select,
            metadata={'adjusted_scores': adjusted_scores.tolist()}
        )
    
    def exact_brute_force(
        self,
        scores: np.ndarray,
        risks: np.ndarray,
        coords: np.ndarray,
        max_combinations: int = 10000
    ) -> ClassicalResult:
        """
        Exact brute force optimization.
        Only feasible for small n (n < 20).
        
        Args:
            scores: Site suitability scores
            risks: Risk indices
            coords: Geographic coordinates
            max_combinations: Maximum combinations to try
            
        Returns:
            ClassicalResult with optimal selection
        """
        import time
        start_time = time.time()
        
        n = len(scores)
        
        # Check if brute force is feasible
        n_combinations = self._nCr(n, self.n_sites_to_select)
        
        if n_combinations > max_combinations:
            logger.warning(f"Too many combinations ({n_combinations}), using sampling")
            return self.random_search(scores, risks, coords, n_iterations=1000)
        
        logger.info(f"Running exact brute force with {n_combinations} combinations...")
        
        # Generate all combinations
        indices = list(range(n))
        best_score = -float('inf')
        best_selection = None
        convergence = []
        
        for i, combo in enumerate(itertools.combinations(indices, self.n_sites_to_select)):
            score = self._evaluate_solution(list(combo), scores, risks, coords)
            
            if score > best_score:
                best_score = score
                best_selection = list(combo)
                convergence.append(score)
            
            if i % 1000 == 0:
                logger.debug(f"Checked {i} combinations...")
        
        execution_time = time.time() - start_time
        
        logger.info(f"Exact optimization complete: score={best_score:.2f}, time={execution_time:.2f}s")
        
        return ClassicalResult(
            selected_indices=best_selection,
            score=best_score,
            method=ClassicalMethod.EXACT,
            execution_time=execution_time,
            iterations=n_combinations,
            convergence_history=convergence,
            feasibility=True,
            metadata={'n_combinations': n_combinations}
        )
    
    def genetic_algorithm(
        self,
        scores: np.ndarray,
        risks: np.ndarray,
        coords: np.ndarray,
        population_size: int = 100,
        generations: int = 100,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.8,
        elite_size: int = 5,
        callback: Optional[Callable] = None
    ) -> ClassicalResult:
        """
        Genetic algorithm for site selection.
        
        Args:
            scores: Site suitability scores
            risks: Risk indices
            coords: Geographic coordinates
            population_size: Size of population
            generations: Number of generations
            mutation_rate: Probability of mutation
            crossover_rate: Probability of crossover
            elite_size: Number of elite individuals to keep
            callback: Progress callback function
            
        Returns:
            ClassicalResult with best selection
        """
        import time
        start_time = time.time()
        
        n = len(scores)
        logger.info(f"Running genetic algorithm with pop={population_size}, gens={generations}")
        
        # Initialize population
        population = self._initialize_population(population_size, n)
        
        best_score = -float('inf')
        best_individual = None
        convergence_history = []
        
        for gen in range(generations):
            # Evaluate fitness
            fitness = []
            for individual in population:
                score = self._evaluate_solution(individual, scores, risks, coords)
                fitness.append(score)
                
                if score > best_score:
                    best_score = score
                    best_individual = individual.copy()
            
            convergence_history.append(best_score)
            
            if callback:
                callback(f"Generation {gen}: best={best_score:.2f}")
            
            # Select parents
            parents = self._selection(population, fitness, population_size)
            
            # Create new population
            new_population = []
            
            # Elitism
            elite_indices = np.argsort(fitness)[-elite_size:]
            for idx in elite_indices:
                new_population.append(population[idx].copy())
            
            # Crossover and mutation
            while len(new_population) < population_size:
                if np.random.random() < crossover_rate and len(parents) >= 2:
                    # Crossover
                    p1, p2 = np.random.choice(len(parents), 2, replace=False)
                    child1, child2 = self._crossover(
                        parents[p1], parents[p2], n
                    )
                    
                    # Mutation
                    if np.random.random() < mutation_rate:
                        child1 = self._mutate(child1, n)
                    if np.random.random() < mutation_rate:
                        child2 = self._mutate(child2, n)
                    
                    new_population.append(child1)
                    if len(new_population) < population_size:
                        new_population.append(child2)
                else:
                    # Copy parent with mutation
                    parent = parents[np.random.randint(len(parents))]
                    child = parent.copy()
                    if np.random.random() < mutation_rate:
                        child = self._mutate(child, n)
                    new_population.append(child)
            
            population = new_population[:population_size]
        
        execution_time = time.time() - start_time
        
        logger.info(f"Genetic algorithm complete: best={best_score:.2f}, time={execution_time:.2f}s")
        
        return ClassicalResult(
            selected_indices=best_individual,
            score=best_score,
            method=ClassicalMethod.GENETIC,
            execution_time=execution_time,
            iterations=generations * population_size,
            convergence_history=convergence_history,
            feasibility=len(best_individual) == self.n_sites_to_select,
            metadata={'generations': generations, 'population_size': population_size}
        )
    
    def simulated_annealing(
        self,
        scores: np.ndarray,
        risks: np.ndarray,
        coords: np.ndarray,
        initial_temp: float = 100.0,
        cooling_rate: float = 0.95,
        min_temp: float = 1e-3,
        iterations_per_temp: int = 100,
        callback: Optional[Callable] = None
    ) -> ClassicalResult:
        """
        Simulated annealing algorithm.
        
        Args:
            scores: Site suitability scores
            risks: Risk indices
            coords: Geographic coordinates
            initial_temp: Initial temperature
            cooling_rate: Cooling factor per iteration
            min_temp: Minimum temperature
            iterations_per_temp: Iterations at each temperature
            callback: Progress callback function
            
        Returns:
            ClassicalResult with best selection
        """
        import time
        start_time = time.time()
        
        n = len(scores)
        logger.info(f"Running simulated annealing: T0={initial_temp}, rate={cooling_rate}")
        
        # Initialize random solution
        current = self._random_solution(n)
        current_score = self._evaluate_solution(current, scores, risks, coords)
        
        best = current.copy()
        best_score = current_score
        
        temp = initial_temp
        iteration = 0
        convergence_history = [best_score]
        
        while temp > min_temp:
            for _ in range(iterations_per_temp):
                # Generate neighbor
                neighbor = self._neighbor_solution(current, n)
                neighbor_score = self._evaluate_solution(neighbor, scores, risks, coords)
                
                # Acceptance probability
                delta = neighbor_score - current_score
                
                if delta > 0 or np.random.random() < np.exp(delta / temp):
                    current = neighbor
                    current_score = neighbor_score
                    
                    if current_score > best_score:
                        best = current.copy()
                        best_score = current_score
                
                iteration += 1
                
                if iteration % 100 == 0 and callback:
                    callback(f"Iter {iteration}: T={temp:.2f}, best={best_score:.2f}")
            
            convergence_history.append(best_score)
            temp *= cooling_rate
        
        execution_time = time.time() - start_time
        
        logger.info(f"Simulated annealing complete: best={best_score:.2f}, time={execution_time:.2f}s")
        
        return ClassicalResult(
            selected_indices=best,
            score=best_score,
            method=ClassicalMethod.SIMULATED_ANNEALING,
            execution_time=execution_time,
            iterations=iteration,
            convergence_history=convergence_history,
            feasibility=len(best) == self.n_sites_to_select,
            metadata={'final_temp': temp}
        )
    
    def kmeans_clustering(
        self,
        scores: np.ndarray,
        risks: np.ndarray,
        coords: np.ndarray,
        use_weighted: bool = True
    ) -> ClassicalResult:
        """
        K-means clustering based selection.
        Groups sites into clusters and selects best from each.
        
        Args:
            scores: Site suitability scores
            risks: Risk indices
            coords: Geographic coordinates
            use_weighted: Use weighted clustering
            
        Returns:
            ClassicalResult with selected indices
        """
        import time
        start_time = time.time()
        
        n = len(scores)
        logger.info(f"Running K-means clustering with K={self.n_sites_to_select}")
        
        # Prepare data for clustering
        if use_weighted:
            # Weight coordinates by score for better clustering
            weights = scores * (1 - risks/10)
            weighted_coords = coords * weights.reshape(-1, 1)
            X = weighted_coords
        else:
            X = coords
        
        # Apply K-means
        kmeans = KMeans(
            n_clusters=self.n_sites_to_select,
            random_state=self.random_seed,
            n_init=10
        )
        labels = kmeans.fit_predict(X)
        
        # Select best site from each cluster
        selected = []
        for cluster in range(self.n_sites_to_select):
            cluster_indices = np.where(labels == cluster)[0]
            
            if len(cluster_indices) > 0:
                # Score sites in cluster
                cluster_scores = scores[cluster_indices]
                risk_penalty = 1 - (risks[cluster_indices] / 10)
                adjusted = cluster_scores * risk_penalty
                
                # Select best
                best_in_cluster = cluster_indices[np.argmax(adjusted)]
                selected.append(int(best_in_cluster))
        
        # Ensure we have exactly K
        if len(selected) < self.n_sites_to_select:
            remaining = [i for i in range(n) if i not in selected]
            remaining_scores = scores[remaining] * (1 - risks[remaining]/10)
            additional = np.argsort(remaining_scores)[-self.n_sites_to_select + len(selected):]
            selected.extend(remaining[i] for i in additional)
        
        final_score = self._evaluate_solution(selected, scores, risks, coords)
        execution_time = time.time() - start_time
        
        logger.info(f"K-means complete: score={final_score:.2f}")
        
        return ClassicalResult(
            selected_indices=selected,
            score=final_score,
            method=ClassicalMethod.KMEANS,
            execution_time=execution_time,
            iterations=kmeans.n_iter_,
            convergence_history=[],
            feasibility=len(selected) == self.n_sites_to_select,
            metadata={'inertia': kmeans.inertia_}
        )
    
    def random_search(
        self,
        scores: np.ndarray,
        risks: np.ndarray,
        coords: np.ndarray,
        n_iterations: int = 1000
    ) -> ClassicalResult:
        """
        Random search baseline.
        
        Args:
            scores: Site suitability scores
            risks: Risk indices
            coords: Geographic coordinates
            n_iterations: Number of random samples
            
        Returns:
            ClassicalResult with best random selection
        """
        import time
        start_time = time.time()
        
        n = len(scores)
        logger.info(f"Running random search with {n_iterations} iterations")
        
        best_score = -float('inf')
        best_selection = None
        convergence = []
        
        for i in range(n_iterations):
            # Generate random solution
            selection = self._random_solution(n)
            score = self._evaluate_solution(selection, scores, risks, coords)
            
            if score > best_score:
                best_score = score
                best_selection = selection
                convergence.append(score)
        
        execution_time = time.time() - start_time
        
        logger.info(f"Random search complete: best={best_score:.2f}")
        
        return ClassicalResult(
            selected_indices=best_selection,
            score=best_score,
            method=ClassicalMethod.RANDOM,
            execution_time=execution_time,
            iterations=n_iterations,
            convergence_history=convergence,
            feasibility=len(best_selection) == self.n_sites_to_select,
            metadata={'n_iterations': n_iterations}
        )
    
    def multi_objective_optimization(
        self,
        scores: np.ndarray,
        risks: np.ndarray,
        coords: np.ndarray,
        grid_distances: Optional[np.ndarray] = None,
        weights: Optional[List[float]] = None
    ) -> ClassicalResult:
        """
        Multi-objective optimization using weighted sum.
        
        Args:
            scores: Site suitability scores
            risks: Risk indices
            coords: Geographic coordinates
            grid_distances: Distance to grid
            weights: Weights for different objectives
            
        Returns:
            ClassicalResult with selected indices
        """
        import time
        start_time = time.time()
        
        n = len(scores)
        
        if weights is None:
            weights = [0.4, 0.3, 0.2, 0.1]  # solar, wind, risk, grid
        
        logger.info("Running multi-objective optimization...")
        
        # Normalize all objectives
        solar_norm = scores / scores.max()
        wind_norm = np.random.rand(n)  # Placeholder - use actual wind data
        risk_norm = 1 - (risks / 10)
        
        # Combined objective
        combined = (
            weights[0] * solar_norm +
            weights[1] * wind_norm +
            weights[2] * risk_norm
        )
        
        if grid_distances is not None:
            grid_norm = 1 - (grid_distances / grid_distances.max())
            combined += weights[3] * grid_norm
        
        # Select top K
        selected = np.argsort(combined)[-self.n_sites_to_select:].tolist()
        
        final_score = self._evaluate_solution(selected, scores, risks, coords)
        execution_time = time.time() - start_time
        
        logger.info(f"Multi-objective complete: score={final_score:.2f}")
        
        return ClassicalResult(
            selected_indices=selected,
            score=final_score,
            method=ClassicalMethod.MULTI_OBJECTIVE,
            execution_time=execution_time,
            iterations=1,
            convergence_history=[],
            feasibility=len(selected) == self.n_sites_to_select,
            metadata={'weights': weights}
        )
    
    def compare_all_methods(
        self,
        scores: np.ndarray,
        risks: np.ndarray,
        coords: np.ndarray,
        grid_distances: Optional[np.ndarray] = None
    ) -> Dict[ClassicalMethod, ClassicalResult]:
        """
        Run all classical methods and compare results.
        
        Returns:
            Dictionary of results by method
        """
        logger.info("Running all classical methods for comparison...")
        
        results = {}
        
        # Greedy
        results[ClassicalMethod.GREEDY] = self.greedy_selection(
            scores, risks, coords, grid_distances
        )
        
        # Random search
        results[ClassicalMethod.RANDOM] = self.random_search(
            scores, risks, coords
        )
        
        # Genetic algorithm
        results[ClassicalMethod.GENETIC] = self.genetic_algorithm(
            scores, risks, coords
        )
        
        # Simulated annealing
        results[ClassicalMethod.SIMULATED_ANNEALING] = self.simulated_annealing(
            scores, risks, coords
        )
        
        # K-means
        results[ClassicalMethod.KMEANS] = self.kmeans_clustering(
            scores, risks, coords
        )
        
        # Multi-objective
        results[ClassicalMethod.MULTI_OBJECTIVE] = self.multi_objective_optimization(
            scores, risks, coords, grid_distances
        )
        
        # Exact for small n
        if len(scores) <= 15:
            results[ClassicalMethod.EXACT] = self.exact_brute_force(
                scores, risks, coords
            )
        
        # Log comparison
        logger.info("=== Method Comparison ===")
        for method, result in results.items():
            logger.info(f"{method.value:20s}: score={result.score:8.2f}, time={result.execution_time:6.3f}s")
        
        return results
    
    def _initialize_population(self, pop_size: int, n: int) -> List[List[int]]:
        """Initialize random population."""
        population = []
        for _ in range(pop_size):
            population.append(self._random_solution(n))
        return population
    
    def _random_solution(self, n: int) -> List[int]:
        """Generate random valid solution."""
        indices = list(range(n))
        selected = list(np.random.choice(indices, self.n_sites_to_select, replace=False))
        return sorted(selected)
    
    def _selection(
        self,
        population: List[List[int]],
        fitness: List[float],
        pop_size: int
    ) -> List[List[int]]:
        """Tournament selection."""
        parents = []
        tournament_size = 3
        
        for _ in range(pop_size):
            # Tournament
            tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
            tournament_fitness = [fitness[i] for i in tournament_indices]
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            parents.append(population[winner_idx].copy())
        
        return parents
    
    def _crossover(self, p1: List[int], p2: List[int], n: int) -> Tuple[List[int], List[int]]:
        """Crossover two parents."""
        # Convert to sets for easier manipulation
        set1 = set(p1)
        set2 = set(p2)
        
        # Single-point crossover on the sets
        crossover_point = np.random.randint(1, self.n_sites_to_select - 1)
        
        # Child1 takes first crossover_point from p1, rest from p2
        child1 = list(p1[:crossover_point])
        remaining = [x for x in p2 if x not in child1]
        child1.extend(remaining[:self.n_sites_to_select - len(child1)])
        
        # Child2 takes first crossover_point from p2, rest from p1
        child2 = list(p2[:crossover_point])
        remaining = [x for x in p1 if x not in child2]
        child2.extend(remaining[:self.n_sites_to_select - len(child2)])
        
        return sorted(child1), sorted(child2)
    
    def _mutate(self, individual: List[int], n: int) -> List[int]:
        """Mutate individual by swapping a site."""
        if len(individual) < n:
            # Swap one selected with one unselected
            unselected = [i for i in range(n) if i not in individual]
            if unselected:
                idx_to_remove = np.random.randint(len(individual))
                idx_to_add = np.random.randint(len(unselected))
                
                new_individual = individual.copy()
                new_individual[idx_to_remove] = unselected[idx_to_add]
                return sorted(new_individual)
        
        return individual
    
    def _neighbor_solution(self, solution: List[int], n: int) -> List[int]:
        """Generate neighbor solution for simulated annealing."""
        # Swap one site
        unselected = [i for i in range(n) if i not in solution]
        if unselected:
            idx_to_remove = np.random.randint(len(solution))
            idx_to_add = np.random.randint(len(unselected))
            
            neighbor = solution.copy()
            neighbor[idx_to_remove] = unselected[idx_to_add]
            return sorted(neighbor)
        
        return solution
    
    def _evaluate_solution(
        self,
        solution: List[int],
        scores: np.ndarray,
        risks: np.ndarray,
        coords: np.ndarray
    ) -> float:
        """Evaluate solution quality."""
        if len(solution) != self.n_sites_to_select:
            return -float('inf')
        
        # Sum of scores
        score_sum = sum(scores[i] for i in solution)
        
        # Risk penalty
        risk_penalty = sum(risks[i] for i in solution) * 10
        
        # Diversity bonus (based on distances)
        diversity = 0
        if len(solution) > 1:
            for i in range(len(solution)):
                for j in range(i+1, len(solution)):
                    dist = self._haversine_distance(
                        coords[solution[i]],
                        coords[solution[j]]
                    )
                    diversity += dist
        
        return score_sum - risk_penalty + diversity * 100
    
    def _haversine_distance(self, coord1: np.ndarray, coord2: np.ndarray) -> float:
        """Calculate Haversine distance in kilometers."""
        lat1, lon1 = np.radians(coord1)
        lat2, lon2 = np.radians(coord2)
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        return c * 6371  # Earth radius in km
    
    def _nCr(self, n: int, r: int) -> int:
        """Calculate number of combinations."""
        from math import comb
        return comb(n, r)


# Utility function for quick comparison
def benchmark_classical_methods(
    scores: np.ndarray,
    risks: np.ndarray,
    coords: np.ndarray,
    n_select: int = 5
) -> pd.DataFrame:
    """
    Benchmark all classical methods and return comparison DataFrame.
    
    Example:
        >>> df = benchmark_classical_methods(scores, risks, coords, n_select=5)
    """
    import pandas as pd
    
    baselines = ClassicalBaselines(n_sites_to_select=n_select)
    results = baselines.compare_all_methods(scores, risks, coords)
    
    data = []
    for method, result in results.items():
        data.append({
            'Method': method.value,
            'Score': round(result.score, 2),
            'Time (s)': round(result.execution_time, 3),
            'Iterations': result.iterations,
            'Feasible': result.feasibility
        })
    
    df = pd.DataFrame(data)
    df = df.sort_values('Score', ascending=False)
    
    return df


# Example usage
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    n = 30
    
    scores = np.random.uniform(0, 100, n)
    risks = np.random.randint(2, 10, n)
    coords = np.random.uniform([31.25, 34.20], [31.58, 34.55], (n, 2))
    
    # Initialize baselines
    baselines = ClassicalBaselines(n_sites_to_select=5)
    
    # Run comparison
    results = baselines.compare_all_methods(scores, risks, coords)
    
    # Print best method
    best_method = max(results.items(), key=lambda x: x[1].score)
    print(f"\nBest method: {best_method[0].value} with score {best_method[1].score:.2f}")
    
    # Create comparison table
    try:
        import pandas as pd
        df = benchmark_classical_methods(scores, risks, coords)
        print("\nComparison Table:")
        print(df.to_string(index=False))
    except ImportError:
        pass
