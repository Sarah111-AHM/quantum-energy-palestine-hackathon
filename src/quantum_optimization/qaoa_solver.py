"""
Quantum Approximate Optimization Algorithm (QAOA) Solver
for optimal renewable energy site selection in Gaza Strip.

This module implements a robust QAOA solver with multiple fallback strategies,
noise resilience, and adaptive parameter optimization.
"""

import numpy as np
from qiskit import QuantumCircuit, ClassicalRegister
from qiskit_algorithms.optimizers import COBYLA, SPSA, ADAM
from qiskit_algorithms.minimum_eigensolvers import QAOA
from qiskit.primitives import StatevectorSampler, BackendSampler
from qiskit.quantum_info import SparsePauliOp, Pauli
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
import warnings
import logging
from typing import List, Tuple, Optional, Dict, Callable
from dataclasses import dataclass
from enum import Enum
import time

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptimizationMode(Enum):
    """Optimization modes for different scenarios."""
    QUANTUM = "quantum"
    HYBRID = "hybrid"
    CLASSICAL = "classical"
    ROBUST = "robust"

@dataclass
class OptimizationResult:
    """Container for optimization results."""
    selected_indices: List[int]
    energy: float
    mode: OptimizationMode
    bitstring: str
    probabilities: Dict[str, float]
    execution_time: float
    iterations: int
    convergence_history: List[float]

class QAOASolver:
    """
    Enhanced QAOA solver for site selection optimization.
    
    Features:
    - Multiple optimization strategies with automatic fallback
    - Noise-resilient circuit design
    - Adaptive parameter optimization
    - Comprehensive result tracking
    - Real quantum hardware preparation
    """
    
    def __init__(
        self,
        n_sites_to_select: int = 5,
        qaoa_layers: int = 2,
        backend_type: str = "aer_simulator",
        noise_model: Optional[NoiseModel] = None,
        optimization_mode: OptimizationMode = OptimizationMode.HYBRID,
        shots: int = 1024,
        seed: int = 42
    ):
        """
        Initialize the QAOA solver.
        
        Args:
            n_sites_to_select: Number of sites to select (K)
            qaoa_layers: Number of QAOA layers (p)
            backend_type: Type of backend to use
            noise_model: Optional noise model for realistic simulation
            optimization_mode: Optimization strategy to use
            shots: Number of measurement shots
            seed: Random seed for reproducibility
        """
        self.n_sites_to_select = n_sites_to_select
        self.qaoa_layers = qaoa_layers
        self.backend_type = backend_type
        self.noise_model = noise_model
        self.optimization_mode = optimization_mode
        self.shots = shots
        self.seed = seed
        
        # Set random seed
        np.random.seed(seed)
        
        # Initialize backend
        self._initialize_backend()
        
        # Optimization history
        self.optimization_history = []
        self.convergence_history = []
        
        # Parameters for adaptive optimization
        self.parameter_bounds = {
            'gamma': (-np.pi, np.pi),
            'beta': (0, np.pi)
        }
        
        logger.info(f"Initialized QAOASolver with {qaoa_layers} layers, mode={optimization_mode.value}")
    
    def _initialize_backend(self):
        """Initialize quantum backend with optional noise model."""
        if self.backend_type == "aer_simulator":
            self.backend = AerSimulator()
            if self.noise_model:
                self.backend.set_options(noise_model=self.noise_model)
        elif self.backend_type == "statevector":
            self.backend = StatevectorSampler()
        else:
            # Default to basic simulator
            self.backend = AerSimulator(method='statevector')
        
        logger.info(f"Initialized backend: {self.backend_type}")
    
    def create_enhanced_qubo(
        self,
        suitability_scores: np.ndarray,
        coordinates: np.ndarray,
        risk_scores: np.ndarray,
        grid_distances: Optional[np.ndarray] = None,
        constraints: Optional[Dict] = None
    ) -> np.ndarray:
        """
        Create enhanced QUBO matrix with multiple objectives and constraints.
        
        Args:
            suitability_scores: MCDA scores for each site
            coordinates: Geographic coordinates (lat, lon)
            risk_scores: Risk indices (0-10)
            grid_distances: Distance to grid infrastructure
            constraints: Additional constraints dictionary
            
        Returns:
            QUBO matrix
        """
        n = len(suitability_scores)
        Q = np.zeros((n, n))
        
        # Normalize inputs for numerical stability
        scores_norm = self._normalize_array(suitability_scores)
        risks_norm = self._normalize_array(risk_scores)
        
        # ===== DIAGONAL TERMS (Site-specific weights) =====
        for i in range(n):
            # Primary objective: maximize suitability (negative for minimization)
            Q[i, i] = -scores_norm[i] * 100
            
            # Risk penalty (progressive based on risk level)
            risk_level = risks_norm[i]
            if risk_level > 0.7:  # High risk (>7/10)
                risk_penalty = 80 * risk_level
            elif risk_level > 0.5:  # Medium risk (5-7/10)
                risk_penalty = 40 * risk_level
            else:  # Low risk
                risk_penalty = 10 * risk_level
            
            Q[i, i] += risk_penalty
            
            # Grid proximity reward (if available)
            if grid_distances is not None:
                grid_norm = self._normalize_array(grid_distances)[i]
                Q[i, i] -= 30 * (1 - grid_norm)  # Reward proximity
            
            # Accessibility bonus
            if constraints and 'accessible' in constraints:
                if constraints['accessible'][i] == 1:
                    Q[i, i] -= 20  # Bonus for accessible sites
        
        # ===== OFF-DIAGONAL TERMS (Site interactions) =====
        # Calculate adaptive distance threshold
        distances = self._calculate_distance_matrix(coordinates)
        avg_distance = np.mean(distances[distances > 0])
        threshold = avg_distance * 0.3  # Adaptive threshold (30% of average)
        
        for i in range(n):
            for j in range(i+1, n):
                distance = distances[i, j]
                
                # Geographic diversity penalty
                if distance < threshold:
                    # Progressive penalty based on closeness
                    closeness = 1 - (distance / threshold)
                    penalty = 40 * closeness
                    Q[i, j] = penalty
                else:
                    # Small reward for good separation
                    reward = -5 * min(1, distance / (avg_distance * 2))
                    Q[i, j] = reward
                
                # Risk correlation penalty
                if abs(risk_scores[i] - risk_scores[j]) < 2:
                    if risk_scores[i] > 7 or risk_scores[j] > 7:
                        Q[i, j] += 30  # Penalty for clustering high-risk sites
        
        # ===== CONSTRAINTS =====
        # Exact cardinality constraint: exactly K sites selected
        penalty_strength = 500  # Strong constraint
        for i in range(n):
            Q[i, i] += penalty_strength * (1 - 2 * self.n_sites_to_select)
            for j in range(i+1, n):
                Q[i, j] += 2 * penalty_strength
        
        # Minimum distance constraint (no sites too close)
        min_distance_km = 2.0  # Minimum 2km separation
        for i in range(n):
            for j in range(i+1, n):
                if distances[i, j] < min_distance_km / 6371:  # Convert to radians
                    Q[i, j] += 1000  # Very strong penalty
        
        # Ensure symmetry
        Q = (Q + Q.T) / 2
        
        logger.info(f"Created QUBO matrix of size {n}x{n}")
        return Q
    
    def _normalize_array(self, arr: np.ndarray) -> np.ndarray:
        """Normalize array to [0, 1] range."""
        min_val = arr.min()
        max_val = arr.max()
        if max_val > min_val:
            return (arr - min_val) / (max_val - min_val)
        return np.zeros_like(arr)
    
    def _calculate_distance_matrix(self, coordinates: np.ndarray) -> np.ndarray:
        """Calculate Haversine distance matrix between all points."""
        n = len(coordinates)
        distances = np.zeros((n, n))
        
        # Convert to radians
        coords_rad = np.radians(coordinates)
        
        for i in range(n):
            for j in range(i+1, n):
                lat1, lon1 = coords_rad[i]
                lat2, lon2 = coords_rad[j]
                
                dlat = lat2 - lat1
                dlon = lon2 - lon1
                
                a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
                c = 2 * np.arcsin(np.sqrt(a))
                
                distances[i, j] = distances[j, i] = c
        
        return distances
    
    def qubo_to_ising(self, Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Convert QUBO matrix to Ising model parameters.
        
        Returns:
            h: Local fields
            J: Couplings
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
        
        logger.info(f"Converted to Ising: {n} qubits, offset={offset:.2f}")
        return h, J, offset
    
    def create_ising_hamiltonian(self, h: np.ndarray, J: np.ndarray) -> SparsePauliOp:
        """
        Create Ising Hamiltonian operator.
        
        Args:
            h: Local fields (single-qubit terms)
            J: Couplings (two-qubit terms)
            
        Returns:
            SparsePauliOp Hamiltonian
        """
        n = len(h)
        pauli_list = []
        coeffs = []
        
        # Single-qubit Z terms
        for i in range(n):
            if abs(h[i]) > 1e-10:
                pauli = ['I'] * n
                pauli[i] = 'Z'
                pauli_list.append(''.join(pauli))
                coeffs.append(h[i])
        
        # Two-qubit ZZ terms
        for i in range(n):
            for j in range(i+1, n):
                if abs(J[i, j]) > 1e-10:
                    pauli = ['I'] * n
                    pauli[i] = 'Z'
                    pauli[j] = 'Z'
                    pauli_list.append(''.join(pauli))
                    coeffs.append(J[i, j])
        
        if not pauli_list:
            # Return zero Hamiltonian if no terms
            pauli_list = ['I' * n]
            coeffs = [0.0]
        
        hamiltonian = SparsePauliOp(pauli_list, coeffs=coeffs)
        logger.info(f"Created Hamiltonian with {len(pauli_list)} terms")
        
        return hamiltonian
    
    def create_qaoa_circuit(
        self,
        n_qubits: int,
        gamma_params: np.ndarray,
        beta_params: np.ndarray,
        h: np.ndarray,
        J: np.ndarray
    ) -> QuantumCircuit:
        """
        Create QAOA circuit for the problem.
        
        Args:
            n_qubits: Number of qubits
            gamma_params: Gamma parameters for cost Hamiltonian
            beta_params: Beta parameters for mixer Hamiltonian
            h: Local fields
            J: Couplings
            
        Returns:
            QAOA quantum circuit
        """
        circuit = QuantumCircuit(n_qubits)
        
        # Initial state: uniform superposition
        circuit.h(range(n_qubits))
        
        # QAOA layers
        for layer in range(len(gamma_params)):
            # Cost Hamiltonian layer (problem unitary)
            for i in range(n_qubits):
                if abs(h[i]) > 1e-10:
                    circuit.rz(2 * gamma_params[layer] * h[i], i)
            
            for i in range(n_qubits):
                for j in range(i+1, n_qubits):
                    if abs(J[i, j]) > 1e-10:
                        circuit.rzz(2 * gamma_params[layer] * J[i, j], i, j)
            
            # Mixer Hamiltonian layer
            for i in range(n_qubits):
                circuit.rx(2 * beta_params[layer], i)
        
        logger.debug(f"Created QAOA circuit with {len(gamma_params)} layers")
        return circuit
    
    def solve(
        self,
        suitability_scores: np.ndarray,
        coordinates: np.ndarray,
        risk_scores: np.ndarray,
        grid_distances: Optional[np.ndarray] = None,
        constraints: Optional[Dict] = None,
        progress_callback: Optional[Callable] = None,
        max_attempts: int = 3
    ) -> OptimizationResult:
        """
        Main solving method with multiple strategies and fallbacks.
        
        Args:
            suitability_scores: MCDA scores for each site
            coordinates: Geographic coordinates
            risk_scores: Risk indices
            grid_distances: Distance to grid
            constraints: Additional constraints
            progress_callback: Callback for progress updates
            max_attempts: Maximum optimization attempts
            
        Returns:
            OptimizationResult object
        """
        start_time = time.time()
        n = len(suitability_scores)
        
        if progress_callback:
            progress_callback(f"Starting optimization with {n} sites")
        
        # Validate inputs
        if n < self.n_sites_to_select:
            logger.warning(f"Not enough sites: {n} < {self.n_sites_to_select}")
            return self._create_fallback_result(
                list(range(n)),
                mode=OptimizationMode.ROBUST,
                start_time=start_time
            )
        
        # Try optimization strategies in order
        strategies = [
            (OptimizationMode.QUANTUM, self._quantum_optimization),
            (OptimizationMode.HYBRID, self._hybrid_optimization),
            (OptimizationMode.CLASSICAL, self._classical_optimization),
            (OptimizationMode.ROBUST, self._robust_optimization)
        ]
        
        for mode, strategy in strategies:
            if self.optimization_mode == OptimizationMode.HYBRID:
                # In hybrid mode, try all but skip if earlier succeeds
                pass
            
            try:
                if progress_callback:
                    progress_callback(f"Trying {mode.value} optimization...")
                
                result = strategy(
                    suitability_scores, coordinates, risk_scores,
                    grid_distances, constraints, progress_callback
                )
                
                if result and self._validate_result(result, n):
                    execution_time = time.time() - start_time
                    result.execution_time = execution_time
                    
                    logger.info(f"Success with {mode.value} mode in {execution_time:.2f}s")
                    
                    if progress_callback:
                        progress_callback(f"✅ {mode.value} optimization successful!")
                    
                    return result
                    
            except Exception as e:
                logger.warning(f"{mode.value} optimization failed: {str(e)}")
                if progress_callback:
                    progress_callback(f"⚠️ {mode.value} failed, trying next...")
                continue
        
        # Ultimate fallback
        logger.error("All optimization strategies failed")
        return self._create_fallback_result(
            self._greedy_selection(suitability_scores, risk_scores),
            mode=OptimizationMode.ROBUST,
            start_time=start_time
        )
    
    def _quantum_optimization(
        self,
        scores: np.ndarray,
        coords: np.ndarray,
        risks: np.ndarray,
        grid_dist: Optional[np.ndarray],
        constraints: Optional[Dict],
        callback: Optional[Callable]
    ) -> Optional[OptimizationResult]:
        """
        Pure quantum optimization with QAOA.
        """
        if callback:
            callback("🔮 Creating QUBO matrix...")
        
        # Create QUBO
        Q = self.create_enhanced_qubo(scores, coords, risks, grid_dist, constraints)
        
        if callback:
            callback("⚡ Converting to Ising Hamiltonian...")
        
        # Convert to Ising
        h, J, offset = self.qubo_to_ising(Q)
        hamiltonian = self.create_ising_hamiltonian(h, J)
        
        if callback:
            callback(f"🎯 Running QAOA with {self.qaoa_layers} layers...")
        
        # Multiple attempts with different optimizers
        optimizers = [
            COBYLA(maxiter=200, tol=1e-3),
            SPSA(maxiter=200),
            ADAM(maxiter=200)
        ]
        
        best_result = None
        best_energy = float('inf')
        
        for idx, optimizer in enumerate(optimizers[:2]):  # Try first two
            try:
                if callback:
                    callback(f"Attempt {idx+1} with {optimizer.__class__.__name__}")
                
                # Setup QAOA
                sampler = StatevectorSampler()
                
                qaoa = QAOA(
                    sampler=sampler,
                    optimizer=optimizer,
                    reps=self.qaoa_layers,
                    initial_point=None
                )
                
                # Run optimization
                result = qaoa.compute_minimum_eigenvalue(hamiltonian)
                
                if result.eigenvalue.real < best_energy:
                    best_energy = result.eigenvalue.real
                    best_result = result
                    
            except Exception as e:
                logger.debug(f"Optimizer attempt {idx+1} failed: {str(e)}")
                continue
        
        if best_result is None:
            return None
        
        if callback:
            callback("📊 Decoding quantum solution...")
        
        # Extract solution
        selected, bitstring, probs = self._decode_quantum_solution(
            best_result, len(scores)
        )
        
        # Adjust if necessary
        selected = self._adjust_selection_count(selected, scores, risks)
        
        # Calculate energy
        energy = best_result.eigenvalue.real + offset
        
        return OptimizationResult(
            selected_indices=selected,
            energy=energy,
            mode=OptimizationMode.QUANTUM,
            bitstring=bitstring,
            probabilities=probs,
            execution_time=0,
            iterations=best_result.cost_function_evals if hasattr(best_result, 'cost_function_evals') else 0,
            convergence_history=self.convergence_history.copy()
        )
    
    def _hybrid_optimization(
        self,
        scores: np.ndarray,
        coords: np.ndarray,
        risks: np.ndarray,
        grid_dist: Optional[np.ndarray],
        constraints: Optional[Dict],
        callback: Optional[Callable]
    ) -> Optional[OptimizationResult]:
        """
        Hybrid quantum-classical optimization.
        Uses quantum for exploration and classical for refinement.
        """
        if callback:
            callback("🔄 Running hybrid optimization...")
        
        # Step 1: Quantum exploration (reduced layers)
        original_layers = self.qaoa_layers
        self.qaoa_layers = max(1, self.qaoa_layers - 1)
        
        quantum_result = self._quantum_optimization(
            scores, coords, risks, grid_dist, constraints, callback
        )
        
        self.qaoa_layers = original_layers
        
        if not quantum_result:
            return None
        
        # Step 2: Classical refinement
        if callback:
            callback("📈 Refining with classical optimization...")
        
        refined = self._refine_solution(
            quantum_result.selected_indices,
            scores, risks, coords
        )
        
        # Create result object
        return OptimizationResult(
            selected_indices=refined,
            energy=quantum_result.energy,
            mode=OptimizationMode.HYBRID,
            bitstring=quantum_result.bitstring,
            probabilities=quantum_result.probabilities,
            execution_time=0,
            iterations=quantum_result.iterations,
            convergence_history=quantum_result.convergence_history
        )
    
    def _classical_optimization(
        self,
        scores: np.ndarray,
        coords: np.ndarray,
        risks: np.ndarray,
        grid_dist: Optional[np.ndarray],
        constraints: Optional[Dict],
        callback: Optional[Callable]
    ) -> Optional[OptimizationResult]:
        """
        Classical optimization fallback.
        """
        if callback:
            callback("💻 Using classical optimization...")
        
        # Multi-objective scoring
        weights = scores / scores.sum()
        risk_penalty = 1 - (risks / 10)
        
        # Add grid proximity if available
        if grid_dist is not None:
            grid_norm = 1 - (grid_dist / grid_dist.max())
            combined = weights * risk_penalty * (0.7 + 0.3 * grid_norm)
        else:
            combined = weights * risk_penalty
        
        # Select top sites
        selected = np.argsort(combined)[-self.n_sites_to_select:].tolist()
        
        # Enforce minimum distance
        selected = self._enforce_min_distance(selected, coords, risks, scores)
        
        return OptimizationResult(
            selected_indices=selected,
            energy=combined[selected].sum(),
            mode=OptimizationMode.CLASSICAL,
            bitstring=self._indices_to_bitstring(selected, len(scores)),
            probabilities={},
            execution_time=0,
            iterations=0,
            convergence_history=[]
        )
    
    def _robust_optimization(
        self,
        scores: np.ndarray,
        coords: np.ndarray,
        risks: np.ndarray,
        grid_dist: Optional[np.ndarray],
        constraints: Optional[Dict],
        callback: Optional[Callable]
    ) -> OptimizationResult:
        """
        Ultra-robust fallback that always returns a valid solution.
        """
        if callback:
            callback("🛡️ Using robust fallback...")
        
        selected = self._greedy_selection(scores, risks)
        
        return OptimizationResult(
            selected_indices=selected,
            energy=0.0,
            mode=OptimizationMode.ROBUST,
            bitstring=self._indices_to_bitstring(selected, len(scores)),
            probabilities={},
            execution_time=0,
            iterations=0,
            convergence_history=[]
        )
    
    def _decode_quantum_solution(
        self,
        result,
        n_qubits: int
    ) -> Tuple[List[int], str, Dict[str, float]]:
        """
        Decode quantum result to site selection.
        
        Returns:
            selected_indices, bitstring, probabilities
        """
        eigenstate = result.eigenstate
        
        if hasattr(eigenstate, 'binary_probabilities'):
            probs = eigenstate.binary_probabilities()
            
            # Sort by probability
            sorted_states = sorted(probs.items(), key=lambda x: x[1], reverse=True)
            
            # Try top 5 states
            for bitstring, prob in sorted_states[:5]:
                selected = [i for i, bit in enumerate(bitstring) if bit == '1']
                if len(selected) == self.n_sites_to_select:
                    return selected, bitstring, probs
            
            # Return top state even if count is wrong
            top_bitstring = sorted_states[0][0]
            selected = [i for i, bit in enumerate(top_bitstring) if bit == '1']
            return selected, top_bitstring, probs
        
        # Fallback measurement simulation
        return self._simulate_measurement(result, n_qubits)
    
    def _simulate_measurement(
        self,
        result,
        n_qubits: int
    ) -> Tuple[List[int], str, Dict[str, float]]:
        """Simulate measurement if eigenstate doesn't have probabilities."""
        try:
            # Try to get statevector
            statevector = result.eigenstate
            if hasattr(statevector, 'data'):
                probs = np.abs(statevector.data) ** 2
                
                # Get most probable state
                max_idx = np.argmax(probs)
                bitstring = format(max_idx, f'0{n_qubits}b')
                
                selected = [i for i, bit in enumerate(bitstring) if bit == '1']
                
                # Create simple probability dict
                prob_dict = {bitstring: float(probs[max_idx])}
                
                return selected, bitstring, prob_dict
        except:
            pass
        
        # Ultimate fallback
        return [], '0' * n_qubits, {}
    
    def _refine_solution(
        self,
        initial: List[int],
        scores: np.ndarray,
        risks: np.ndarray,
        coords: np.ndarray
    ) -> List[int]:
        """Refine solution with local search."""
        current = set(initial)
        n = len(scores)
        
        # Try to improve by swapping
        improved = True
        max_iterations = 100
        
        while improved and max_iterations > 0:
            improved = False
            max_iterations -= 1
            
            current_score = self._evaluate_solution(current, scores, risks, coords)
            
            # Try swapping each selected site with an unselected one
            for selected in list(current):
                for candidate in range(n):
                    if candidate in current:
                        continue
                    
                    # Try swap
                    new_set = current.copy()
                    new_set.remove(selected)
                    new_set.add(candidate)
                    
                    new_score = self._evaluate_solution(new_set, scores, risks, coords)
                    
                    if new_score > current_score:
                        current = new_set
                        current_score = new_score
                        improved = True
                        break
                
                if improved:
                    break
        
        return sorted(list(current))
    
    def _evaluate_solution(
        self,
        solution: set,
        scores: np.ndarray,
        risks: np.ndarray,
        coords: np.ndarray
    ) -> float:
        """Evaluate solution quality."""
        score_sum = sum(scores[i] for i in solution)
        risk_penalty = sum(risks[i] for i in solution) * 10
        
        # Diversity bonus
        diversity = 0
        if len(solution) > 1:
            sol_list = list(solution)
            for i in range(len(sol_list)):
                for j in range(i+1, len(sol_list)):
                    lat1, lon1 = coords[sol_list[i]]
                    lat2, lon2 = coords[sol_list[j]]
                    dist = np.sqrt((lat1-lat2)**2 + (lon1-lon2)**2)
                    diversity += dist
        
        return score_sum - risk_penalty + diversity * 100
    
    def _enforce_min_distance(
        self,
        selected: List[int],
        coords: np.ndarray,
        risks: np.ndarray,
        scores: np.ndarray
    ) -> List[int]:
        """Enforce minimum distance between selected sites."""
        if len(selected) <= 1:
            return selected
        
        result = []
        min_distance_rad = 2.0 / 6371  # 2km in radians
        
        # Sort by score
        sorted_by_score = sorted(selected, key=lambda i: scores[i], reverse=True)
        
        for idx in sorted_by_score:
            too_close = False
            for selected_idx in result:
                lat1, lon1 = coords[idx]
                lat2, lon2 = coords[selected_idx]
                
                dlat = np.radians(lat2 - lat1)
                dlon = np.radians(lon2 - lon1)
                a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
                distance = 2 * np.arcsin(np.sqrt(a))
                
                if distance < min_distance_rad:
                    too_close = True
                    break
            
            if not too_close:
                result.append(idx)
            
            if len(result) >= self.n_sites_to_select:
                break
        
        # If we don't have enough, add the best remaining regardless of distance
        if len(result) < self.n_sites_to_select:
            remaining = [i for i in sorted_by_score if i not in result]
            result.extend(remaining[:self.n_sites_to_select - len(result)])
        
        return result[:self.n_sites_to_select]
    
    def _greedy_selection(
        self,
        scores: np.ndarray,
        risks: np.ndarray
    ) -> List[int]:
        """Greedy selection fallback."""
        # Penalize high-risk sites
        adjusted = scores.copy()
        for i in range(len(scores)):
            if risks[i] > 7:
                adjusted[i] *= 0.3
            elif risks[i] > 5:
                adjusted[i] *= 0.7
        
        return np.argsort(adjusted)[-self.n_sites_to_select:].tolist()
    
    def _adjust_selection_count(
        self,
        selected: List[int],
        scores: np.ndarray,
        risks: np.ndarray
    ) -> List[int]:
        """Adjust selection to exactly K sites."""
        current = set(selected)
        n = len(scores)
        
        if len(current) < self.n_sites_to_select:
            # Add more sites
            candidates = set(range(n)) - current
            if candidates:
                # Score candidates
                candidate_scores = []
                for i in candidates:
                    risk_penalty = 1.0 if risks[i] <= 7 else 0.5
                    candidate_scores.append((i, scores[i] * risk_penalty))
                
                candidate_scores.sort(key=lambda x: x[1], reverse=True)
                to_add = [i for i, _ in candidate_scores[:self.n_sites_to_select - len(current)]]
                current.update(to_add)
        
        elif len(current) > self.n_sites_to_select:
            # Remove worst sites
            site_scores = [(i, scores[i] * (1 if risks[i] <= 7 else 0.5)) for i in current]
            site_scores.sort(key=lambda x: x[1])
            to_remove = [i for i, _ in site_scores[:len(current) - self.n_sites_to_select]]
            current -= set(to_remove)
        
        return sorted(list(current))
    
    def _validate_result(self, result: OptimizationResult, n: int) -> bool:
        """Validate optimization result."""
        if not result:
            return False
        
        if len(result.selected_indices) != self.n_sites_to_select:
            return False
        
        if any(i < 0 or i >= n for i in result.selected_indices):
            return False
        
        return True
    
    def _indices_to_bitstring(self, indices: List[int], n: int) -> str:
        """Convert indices to bitstring."""
        bits = ['0'] * n
        for i in indices:
            bits[i] = '1'
        return ''.join(bits)
    
    def _create_fallback_result(
        self,
        indices: List[int],
        mode: OptimizationMode,
        start_time: float
    ) -> OptimizationResult:
        """Create fallback result."""
        return OptimizationResult(
            selected_indices=indices,
            energy=0.0,
            mode=mode,
            bitstring=self._indices_to_bitstring(indices, len(indices)),
            probabilities={},
            execution_time=time.time() - start_time,
            iterations=0,
            convergence_history=[]
        )
    
    def add_noise_model(self, noise_level: float = 0.01):
        """Add depolarizing noise model for realistic simulation."""
        noise_model = NoiseModel()
        error = depolarizing_error(noise_level, 1)
        noise_model.add_all_qubit_quantum_error(error, ['u1', 'u2', 'u3'])
        self.noise_model = noise_model
        self._initialize_backend()
        logger.info(f"Added noise model with level {noise_level}")
    
    def get_statistics(self) -> Dict:
        """Get optimization statistics."""
        return {
            'total_optimizations': len(self.optimization_history),
            'convergence_rate': len(self.convergence_history) / max(1, len(self.optimization_history)),
            'average_iterations': np.mean([r.iterations for r in self.optimization_history]) if self.optimization_history else 0,
            'modes_used': list(set([r.mode.value for r in self.optimization_history])) if self.optimization_history else []
        }


# Utility function for quick optimization
def quick_optimize(
    scores: np.ndarray,
    coords: np.ndarray,
    risks: np.ndarray,
    n_select: int = 5,
    layers: int = 2
) -> List[int]:
    """
    Quick optimization utility function.
    
    Example:
        >>> selected = quick_optimize(scores, coords, risks, n_select=5)
    """
    solver = QAOASolver(
        n_sites_to_select=n_select,
        qaoa_layers=layers,
        optimization_mode=OptimizationMode.HYBRID
    )
    
    result = solver.solve(scores, coords, risks)
    return result.selected_indices


# Example usage
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    n_sites = 20
    scores = np.random.rand(n_sites) * 100
    coords = np.random.rand(n_sites, 2) * 10
    risks = np.random.randint(1, 10, n_sites)
    
    # Create solver
    solver = QAOASolver(
        n_sites_to_select=5,
        qaoa_layers=2,
        optimization_mode=OptimizationMode.HYBRID
    )
    
    # Solve
    def progress_callback(msg):
        print(f"Progress: {msg}")
    
    result = solver.solve(scores, coords, risks, progress_callback=progress_callback)
    
    print(f"\n✅ Optimization Complete!")
    print(f"Mode: {result.mode.value}")
    print(f"Selected sites: {result.selected_indices}")
    print(f"Energy: {result.energy:.2f}")
    print(f"Time: {result.execution_time:.2f}s")
