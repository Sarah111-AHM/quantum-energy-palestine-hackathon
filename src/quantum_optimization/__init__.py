"""
Quantum Optimization Module for Gaza Strip Energy Infrastructure Planning.
Provides QAOA-based solvers, QUBO formulation, and classical baselines.
"""

from .quantum_optimization.qaoa_solver import QAOASolver, OptimizationResult, OptimizationMode
from .quantum_optimization.qubo_formulation import QUBOFormulation, QUBOParameters, create_qubo_from_data
from .quantum_optimization.classical_baselines import ClassicalBaselines, ClassicalMethod, benchmark_classical_methods

__all__ = [
    'QAOASolver',
    'OptimizationResult',
    'OptimizationMode',
    'QUBOFormulation',
    'QUBOParameters',
    'create_qubo_from_data',
    'ClassicalBaselines',
    'ClassicalMethod',
    'benchmark_classical_methods'
]

__version__ = '1.0.0'
