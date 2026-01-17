"""
qaoa_solver.py
Solve QUBO via QAOA (Quantum Approximate Optimization Algorithm)
"""

import numpy as np
import pandas as pd
import json, os
from qiskit import Aer
from qiskit.algorithms import QAOA
from qiskit.algorithms.optimizers import COBYLA
from qiskit.opflow import PauliSumOp
from qiskit.utils import QuantumInstance
import matplotlib.pyplot as plt

class QAOASolver:
    """
    Solve site selection problem with QAOA
    """
    def __init__(self, backend='qasm_simulator', shots=1024):
        self.backend = Aer.get_backend(backend)
        self.q_instance = QuantumInstance(self.backend, shots=shots, seed_simulator=42, seed_transpiler=42)
        self.optimizer = COBYLA(maxiter=50)
        self.result = None
        self.selected_indices = []

    # ---------------- QUBO → Ising ----------------
    def qubo_to_ising(self, qubo_dict, n_qubits):
        h = np.zeros(n_qubits)
        J = np.zeros((n_qubits, n_qubits))
        for (i,j), val in qubo_dict.items():
            if i==j: h[i] += val/2
            else:
                J[i,j] += val/4
                h[i] += val/4
                h[j] += val/4

        paulis = []
        for i in range(n_qubits):
            if abs(h[i])>1e-10:
                s = 'I'*n_qubits
                s = s[:i]+'Z'+s[i+1:]
                paulis.append((s,h[i]))
        for i in range(n_qubits):
            for j in range(i+1,n_qubits):
                if abs(J[i,j])>1e-10:
                    s = 'I'*n_qubits
                    s = s[:i]+'Z'+s[i+1:]
                    s = s[:j]+'Z'+s[j+1:]
                    paulis.append((s,J[i,j]))
        return PauliSumOp.from_list(paulis)

    # ---------------- Solve ----------------
    def solve(self, qubo_dict, n_qubits, p=1):
        print(f"Running QAOA: qubits={n_qubits}, p={p}")
        hamiltonian = self.qubo_to_ising(qubo_dict, n_qubits)
        qaoa = QAOA(optimizer=self.optimizer, reps=p, quantum_instance=self.q_instance)
        self.result = qaoa.compute_minimum_eigenvalue(hamiltonian)

        eigenstate = getattr(self.result,'eigenstate',None)
        if eigenstate is None: return {}

        probs = np.abs(eigenstate)**2
        best_idx = np.argmax(probs)
        best_state = format(best_idx,f'0{n_qubits}b')
        self.selected_indices = [i for i,bit in enumerate(best_state) if bit=='1']

        return {
            'energy': float(self.result.eigenvalue.real),
            'best_state': best_state,
            'probability': float(probs[best_idx]),
            'selected_indices': self.selected_indices
        }

    # ---------------- Map Selection ----------------
    def map_selection(self, df: pd.DataFrame):
        if not self.selected_indices: return pd.DataFrame(), {}
        sel = df.iloc[self.selected_indices].copy()
        sel['selection_order'] = range(1,len(sel)+1)
        sel['quantum_selected'] = True
        summary = {
            'n_selected': len(sel),
            'total_score': float(sel['final_score'].sum()),
            'avg_score': float(sel['final_score'].mean()),
            'total_cost': int(sel['estimated_cost_usd'].sum()),
            'total_population': int(sel['population_served'].sum()),
            'cost_per_person': float(sel['estimated_cost_usd'].sum()/sel['population_served'].sum())
                              if sel['population_served'].sum()>0 else 0
        }
        return sel, summary

    # ---------------- Save & Plot ----------------
    def save_results(self, results, selected_sites, summary, out='results/quantum_selections'):
        os.makedirs(out,exist_ok=True)
        with open(f"{out}/qaoa_results.json",'w') as f: json.dump(results,f,indent=2)
        if not selected_sites.empty: selected_sites.to_csv(f"{out}/selected_sites.csv",index=False)
        with open(f"{out}/selection_summary.json",'w') as f: json.dump(summary,f,indent=2)
        print(f"✓ Results saved in {out}")

    def plot_results(self, results, out='results/quantum_selections'):
        os.makedirs(out,exist_ok=True)
        if 'optimal_parameters' in results:
            params = results['optimal_parameters']
            plt.figure(figsize=(8,5))
            plt.bar(range(len(params)), list(params.values()))
            plt.title("Optimal QAOA parameters")
            plt.xlabel("Parameter index"); plt.ylabel("Value"); plt.grid(True,alpha=0.3)
            plt.savefig(f"{out}/qaoa_params.png",dpi=300,bbox_inches='tight')
            plt.close()
            print(f"✓ QAOA params plot saved in {out}")

# ---------------- MAIN ----------------
def main():
    from qubo_formulation import QUBOFormer
    former = QUBOFormer()
    Q = np.load('results/quantum_selections/qubo_matrix.npy')
    n = Q.shape[0]
    qubo_dict = former.qubo_to_dict(Q)

    solver = QAOASolver()
    results = solver.solve(qubo_dict,n_qubits=n,p=1)

    df = pd.read_csv('data/processed/scored_sites.csv')
    selected_sites, summary = solver.map_selection(df)
    solver.save_results(results, selected_sites, summary)
    solver.plot_results(results)
    print(f"Selected sites:\n{selected_sites[['name_ar','final_score']]}")
    return solver, results, selected_sites

if __name__=="__main__":
    solver, results, selected_sites = main()
