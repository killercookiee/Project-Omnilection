"""
Genetic_algorithm_processes/S2_recombination/methods/crossover/lineage_crossover.py

1-point crossover engine designed to process the crossover hints 
provided by the LineagePairing algorithm.
"""

import random

class LineageCrossover:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose

        if self.verbose:
            print(f"[LineageCrossover] Initialized — 1-point Prefix/Suffix grafting")

    def crossover(self, p1_chain: list, p2_chain: list, crossover_hint: int | None) -> list:
        """
        Executes a 1-point crossover.
        - P1 contributes the prefix.
        - P2 contributes the suffix.
        """
        if self.verbose:
            print(f"\n[LineageCrossover] Crossing P1 (len {len(p1_chain)}) with P2 (len {len(p2_chain)})")

        if crossover_hint is None:
            # Standard Exploitation: Random 1-point crossover within the valid bounds
            min_len = min(len(p1_chain), len(p2_chain))
            crossover_point = random.randint(1, max(1, min_len - 1))
            if self.verbose:
                print(f"  Hint: None -> Random crossover point determined: {crossover_point}")
        else:
            # Infidelity/Mentorship: Strict graft based on pairing hint
            crossover_point = crossover_hint
            if self.verbose:
                print(f"  Hint: {crossover_hint} -> Strict prefix graft applied.")

        # Graft P1's prefix with P2's continuation
        prefix = p1_chain[:crossover_point]
        suffix = p2_chain[crossover_point:]
        
        offspring = prefix + suffix

        if self.verbose:
            print(f"  Offspring generated: length {len(offspring)}")
            
        return offspring