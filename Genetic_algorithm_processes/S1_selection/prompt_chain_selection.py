"""
Genetic_algorithm_processes/S1_selection/prompt_chain_selection.py
"""

from Genetic_algorithm_processes.S1_selection.methods.selection.stochastic_universal_sampling import StochasticUniversalSampling

class PromptChainSelection:
    def __init__(self,
        selection_algorithm=StochasticUniversalSampling(),
        verbose: bool = False
    ):
        self.selection_algorithm = selection_algorithm
        self.verbose = verbose

        if self.verbose:
            print(f"[PromptChainSelection] Initialized — algorithm: {type(self.selection_algorithm).__name__}")

    def select_prompt_chains(self, population_records: list[tuple]) -> list[tuple]:
        """
        Accepts evaluated population records: [(chain_id, chain, fitness, metadata), ...]
        Returns a selected subset of the full records.
        """
        if self.verbose:
            print(f"\n[PromptChainSelection] Running {type(self.selection_algorithm).__name__} on {len(population_records)} individuals...")

        # Map to the format (record, fitness) for the selection algorithm.
        # We add a microscopic epsilon (1e-9) to prevent division-by-zero crashes 
        # in the rare event that the ENTIRE population completely fails (fitness = 0.0).
        population_fitness = [(rec, max(rec[2], 1e-9)) for rec in population_records]
        
        # Selection algorithm returns the selected records
        selected_records = self.selection_algorithm.select(population_fitness)

        if self.verbose:
            print(f"  ✅  Selected {len(selected_records)} / {len(population_records)} chains")

        return selected_records