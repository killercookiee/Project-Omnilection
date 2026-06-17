"""
Genetic_algorithm_processes/S2_recombination/prompt_chain_recombination.py
"""
import hashlib
from Genetic_algorithm_processes.S2_recombination.methods.pairing.lineage_pairing import LineagePairing
from Genetic_algorithm_processes.S2_recombination.methods.crossover.lineage_crossover import LineageCrossover
from Genetic_algorithm_processes.S2_recombination.methods.crossover.model_based_crossover import ModelBasedCrossover
from Genetic_algorithm_processes.Data.general_datamanager import GeneralDataManager

def get_chain_id(chain: list) -> str:
    return f"chain_{hashlib.md5(str(chain).encode('utf-8')).hexdigest()[:12]}"

class PromptChainRecombination:
    def __init__(self,
        prompt_chain_pairing_instance = LineagePairing(),
        prompt_chain_crossover_instance = LineageCrossover(),
        prompt_crossover_instance = ModelBasedCrossover(),
        gdm: GeneralDataManager = None
    ):
        self.prompt_chain_pairing_instance = prompt_chain_pairing_instance
        self.prompt_chain_crossover_instance = prompt_chain_crossover_instance
        self.prompt_crossover_instance = prompt_crossover_instance
        self.gdm = gdm

    def recombine_prompt_chains(self, selected_records: list[tuple]) -> list[dict]:
        if not self.gdm:
            raise ValueError("[PromptChainRecombination] GeneralDataManager (gdm) is required.")

        # Map chain string back to ID so we don't lose parent IDs
        chain_to_id = {str(rec[1]): rec[0] for rec in selected_records}
        
        selected_chains = [rec[1] for rec in selected_records]
        pop_with_lineage = self.gdm.get_population_with_lineage(selected_chains)

        self.prompt_chain_pairing_instance.number_of_pairs = len(selected_records)
        paired_prompt_chains = self.prompt_chain_pairing_instance.pair(pop_with_lineage)

        offspring_records = []
        for (p1_chain, p2_chain, hint) in paired_prompt_chains:
            
            p1_id = chain_to_id.get(str(p1_chain))
            p2_id = chain_to_id.get(str(p2_chain))
            parents = [p1_id, p2_id] if p1_id and p2_id else []

            offspring_chain = self.prompt_chain_crossover_instance.crossover(p1_chain, p2_chain, hint)
            offspring_recombined = self.prompt_crossover_instance.crossover([p1_chain, p2_chain], offspring_chain)
            
            offspring_id = get_chain_id(offspring_recombined)
            
            # ── Determine and Tag Recombination Mode ──
            mode = "exploitation" if hint is None else "exploration"
            
            metadata = {
                "prefix_len": hint if hint is not None else 1,
                "recombined": True,
                "recombination_mode": mode
            }

            self.gdm.register_intermediary_chain(offspring_id, offspring_recombined, parents, metadata)
            
            offspring_records.append({
                "chain_id": offspring_id,
                "chain": offspring_recombined,
                "parents": parents,
                "metadata": metadata
            })
            
        return offspring_records
    