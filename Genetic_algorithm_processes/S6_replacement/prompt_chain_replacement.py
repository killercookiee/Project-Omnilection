"""
Genetic_algorithm_processes/S6_replacement/prompt_chain_replacement.py
"""

from Genetic_algorithm_processes.Data.general_datamanager import GeneralDataManager
from Genetic_algorithm_processes.S6_replacement.methods.replacement.lineage_replacement import LineageReplacement

class PromptChainReplacement:
    def __init__(self,
        replacement_algorithm = LineageReplacement(),
        gdm: GeneralDataManager = None
    ):
        self.replacement_algorithm = replacement_algorithm
        self.gdm = gdm

    def replace_population(self, current_population_records: list[tuple], offspring_population_records: list[tuple]) -> list[tuple]:
        """
        Takes evaluated tuples from both generations: [(chain_id, chain, fitness, metadata), ...]
        """
        if not self.gdm:
            raise ValueError("[PromptChainReplacement] GeneralDataManager (gdm) is required to fetch lineage stats.")

        # 1. Sync the newly evaluated offspring so they exist in the heritage DB
        self.gdm.sync_population(offspring_population_records)
        
        # 2. Refresh lineage scores globally now that new genetics are added
        self.gdm.refresh_lineage_scores()

        combined_raw = current_population_records + offspring_population_records
        heritage_db = self.gdm.heritage_data_manager.local_heritage_database.get("prompt_chains", {})

        # 3. Build the enriched population tuples for the replacement algorithm
        # Format: (chain_tuple, fitness, lineage_score, family_size)
        enriched_population = []
        for entry in combined_raw:
            chain_id = entry[0]
            base_chain = entry[1]
            base_len = len(base_chain)
            fitness = entry[2]
            
            lf_score = self.gdm.lineage_score(chain_id)
            
            # Dynamic Family Size: How many chains in history START with this exact sequence?
            family_size = sum(
                1 for cid in heritage_db.keys()
                if (c := self.gdm.id_to_promptchain_manager._mapping.get(cid)) 
                and len(c) >= base_len 
                and c[:base_len] == base_chain
            )

            enriched_population.append((entry, fitness, lf_score, family_size))

        # 4. Run the replacement algorithm
        # (Note: Update lineage_replacement.py to expect ind[3] as family_size instead of ind[4])
        survivors_enriched = self.replacement_algorithm.replace(enriched_population)

        # 5. Extract the base (chain_id, chain, fitness, metadata) records of the survivors
        final_population = [ind[0] for ind in survivors_enriched]

        return final_population