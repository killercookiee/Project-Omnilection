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
        if not self.gdm:
            raise ValueError("[PromptChainReplacement] GeneralDataManager (gdm) is required to fetch lineage stats.")

        self.gdm.sync_population(offspring_population_records)
        self.gdm.refresh_lineage_scores()

        combined_raw = current_population_records + offspring_population_records
        heritage_db = self.gdm.heritage_data_manager.local_heritage_database.get("prompt_chains", {})

        enriched_population = []
        for entry in combined_raw:
            chain_id = entry[0]
            base_chain = entry[1]
            base_len = len(base_chain)
            fitness = entry[2]
            
            lf_score = self.gdm.lineage_score(chain_id)
            
            family_size = sum(
                1 for cid in heritage_db.keys()
                if (c := self.gdm.id_to_promptchain_manager._mapping.get(cid)) 
                and len(c) >= base_len 
                and c[:base_len] == base_chain
            )

            enriched_population.append((entry, fitness, lf_score, family_size))

        survivors_enriched = self.replacement_algorithm.replace(enriched_population)

        # 5. Extract the base records of the survivors
        # LineageReplacement ALREADY strips the enrichment down to the base tuple, 
        # so survivors_enriched is exactly what we need to return!
        return survivors_enriched