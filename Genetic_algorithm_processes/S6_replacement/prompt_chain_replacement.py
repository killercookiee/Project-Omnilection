"""
Genetic_algorithm_processes/S6_replacement/prompt_chain_replacement.py
"""
import math
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

        # ── 1. Calculate Task Difficulty Multipliers ──
        task_fitness_sums = {}
        task_counts = {}
        global_sum = 0.0
        current_gen = 0

        for entry in combined_raw:
            fit = entry[2]
            meta = entry[3]
            task_id = meta.get("task_id", "unknown")

            task_fitness_sums[task_id] = task_fitness_sums.get(task_id, 0.0) + fit
            task_counts[task_id] = task_counts.get(task_id, 0) + 1
            global_sum += fit

            # Track current generation for age calculation
            gen_val = meta.get("generation", 0)
            if isinstance(gen_val, list):
                gen_val = max(gen_val) if gen_val else 0
            if gen_val > current_gen:
                current_gen = gen_val

        global_mean = global_sum / max(1, len(combined_raw))
        task_multipliers = {}
        for tid, count in task_counts.items():
            t_mean = task_fitness_sums[tid] / count
            # Difficulty Multiplier: >1.0 for hard tasks, <1.0 for easy tasks (with 0.1 smoother)
            task_multipliers[tid] = (global_mean + 0.1) / (t_mean + 0.1)

        # ── 2. Apply Adjustments & Map to Lineage Replacement ──
        enriched_population = []
        for entry in combined_raw:
            chain_id, base_chain, raw_fitness, meta = entry

            # Apply Difficulty Multiplier
            task_id = meta.get("task_id", "unknown")
            adjusted_fitness = raw_fitness * task_multipliers.get(task_id, 1.0)

            # Apply Exponential Age Penalty (Effective Cap around Gen 6)
            birth_gen = meta.get("generation", current_gen)
            if isinstance(birth_gen, list):
                birth_gen = min(birth_gen) if birth_gen else current_gen
            
            age = current_gen - birth_gen
            if age > 3: # 3 generation grace period
                # Decay factor: e^-0.4 ~ 0.67 at age 4, ~ 0.30 at age 6
                adjusted_fitness *= math.exp(-0.4 * (age - 3))

            # Repack the tuple with the newly adjusted posthumous fitness
            adjusted_entry = (chain_id, base_chain, adjusted_fitness, meta)

            lf_score = self.gdm.lineage_score(chain_id)
            base_len = len(base_chain)
            family_size = sum(
                1 for cid in heritage_db.keys()
                if (c := self.gdm.id_to_promptchain_manager._mapping.get(cid)) 
                and len(c) >= base_len 
                and c[:base_len] == base_chain
            )

            enriched_population.append((adjusted_entry, adjusted_fitness, lf_score, family_size))

        survivors_enriched = self.replacement_algorithm.replace(enriched_population)
        return survivors_enriched