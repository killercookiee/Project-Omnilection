"""
Genetic_algorithm_processes/S3_mutation/prompt_chain_mutation.py
"""

import random
import hashlib
from typing import Any
from unittest import runner
from Genetic_algorithm_processes.Data.general_datamanager import GeneralDataManager
from Genetic_algorithm_processes.S3_mutation.methods.delete_mutation import DeleteMutation
from Genetic_algorithm_processes.S3_mutation.methods.semantic_llm_mutation import SemanticLLMMutation
from Genetic_algorithm_processes.S3_mutation.methods.synonym_mutation import SynonymMutation

def get_chain_id(chain: list) -> str:
    return f"chain_{hashlib.md5(str(chain).encode('utf-8')).hexdigest()[:12]}"
class PromptChainMutation:
    def __init__(self,
        base_mutation_chance: float = 0.10,
        min_mutation_chance: float = 0.02,
        max_mutation_chance: float = 0.30,
        low_lineage_threshold: float = 0.30, # A lineage score < 0.30 is considered an "Explorer"
        target_explorer_ratio: tuple[float, float] = (0.20, 0.40), # We want 20% to 40% of the population to be explorers
        mutation_methods: list[Any] = None,
        gdm: GeneralDataManager = None,
        verbose: bool = False
    ):
        self.current_mutation_chance = base_mutation_chance
        self.min_mutation_chance = min_mutation_chance
        self.max_mutation_chance = max_mutation_chance
        
        self.low_lineage_threshold = low_lineage_threshold
        self.target_explorer_min = target_explorer_ratio[0]
        self.target_explorer_max = target_explorer_ratio[1]

        self.mutation_methods = mutation_methods or [
            SemanticLLMMutation(runner=runner, mutator_model="qwen2.5-coder:0.5b", verbose=False).mutate,
            DeleteMutation(min_segment_fraction=0.1, max_segment_fraction=0.3).mutate,
            SynonymMutation(
            mutation_rate=0.5,
            pos_tags_to_mutate=['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ', 'JJR', 'JJS']
            ).mutate
        ]
        self.gdm = gdm
        self.verbose = verbose

    def adjust_mutation_rate(self, current_population_records: list[tuple]) -> None:
        """
        The Thermostat: Analyzes the current population's lineage scores 
        and adjusts the global mutation rate to maintain diversity without causing chaos.
        """
        if not self.gdm or not current_population_records:
            return

        explorer_count = 0
        total_count = len(current_population_records)

        for rec in current_population_records:
            chain_id = rec[0]
            lf_score = self.gdm.lineage_score(chain_id)
            if lf_score < self.low_lineage_threshold:
                explorer_count += 1

        explorer_ratio = explorer_count / total_count

        # Adjust the thermostat
        old_rate = self.current_mutation_chance
        if explorer_ratio < self.target_explorer_min:
            # Too stagnant! Crank up mutation by 50%
            self.current_mutation_chance = min(self.max_mutation_chance, self.current_mutation_chance * 1.5)
            status = "🔥 HEATING UP"
        elif explorer_ratio > self.target_explorer_max:
            # Too chaotic! Cool down mutation by 25%
            self.current_mutation_chance = max(self.min_mutation_chance, self.current_mutation_chance * 0.75)
            status = "❄️ COOLING DOWN"
        else:
            status = "✅ STABLE"

        if self.verbose:
            print(f"\n[Mutation Thermostat] Explorers: {explorer_ratio:.0%} (Target: {self.target_explorer_min:.0%}-{self.target_explorer_max:.0%})")
            if old_rate != self.current_mutation_chance:
                print(f"  {status} -> Rate shifted from {old_rate:.1%} to {self.current_mutation_chance:.1%}")
            else:
                print(f"  {status} -> Rate remains at {self.current_mutation_chance:.1%}")

    def mutate_population(self, offspring_records: list[dict]) -> list[dict]:
        mutated_records = []
        mutations_applied = 0
        
        for record in offspring_records:
            if random.random() < self.current_mutation_chance and self.mutation_methods:
                mutation_func = random.choice(self.mutation_methods)
                
                # Execute the mutation
                mutated_chain = mutation_func(record["chain"])
                
                # Check if it actually changed (LLM might have failed)
                if mutated_chain != record["chain"]:
                    new_id = get_chain_id(mutated_chain)
                    old_id = record["chain_id"]
                    
                    metadata = record["metadata"].copy()
                    metadata["mutated"] = True
                    # If it's an object method, try to get class name, else function name
                    method_name = getattr(mutation_func, '__self__', mutation_func).__class__.__name__
                    metadata["mutation_method"] = method_name
                    
                    # Register the major mutation in heritage history immediately
                    if self.gdm:
                        self.gdm.register_intermediary_chain(new_id, mutated_chain, [old_id], metadata)
                    
                    mutated_records.append({
                        "chain_id": new_id,
                        "chain": mutated_chain,
                        "parents": [old_id],
                        "metadata": metadata
                    })
                    mutations_applied += 1
                else:
                    mutated_records.append(record)
            else:
                mutated_records.append(record)
                
        if self.verbose:
            print(f"  [Mutation] {mutations_applied} individuals significantly mutated.")
            
        return mutated_records