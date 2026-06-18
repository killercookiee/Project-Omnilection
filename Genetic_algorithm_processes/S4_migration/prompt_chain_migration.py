"""
Genetic_algorithm_processes/S4_migration/prompt_chain_migration.py
"""

import random
from Genetic_algorithm_processes.S4_migration.methods.topology.model_based_topology import ModelBasedTopology

class PromptChainMigration:
    def __init__(self, 
                 base_migration_chance: float = 0.2, 
                 migration_method = None,
                 gdm = None, 
                 verbose: bool = False):
        self.base_migration_chance = base_migration_chance
        self.current_migration_chance = base_migration_chance
        
        self.migration_method = migration_method or ModelBasedTopology()
        self.base_temperature = self.migration_method.default_temperature
        self.current_temperature = self.base_temperature
        
        self.gdm = gdm 
        self.verbose = verbose
        
        self.best_fitness_history = []
        self.stagnation_threshold = 3 

    def adjust_migration_rate(self, current_population: list[tuple]) -> None:
        if not current_population:
            return
            
        current_best = max(ind[2] for ind in current_population)
        self.best_fitness_history.append(current_best)
        
        if len(self.best_fitness_history) > 10:
            self.best_fitness_history.pop(0)

        # ── Calculate Global Crowdedness ──
        global_crowdedness = 0.0
        if self.gdm:
            lf_sum = sum(self.gdm.lineage_score(ind[0]) for ind in current_population)
            global_crowdedness = lf_sum / len(current_population)

        # ── Stagnation Check ──
        is_stagnant = False
        if len(self.best_fitness_history) >= self.stagnation_threshold:
            recent_history = self.best_fitness_history[-self.stagnation_threshold:]
            if (max(recent_history) - min(recent_history)) < 0.01:
                is_stagnant = True

        # Trigger migration bump if Stagnant OR Highly Crowded
        if is_stagnant or global_crowdedness > 0.6:
            crowd_boost = max(0.0, global_crowdedness - 0.5) * 0.5 
            
            self.current_migration_chance = min(0.90, self.base_migration_chance + 0.30 + crowd_boost)
            self.current_temperature = min(0.95, self.base_temperature + 0.30 + crowd_boost)
            
            if self.verbose:
                reason = "Stagnation + Crowdedness" if (is_stagnant and global_crowdedness > 0.6) else ("Crowdedness" if global_crowdedness > 0.6 else "Stagnation")
                print(f"  [Migration] ⚠️ {reason} detected (Crowd={global_crowdedness:.2f})! Increasing chance to {self.current_migration_chance:.0%} and temp to {self.current_temperature:.2f}")
        else:
            self.current_migration_chance = max(self.base_migration_chance, self.current_migration_chance - 0.05)
            self.current_temperature = max(self.base_temperature, self.current_temperature - 0.10)
            if self.verbose and self.current_migration_chance > self.base_migration_chance:
                print(f"  [Migration] ✓ Population healthy. Cooling chance to {self.current_migration_chance:.0%} and temp to {self.current_temperature:.2f}")

    def migrate_population(self, offspring_records: list[dict]) -> list[dict]:
        migrated_records = []
        migrations_performed = 0
        
        for record in offspring_records:
            if random.random() < self.current_migration_chance:
                migrated_chain = self.migration_method.migrate(
                    record["chain"], 
                    temperature=self.current_temperature
                )
                
                new_record = record.copy()
                new_record["chain"] = migrated_chain
                new_record["metadata"] = record["metadata"].copy()
                new_record["metadata"]["migrated"] = True
                new_record["metadata"]["migration_temp"] = self.current_temperature
                
                migrated_records.append(new_record)
                migrations_performed += 1
            else:
                migrated_records.append(record)
                
        if self.verbose and migrations_performed > 0:
            print(f"  [Migration] Migrated models for {migrations_performed} offspring chains.")
            
        return migrated_records