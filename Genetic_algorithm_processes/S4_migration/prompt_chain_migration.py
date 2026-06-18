"""
Genetic_algorithm_processes/S4_migration/prompt_chain_migration.py
"""

import random
from Genetic_algorithm_processes.S4_migration.methods.topology.model_based_topology import ModelBasedTopology

class PromptChainMigration:
    def __init__(self, 
                 base_migration_chance: float = 0.2, 
                 migration_method = None,
                 verbose: bool = False):
        self.base_migration_chance = base_migration_chance
        self.current_migration_chance = base_migration_chance
        
        self.migration_method = migration_method or ModelBasedTopology()
        self.base_temperature = self.migration_method.default_temperature
        self.current_temperature = self.base_temperature
        
        self.verbose = verbose
        
        # Stagnation tracking
        self.best_fitness_history = []
        self.stagnation_threshold = 3  # Generations without improvement

    def adjust_migration_rate(self, current_population: list[tuple]) -> None:
        """Dynamically adjusts migration parameters based on population stagnation."""
        if not current_population:
            return
            
        # Extract the highest fitness in the current population
        current_best = max(ind[2] for ind in current_population)
        self.best_fitness_history.append(current_best)
        
        # Keep history buffer small
        if len(self.best_fitness_history) > 10:
            self.best_fitness_history.pop(0)

        # Check for stagnation
        if len(self.best_fitness_history) >= self.stagnation_threshold:
            recent_history = self.best_fitness_history[-self.stagnation_threshold:]
            improvement = max(recent_history) - min(recent_history)
            
            if improvement < 0.01:
                # STAGNATION: Trigger mass migration & higher temperature
                self.current_migration_chance = min(0.80, self.current_migration_chance + 0.15)
                self.current_temperature = min(0.90, self.current_temperature + 0.20)
                if self.verbose:
                    print(f"  [Migration] ⚠️ Stagnation detected! Increasing chance to {self.current_migration_chance:.0%} and temp to {self.current_temperature:.2f}")
            else:
                # IMPROVING: Gradually cool down back to base levels
                self.current_migration_chance = max(self.base_migration_chance, self.current_migration_chance - 0.05)
                self.current_temperature = max(self.base_temperature, self.current_temperature - 0.10)
                if self.verbose and self.current_migration_chance > self.base_migration_chance:
                    print(f"  [Migration] ✓ Fitness improving. Cooling chance to {self.current_migration_chance:.0%} and temp to {self.current_temperature:.2f}")

    def migrate_population(self, offspring_records: list[dict]) -> list[dict]:
        migrated_records = []
        migrations_performed = 0
        
        for record in offspring_records:
            if random.random() < self.current_migration_chance:
                # Pass the dynamic temperature to the topology system
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