"""
Genetic_algorithm_processes/S4_migration/prompt_chain_migration.py
"""

import random
from Genetic_algorithm_processes.S4_migration.methods.topology.model_based_topology import ModelBasedTopology

class PromptChainMigration:
    def __init__(self, migration_chance: float = 0.4, migration_method = None):
        self.migration_chance = migration_chance
        self.migration_method = migration_method or ModelBasedTopology()

    def migrate_population(self, offspring_records: list[dict]) -> list[dict]:
        migrated_records = []
        for record in offspring_records:
            if random.random() < self.migration_chance:
                migrated_chain = self.migration_method.migrate(record["chain"])
                
                new_record = record.copy()
                new_record["chain"] = migrated_chain
                new_record["metadata"] = record["metadata"].copy()
                new_record["metadata"]["migrated"] = True
                
                migrated_records.append(new_record)
            else:
                migrated_records.append(record)
        return migrated_records