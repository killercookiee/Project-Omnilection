import random

from Genetic_algorithm_processes.S4_migration.methods.clustering.model_clustering import ModelClusterer
from Genetic_algorithm_processes.S4_migration.methods.topology.model_based_topology import ModelBasedTopology
from Genetic_algorithm_processes.S4_migration.methods.topology.universal_topology import UniversalTopologyMigration

class PromptChainMigration:
    def __init__(self, migration_chance: float = 0.4,
        migration_method = ModelBasedTopology()
        ):
        """
        Initialize the Prompt Chain Migration system.
        
        Parameters:
        - migration_chance: Probability (0.0 to 1.0) that migration will occur
        - migration_method: Migration strategy to use (default: ModelClusteringTopology)
        """
        self.migration_chance = migration_chance
        self.migration_method = migration_method

    def migrate_prompt_chain(self, prompt_chain):
        """
        Migrate a single prompt chain based on migration chance.
        
        Parameters:
        - prompt_chain: List of tuples [(model_name, prompt_part1, ...), ...]
        
        Returns:
        - Migrated or original prompt chain
        """
        if random.random() < self.migration_chance:
            return self.migration_method.migrate(prompt_chain)
        return prompt_chain
    
    def migrate_population(self, prompt_chain_population):
        """
        Migrate an entire population of prompt chains.
        
        Parameters:
        - prompt_chain_population: List of prompt chains
        
        Returns:
        - List of migrated prompt chains
        """
        migrated_population = []
        for prompt_chain in prompt_chain_population:
            migrated_chain = self.migrate_prompt_chain(prompt_chain)
            migrated_population.append(migrated_chain)
        return migrated_population
    

if __name__ == "__main__":
    # Use models that are actually in your model registry
    prompt_chain_population = [
        [("qwen3:0.6b", "This is an ", "input prompt"), 
         ("gemma3:270m", "This ", "another ", "input ", "prompt")],
        [("qwen2.5:0.5b", "Different input prompt here"), 
         ("smollm2:360m", "", "Yet another prompt input")]
    ]
    
    print("=" * 80)
    print("Prompt Chain Migration with migration_chance=1.0")
    print("=" * 80)
    
    migration_instance = PromptChainMigration(
        migration_chance=1.0,
        migration_method=ModelBasedTopology(
            migration_ratio=0.5,
            default_temperature=0.3,
            clusterer=ModelClusterer(
                model_registry_path="LLM_models/model_registry.json",
                n_clusters=5,
                feature_config=[
                    ('parameters_count', 1.0),
                    ('context_length', 1.0),
                    ('benchmark.total_duration', 1.0)
                ]
            )
        )
    )
    
    print("\nOriginal Population:")
    for i, chain in enumerate(prompt_chain_population):
        print(f"Chain {i}: {chain}")
    
    print("\n" + "=" * 80)
    for run in range(3):
        print(f"\nRun {run + 1}:")
        print("-" * 80)
        migrated_population = migration_instance.migrate_population(prompt_chain_population)
        for i, chain in enumerate(migrated_population):
            print(f"Chain {i}: {chain}")