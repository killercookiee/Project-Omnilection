# Class input model_registry.json file get the available models and their details for migration
# Since the topology is universal, migration between any two models is the same
# input prompt chain which is a list of tuples (model_name, prompt_part1, prompt_part2, ...)
# out put new prompt chain with a migration ratio of new model that is randomly chosen from the available models in the registry (excluding the current model)

# Input parameters:
# model_registry_path: path to the model registry json file
# migration_ratio: percentage of prompt in prompt chain to migrate (e.g. 0.1 means 10% of the prompts in the chain will be migrated)

import json
import random
from typing import Any, Dict


class UniversalTopologyMigration:
    def __init__(self,
        migration_ratio: float = 0.5,
        model_registry_path: str = 'LLM_models/model_registry.json'
    ):
        self.migration_ratio = migration_ratio
        self.model_registry_path = model_registry_path
        self.model_registry: Dict[str, Dict[str, Any]] = {}
        self.load_model_registry()
    
    def load_model_registry(self) -> None:
        """Load the model registry from the JSON file if it exists."""
        try:
            with open(self.model_registry_path, 'r') as f:
                self.model_registry = json.load(f)
            print(f"Loaded {len(self.model_registry)} models from registry")
        except FileNotFoundError:
            print(f"Model registry not found at {self.model_registry_path}")
            print("Run update_model_registry() to create it")
        except json.JSONDecodeError as e:
            print(f"Error parsing model registry: {e}")

    def migrate(self, prompt_chain):
        """
        Migrate the model in the prompt at prompt_index to a different model from the registry.
        
        Parameters:
        - prompt_chain: List of sets containing (model_name, prompt_input_segment_1, prompt_input_segment_2, ... for each step in the chain
        
        Returns:
        - new_prompt_chain: The prompt chain after migration
        """
        if not self.model_registry:
            print("Model registry is empty. Cannot perform migration.")
            return prompt_chain
        
        prompt_index = random.randint(0, len(prompt_chain) - 1)
        
        current_model = prompt_chain[prompt_index][0]
        available_models = [model for model in self.model_registry.keys() if model != current_model]
        print(f"\nCurrent model: {current_model}. Available models for migration: {available_models}")

        if not available_models:
            print("No alternative models available for migration.")
            return prompt_chain
        
        new_model = random.choice(available_models)
        new_prompt_chain = list(prompt_chain)  # Create a copy of the original chain
        new_prompt_chain[prompt_index] = (new_model,) + tuple(new_prompt_chain[prompt_index][1:])  # Replace model name
        
        return new_prompt_chain
    

if __name__ == "__main__":
    migration_instance = UniversalTopologyMigration(migration_ratio=0.5)
    sample_prompt_chain = [("gpt-3.5-turbo", "This is an ", "input prompt"), ("gpt-4", "This ", "another ", "input ", "prompt")]
    migrated_prompt_chain = migration_instance.migrate(sample_prompt_chain)
    print(f"Original Prompt Chain: {sample_prompt_chain}")
    print(f"Migrated Prompt Chain: {migrated_prompt_chain}")