"""
Model Based Topology Migration System

This module implements a migration system for prompt chains based on model clustering.
Models are clustered by their characteristics (e.g., parameters, context length, performance),
and migration targets are selected using distance-weighted probability distributions.

Key Features:
- Accepts a pre-configured ModelClusterer instance for flexible clustering strategies
- Migrates a specified ratio of prompts in a chain to similar models
- Uses temperature-controlled probability distribution for migration target selection:
  * Lower temperature (e.g., 0.1): Conservative, prefers very similar models
  * Higher temperature (e.g., 0.9): Adventurous, considers more diverse models
- Distance-based selection uses truncated exponential decay for probability calculation

Input:
- prompt_chain: List of tuples [(model_name, prompt_part1, prompt_part2, ...), ...]
- migration_ratio: Percentage of prompts in chain to migrate (0.0 to 1.0)
- temperature: Controls how conservative/adventurous the migration is (0 < temp <= 1)
- clusterer: Pre-configured ModelClusterer instance with desired clustering strategy

Output:
- new_prompt_chain: Modified chain where selected prompts have migrated to new models
  based on distance-weighted probability distribution

Example:
    clusterer = ModelClusterer(
        model_registry_path="LLM_models/model_registry.json",
        n_clusters=5,
        feature_config=[('parameters_count', 1.0), ('context_length', 1.0)]
    )
    
    migration_system = ModelBasedTopology(
        migration_ratio=0.5,
        default_temperature=0.3,
        clusterer=clusterer
    )
    
    new_chain = migration_system.migrate(original_chain, temperature=0.5)
"""

import random
import numpy as np
from typing import Any, List, Tuple, Dict
from Genetic_algorithm_processes.S4_migration.methods.clustering.model_clustering import ModelClusterer


class ModelBasedTopology:
    def __init__(self,
        migration_ratio: float = 0.5,
        default_temperature: float = 0.5,
        clusterer = ModelClusterer()
    ):
        """
        Initialize the Model Clustering Topology system.
        
        Parameters:
        - migration_ratio: Percentage of prompts in chain to migrate (0.0 to 1.0)
        - model_registry_path: Path to the model registry JSON file
        - n_clusters: Number of clusters for k-means
        - feature_config: List of tuples [(feature_name, weight), ...]
        - default_temperature: Default temperature for migration probability (0 < temp <= 1)
        """
        self.migration_ratio = migration_ratio
        self.default_temperature = default_temperature
        self.clusterer = clusterer
        
        print(f"\nMigration system initialized with ratio={migration_ratio}, temp={default_temperature}")
    
    def calculate_migration_probability(
        self,
        distances: Dict[str, float],
        temperature: float
    ) -> Dict[str, float]:
        """
        Calculate migration probabilities using truncated exponential decay.
        
        Formula: P(x) = [λ * e^(-λx)] / (1 - e^(-λ))
        where λ = -ln(temperature), and x is normalized distance
        
        Parameters:
        - distances: Dictionary mapping model names to normalized distances
        - temperature: Temperature parameter (0 < temp <= 1)
                      Lower temp = more peaked toward closest models
                      Higher temp = more uniform distribution
        
        Returns:
        - Dictionary mapping model names to migration probabilities
        """
        if not distances:
            return {}
        
        # Ensure temperature is valid
        temperature = max(0.01, min(1.0, temperature))
        
        # Calculate lambda
        lambda_param = -np.log(temperature)
        
        # Normalize distances to [0, 1] range
        dist_values = np.array(list(distances.values()))
        if dist_values.max() > 0:
            normalized_dists = dist_values / dist_values.max()
        else:
            normalized_dists = dist_values
        
        # Calculate probabilities using truncated exponential
        probabilities = {}
        normalizer = 1 - np.exp(-lambda_param)  # Normalization constant
        
        for i, (model_name, _) in enumerate(distances.items()):
            x = normalized_dists[i]
            prob = (lambda_param * np.exp(-lambda_param * x)) / normalizer
            probabilities[model_name] = float(prob)
        
        # Normalize to ensure they sum to 1
        total_prob = sum(probabilities.values())
        if total_prob > 0:
            probabilities = {k: v / total_prob for k, v in probabilities.items()}
        
        return probabilities
    
    def select_migration_target(
        self,
        current_model: str,
        temperature: float
    ) -> str:
        """
        Select a migration target model based on distance-weighted probabilities.
        
        Parameters:
        - current_model: Name of the current model
        - temperature: Temperature for probability distribution
        
        Returns:
        - Name of the selected target model
        """
        # Get distances to all other models
        distances = self.clusterer.get_all_distances(current_model)
        
        if not distances:
            print(f"Warning: No alternative models found for {current_model}")
            return current_model
        
        # Calculate migration probabilities
        probabilities = self.calculate_migration_probability(distances, temperature)
        
        # Sample from the probability distribution
        models = list(probabilities.keys())
        probs = list(probabilities.values())
        
        selected_model = np.random.choice(models, p=probs)
        
        return selected_model
    
    def migrate(
        self,
        prompt_chain: List[Tuple[str, ...]],
        temperature: float = None
    ) -> List[Tuple[str, ...]]:
        """
        Migrate models in the prompt chain based on migration ratio.
        
        Parameters:
        - prompt_chain: List of tuples [(model_name, prompt_part1, prompt_part2, ...), ...]
        - temperature: Temperature for migration (uses default if None)
        
        Returns:
        - new_prompt_chain: The prompt chain after migration
        """
        if temperature is None:
            temperature = self.default_temperature
        
        # Validate temperature
        if temperature <= 0 or temperature > 1:
            print(f"Warning: Invalid temperature {temperature}. Using default {self.default_temperature}")
            temperature = self.default_temperature
        
        if not prompt_chain:
            print("Warning: Empty prompt chain provided")
            return prompt_chain
        
        # Determine which prompts to migrate
        n_prompts = len(prompt_chain)
        n_to_migrate = int(np.ceil(n_prompts * self.migration_ratio))
        
        # Randomly select indices to migrate
        indices_to_migrate = random.sample(range(n_prompts), n_to_migrate)
        
        print(f"\nMigrating {n_to_migrate}/{n_prompts} prompts (ratio={self.migration_ratio})")
        print(f"Indices to migrate: {sorted(indices_to_migrate)}")
        
        # Create new prompt chain
        new_prompt_chain = []
        
        for i, prompt_tuple in enumerate(prompt_chain):
            if i in indices_to_migrate:
                current_model = prompt_tuple[0]
                new_model = self.select_migration_target(current_model, temperature)
                
                # Replace model name, keep rest of the tuple
                new_tuple = (new_model,) + prompt_tuple[1:]
                new_prompt_chain.append(new_tuple)
                
                print(f"  Step {i}: {current_model} → {new_model}")
            else:
                new_prompt_chain.append(prompt_tuple)
                print(f"  Step {i}: {prompt_tuple[0]} (unchanged)")
        
        return new_prompt_chain


if __name__ == "__main__":
    # Example usage
    migration_system = ModelBasedTopology(
        migration_ratio=0.5,
        default_temperature=0.3,
        clusterer=ModelClusterer()
    )
    
    # Sample prompt chain
    sample_prompt_chain = [
        ("qwen3:0.6b", "This is ", "input prompt 1"),
        ("gemma3:270m", "This is ", "input prompt 2"),
        ("qwen2.5:0.5b", "This is ", "input prompt 3"),
        ("smollm2:360m", "This is ", "input prompt 4")
    ]
    
    print("\n" + "="*60)
    print("ORIGINAL PROMPT CHAIN:")
    print("="*60)
    for i, prompt in enumerate(sample_prompt_chain):
        print(f"Step {i}: {prompt}")
    
    # Perform migration with default temperature
    print("\n" + "="*60)
    print("MIGRATION WITH DEFAULT TEMPERATURE (0.3):")
    print("="*60)
    migrated_chain_1 = migration_system.migrate(sample_prompt_chain)
    
    # Perform migration with higher temperature (more adventurous)
    print("\n" + "="*60)
    print("MIGRATION WITH HIGH TEMPERATURE (0.9 - more adventurous):")
    print("="*60)
    migrated_chain_2 = migration_system.migrate(sample_prompt_chain, temperature=0.9)
    
    # Perform migration with low temperature (conservative)
    print("\n" + "="*60)
    print("MIGRATION WITH LOW TEMPERATURE (0.1 - conservative):")
    print("="*60)
    migrated_chain_3 = migration_system.migrate(sample_prompt_chain, temperature=0.1)