"""
Genetic_algorithm_processes/S3_mutation/prompt_chain_mutation.py
"""

from typing import Any
import random


from Genetic_algorithm_processes.S3_mutation.methods.synonym_mutation import SynonymMutation
from Genetic_algorithm_processes.S3_mutation.methods.shuffle_mutation import ShuffleMutation
from Genetic_algorithm_processes.S3_mutation.methods.delete_mutation import DeleteMutation

class PromptChainMutation:
    def __init__(self,
        mutation_chance: float = 0.2,
        mutation_methods: list[Any] = [
            SynonymMutation(mutation_rate=0.1).mutate,
            ShuffleMutation(N_segment_cut=3, shuffling_constant=1.0).mutate,
            DeleteMutation(min_segment_fraction=0.1, max_segment_fraction=0.3).mutate
        ]
    ):
        self.mutation_chance = mutation_chance
        self.mutation_methods = mutation_methods

    def mutate_population(self, prompt_chain_population):
        mutated_population = []
        for prompt_chain in prompt_chain_population:
            if random.random() < self.mutation_chance:
                mutation_func = random.choice(self.mutation_methods)
                mutated_prompt_chain = mutation_func(prompt_chain)
                mutated_population.append(mutated_prompt_chain)
            else:
                mutated_population.append(prompt_chain)
        return mutated_population
    
    def mutate_prompt_chain(self, prompt_chain):
        if random.random() >= self.mutation_chance:
            mutatation_func = random.choice(self.mutation_methods)
            return mutatation_func(prompt_chain)
        else:
            return prompt_chain

if __name__ == "__main__":
    offspring_population = [
        [("gpt-3.5-turbo", "This is an ", "input prompt"), ("gpt-4", "This ", "another ", "input ", "prompt")],
        [("gpt-4", "Different input prompt here"), ("gpt-3.5-turbo", "", "Yet another prompt input")],
        [("gpt-3.5-turbo", "Sample prompt one"), ("gpt-4", "Sample prompt two"), ("gpt-3.5-turbo", "Sample prompt three")],
        [("gpt-4", "First prompt segment"), ("gpt-4", "Second prompt segment")],
        [("gpt-3.5-turbo", "Only one prompt in this chain")]
    ] 
    mutation_instance = PromptChainMutation(mutation_chance=0.1)
    mutated_population = mutation_instance.mutate_population(offspring_population)
    print(f"Mutated Prompt Chain Population: {mutated_population}")