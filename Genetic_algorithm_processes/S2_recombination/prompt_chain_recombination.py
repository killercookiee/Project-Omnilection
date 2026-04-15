# Important notes:
# 

"""
Genetic_algorithm_processes/S2_recombination/prompt_chain_recombination.py
"""


from Genetic_algorithm_processes.S2_recombination.methods.pairing._random_pairing import RandomPairing
from Genetic_algorithm_processes.S2_recombination.methods.crossover.N_crossover_multiparent import NCrossoverMultiparent
from Genetic_algorithm_processes.S2_recombination.methods.crossover._model_based_crossover import ModelBasedCrossover


class PromptChainRecombination:
    def __init__(self,
        prompt_chain_pairing_instance = RandomPairing(),
        prompt_chain_crossover_instance = NCrossoverMultiparent(),
        prompt_crossover_instance = ModelBasedCrossover()
    ):
        self.prompt_chain_pairing_instance = prompt_chain_pairing_instance
        self.prompt_chain_crossover_instance = prompt_chain_crossover_instance
        self.prompt_crossover_instance = prompt_crossover_instance

        prompt_chain_crossover_instance.crossover_num = prompt_chain_pairing_instance.size_of_pairs - 1


    def recombine_prompt_chains(self, selected_prompt_chains):
        self.prompt_chain_pairing_instance.number_of_pairs = len(selected_prompt_chains)
        paired_prompt_chains = self.prompt_chain_pairing_instance.pair(selected_prompt_chains)

        offsprings = []
        for parents in paired_prompt_chains:
            offspring = self.prompt_chain_crossover_instance.crossover(parents)

            offspring_recombined = self.prompt_crossover_instance.crossover(parents, offspring)
            offsprings.append(offspring_recombined)
        return offsprings


if __name__ == "__main__":
    selected_prompt_chains = [
        [("gpt-3.5-turbo", "This is an ", "input prompt"), ("gpt-4", "This ", "another ", "input ", "prompt")],
        [("gpt-4", "Different input prompt here"), ("gpt-3.5-turbo", "", "Yet another prompt input")],
        [("gpt-3.5-turbo", "Sample prompt one"), ("gpt-4", "Sample prompt two"), ("gpt-3.5-turbo", "Sample prompt three")],
        [("gpt-4", "First prompt segment"), ("gpt-4", "Second prompt segment")],
        [("gpt-3.5-turbo", "Only one prompt in this chain")]
    ]    

    initial_input = "What is the capital of France?"
    solution_output = "Paris"
    population_cap = 10

    def run_prompt_chain(prompt_chain, initial_input):
        # Output: [(prompt_output, time_taken), ...]
        return [("The capital of France is Paris.", {'total duration': 1.0, 'load_duration': 0.1, 'prompt_eval_count': 10, 'prompt_eval_duration': 0.5, 'prompt_eval_rate': 20.0, 'eval_count': 5, 'eval_duration': 0.5, 'eval_rate': 10.0}),
                ("The capital of Germany is Berlin.", {'total duration': 3.0, 'load_duration': 0.2, 'prompt_eval_count': 15, 'prompt_eval_duration': 1.0, 'prompt_eval_rate': 15.0, 'eval_count': 7, 'eval_duration': 2.0, 'eval_rate': 3.5})]  # Return a mock output and time taken
    
    
    prompt_chain_recombination = PromptChainRecombination()
    prompt_chain_offsprings = prompt_chain_recombination.recombine_prompt_chains(selected_prompt_chains)
    print(f"Prompt Chain Offsprings: {prompt_chain_offsprings}")