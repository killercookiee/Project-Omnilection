from Genetic_algorithm_processes.S1_selection.prompt_chain_selection import PromptChainSelection
from Genetic_algorithm_processes.S2_recombination.methods.pairing.random_pairing import RandomPairing
from Genetic_algorithm_processes.S2_recombination.methods.crossover.N_crossover_multiparent import NCrossoverMultiparent
from Genetic_algorithm_processes.S2_recombination.methods.crossover.total_parent_crossover import TotalParentCrossover


class PromptChainRecombination:
    def __init__(self, run_prompt_chain_function, prompt_chain_selection_instance=PromptChainSelection(),
                 prompt_chain_pairing_instance=RandomPairing(size_of_pairs=2, number_of_pairs=2),
                 prompt_chain_crossover_instance=NCrossoverMultiparent(crossover_num=2), prompt_crossover_instance=TotalParentCrossover()):
        self.run_prompt_chain_function = run_prompt_chain_function
        self.prompt_chain_selection_instance = prompt_chain_selection_instance
        self.prompt_chain_pairing_instance = prompt_chain_pairing_instance
        self.prompt_chain_crossover_instance = prompt_chain_crossover_instance
        self.prompt_crossover_instance = prompt_crossover_instance


    def recombine_prompt_chains(self, prompt_chain_population):
        selected_prompt_chains = self.prompt_chain_selection_instance.select_prompt_chain(prompt_chain_population, self.run_prompt_chain_function)
        paired_prompt_chains = self.prompt_chain_pairing_instance.random_pairing(selected_prompt_chains)

        offsprings = []
        for parents in paired_prompt_chains:
            offspring = self.prompt_chain_crossover_instance.crossover(parents)

            offspring_recombined = self.prompt_crossover_instance.crossover(parents, offspring)
            offsprings.append(offspring_recombined)
        return offsprings


if __name__ == "__main__":
    prompt_chain_population = [
        [("gpt-3.5-turbo", "This is an ", "input prompt"), ("gpt-4", "This ", "another ", "input ", "prompt")],
        [("gpt-4", "Different input prompt here"), ("gpt-3.5-turbo", "", "Yet another prompt input")],
        [("gpt-3.5-turbo", "Sample prompt one"), ("gpt-4", "Sample prompt two"), ("gpt-3.5-turbo", "Sample prompt three")],
        [("gpt-4", "First prompt segment"), ("gpt-4", "Second prompt segment")],
        [("gpt-3.5-turbo", "Only one prompt in this chain")]
    ]    
    def run_prompt_chain(prompt_chain):
        # Output: [(prompt_output, time_taken), ...]
        return [("The capital of France is Paris.", 2.5), ("The capital of Germany is Berlin.", 3.0)]
    prompt_chain_recombination = PromptChainRecombination(run_prompt_chain)
    prompt_chain_offsprings = prompt_chain_recombination.recombine_prompt_chains(prompt_chain_population)
    print(f"Prompt Chain Offsprings: {prompt_chain_offsprings}")