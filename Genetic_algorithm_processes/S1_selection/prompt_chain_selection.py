# Input a population of individuals (prompt chains) as a list
# Output a list of selected individuals based on prompt chain selection

from Genetic_algorithm_processes.S1_selection.methods.fitness.fitness_function import FitnessFunction
from Genetic_algorithm_processes.S1_selection.methods.selection.stochastic_universal_sampling import StochasticUniversalSampling
class PromptChainSelection:
    def __init__(self,
                 selection_algorithm=StochasticUniversalSampling(selection_ratio=0.5) ,
                 fitness_function=FitnessFunction(accuracy_weight=1.0, speed_weight=1.0, token_limit_weight=1.0)):        
        self.selection_algorithm = selection_algorithm
        self.fitness_function = fitness_function

        self.population_scores = {}
    
    def select_prompt_chain(self, population, run_prompt_chain_function):
        # Fitness calculation
        for prompt_chain in population:
            prompt_output_chain = run_prompt_chain_function(prompt_chain)
            total_elapsed_time = sum([time_taken for _, time_taken in prompt_output_chain])
            final_output = prompt_output_chain[-1][0]
            fitness_score = self.fitness_function.get_fitness_score(prompt_chain, final_output, total_elapsed_time)
            self.population_scores[tuple(map(tuple, prompt_chain))] = fitness_score

        # Selection
        selected = self.selection_algorithm.stochastic_universal_sampling(self.population_scores)
        return selected

if __name__ == "__main__":
    prompt_chain_population = [
        [("gpt-3.5-turbo", "This is an ", "input prompt"), 
         ("gpt-4", "This ", "another ", "input ", "prompt")],
        [("gpt-4", "Different input prompt here"), 
         ("gpt-3.5-turbo", "", "Yet another prompt input")]
    ]
    
    def run_prompt_chain(prompt_chain):
        # Simulate different outputs based on the prompt chain
        # In real implementation, this would actually run the prompts
        if len(prompt_chain) == 2 and prompt_chain[0][0] == "gpt-3.5-turbo":
            return [("The capital of France is Paris.", 2.5), 
                    ("The capital of Germany is Berlin.", 3.0)]
        else:
            return [("Different output here.", 1.5), 
                    ("Another different output.", 2.0)]

    prompt_chain_selection = PromptChainSelection()
    selected = prompt_chain_selection.select_prompt_chain(prompt_chain_population, run_prompt_chain)
    print(f"Selected Prompt Chains: {selected}")