# Input a population of individuals (prompt chains) as a list
# Output a list of selected individuals based on prompt chain selection

from methods.fitness.fitness_function import FitnessFunction
from methods.selection.stochastic_universal_sampling import StochasticUniversalSampling
class PromptChainSelection:
    def __init__(self, population, run_prompt_chain_function,
                 selection_algorithm=StochasticUniversalSampling(selection_ratio=0.5),
                 fitness_function=FitnessFunction(accuracy_weight=1.0, model_cost_weight=0.5, speed_weight=1.0, token_limit_weight=1.0)):
        
        self.population = population
        self.run_prompt_chain = run_prompt_chain_function
        self.selection_algorithm = selection_algorithm
        self.fitness_function = fitness_function

        self.population_scores = {}
    
    def select_prompt_chain(self):
        # Fitness calculation
        for individual in self.population:
            prompt_output_chain = self.run_prompt_chain(individual)
            total_elapsed_time = sum([time_taken for _, time_taken in prompt_output_chain])
            final_output = prompt_output_chain[-1][0]
            fitness_score = self.fitness_function.get_fitness_score(individual, final_output, total_elapsed_time)
            self.population_scores[tuple(map(tuple, individual))] = fitness_score

        # Selection
        selected = self.selection_algorithm.stochastic_universal_sampling(self.population_scores)
        return selected

if __name__ == "__main__":
    population = [[("gpt-3.5-turbo", "This is an input prompt"), ("gpt-3", "This another input prompt")],
                  [("gpt-4", "Different input prompt here"), ("gpt-3.5-turbo", "Yet another prompt input")]]
    def run_prompt_chain(prompt_chain):
        # Output: [(prompt_output, time_taken), ...]
        return [("The capital of France is Paris.", 2.5), ("The capital of Germany is Berlin.", 3.0)]

    prompt_chain_selection = PromptChainSelection(population, run_prompt_chain)
    selected = prompt_chain_selection.select_prompt_chain()
    print(f"Selected Prompt Chains: {selected}")