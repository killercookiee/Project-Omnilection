# Input a population of individuals (prompt chains) as a list
# Output a list of selected individuals based on prompt chain selection

"""
Genetic_algorithm_processes/S1_selection/prompt_chain_selection.py
"""

from Genetic_algorithm_processes.ollama_run import PromptChainRunner
from Genetic_algorithm_processes.S1_selection.methods.fitness.fitness_function import FitnessCalculation
from Genetic_algorithm_processes.S1_selection.methods.selection.stochastic_universal_sampling import StochasticUniversalSampling
from Genetic_algorithm_processes.S1_selection.methods.selection.roulette_wheel import RouletteWheelSelection


class PromptChainSelection:
    def __init__(self,
        selection_algorithm=StochasticUniversalSampling(),
        fitness_algorithm=FitnessCalculation(),
        verbose: bool = False
    ):
        self.selection_algorithm = selection_algorithm
        self.fitness_algorithm = fitness_algorithm
        self.verbose = verbose

        if self.verbose:
            print(f"[PromptChainSelection] Initialized — algorithm: {type(self.selection_algorithm).__name__}")

    def evaluate_prompt_chains(self, current_population: list[list[tuple]], run_prompt_chain_function: callable, initial_input: str, solution_output: str) -> list[tuple[list[tuple], float]]:
        if self.verbose:
            print(f"[PromptChainSelection] Evaluating fitness for {len(current_population)} prompt chains...")

        population_fitness = self.fitness_algorithm.evaluate_population(current_population, run_prompt_chain_function, initial_input, solution_output)

        if self.verbose:
            scores = [score for _, score in population_fitness]
            for i, score in enumerate(scores):
                bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
                print(f"  Chain [{i+1}/{len(scores)}] fitness: {score:.3f}  [{bar}]")
            print(f"  Best: {max(scores):.3f}  |  Worst: {min(scores):.3f}  |  Avg: {sum(scores)/len(scores):.3f}")

        return population_fitness

    def select_prompt_chains(self, population_fitness: list[tuple[list[tuple], float]]) -> list[list[tuple]]:
        if self.verbose:
            print(f"\n[PromptChainSelection] Running {type(self.selection_algorithm).__name__} on {len(population_fitness)} individuals...")

        selected = self.selection_algorithm.select(population_fitness)

        if self.verbose:
            print(f"[PromptChainSelection] Selected {len(selected)} / {len(population_fitness)} individuals")

        return selected

    def evauluate_select_prompt_chains(self, current_population: list[list[tuple]], run_prompt_chain_function: callable, initial_input: str, solution_output: str) -> list[list[tuple]]:
        population_fitness = self.evaluate_prompt_chains(current_population, run_prompt_chain_function, initial_input, solution_output)
        selected = self.select_prompt_chains(population_fitness)

        if self.verbose:
            print(f"\n{'─'*40}")
            print(f"  ✅  Selection pipeline complete")
            print(f"      Input    : {len(current_population)} chains")
            print(f"      Selected : {len(selected)} chains")
            print(f"{'─'*40}\n")

        return selected, population_fitness


if __name__ == "__main__":
    prompt_chain_population = [
        [("gemma3:270m", "Summarize the task and give a hint of the answer. ", "Make it concise and clear."),
         ("qwen2.5-coder:0.5b", "Provide 1 word answer only")],

        [("smollm:360m", "Summarize the task. ", "Give a hint of the completely wrong answer."),
         ("qwen2:0.5b", "Give a wrong answer. ", "Explain why the right answer is correct and the wrong answer is wrong.")],

        [("deepseek-coder:latest", "Try to give a wrong but sort of right answer. ")]
    ]
    initial_input = "What is the capital of France?"
    solution_output = "Paris"

    runner = PromptChainRunner(verbose=True, timeout=60)

    # --------------------------------------------------
    # Example 1: Stochastic Universal Sampling (SUS)
    # --------------------------------------------------
    sus_selection = PromptChainSelection(
        selection_algorithm=StochasticUniversalSampling(selection_ratio=0.5),
        verbose=True
    )
    sus_selected, _ = sus_selection.evauluate_select_prompt_chains(
        prompt_chain_population, runner.run_prompt_chain, initial_input, solution_output
    )
    print(f"SUS — Selected {len(sus_selected)} chains\n")

    # --------------------------------------------------
    # Example 2: Roulette Wheel Selection
    # --------------------------------------------------
    roulette_selection = PromptChainSelection(
        selection_algorithm=RouletteWheelSelection(selection_ratio=0.5),
        verbose=True
    )
    roulette_selected, _ = roulette_selection.evauluate_select_prompt_chains(
        prompt_chain_population, runner.run_prompt_chain, initial_input, solution_output
    )
    print(f"Roulette — Selected {len(roulette_selected)} chains: \n {roulette_selected}\n")