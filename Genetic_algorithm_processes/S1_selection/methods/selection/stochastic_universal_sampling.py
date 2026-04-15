# Input a dictionary of individuals with their fitness scores
# Output a list of selected individuals based on stochastic universal sampling


"""
Genetic_algorithm_processes/S1_selection/methods/selection/stochastic_universal_sampling.py
"""

import random


class StochasticUniversalSampling:
    def __init__(self,
        selection_ratio: float = 0.5,
        verbose: bool = False
    ):
        self.selection_ratio = selection_ratio
        self.verbose = verbose

        if self.verbose:
            print(f"[SUS] Initialized — selection ratio: {self.selection_ratio}")

    def select(self, population_fitness) -> list[list[tuple]]:
        num_selections = max(1, int(len(population_fitness) * self.selection_ratio))
        total_fitness = sum(fitness for _, fitness in population_fitness)

        if total_fitness <= 0:
            print(f"[SUS] ❌ Total fitness is {total_fitness} — must be > 0")
            raise ValueError("Total fitness must be > 0 for SUS")

        pointer_distance = total_fitness / num_selections
        start_point = random.uniform(0, pointer_distance)
        pointers = [start_point + i * pointer_distance for i in range(num_selections)]

        if self.verbose:
            print(f"\n[SUS] Pool: {len(population_fitness)} | Selecting: {num_selections} | Total fitness: {total_fitness:.3f}")
            print(f"      Pointer distance: {pointer_distance:.3f} | Start point: {start_point:.3f}")
            for i, (_, fitness) in enumerate(population_fitness):
                pct = fitness / total_fitness * 100
                bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
                print(f"  Individual [{i+1}] fitness: {fitness:.3f}  [{bar}]  ({pct:.1f}% coverage)")

        selected_individuals = []
        cumulative_fitness = 0.0
        pop_index = 0
        current_individual, current_fitness = population_fitness[pop_index]

        for pointer in pointers:
            while cumulative_fitness + current_fitness < pointer:
                cumulative_fitness += current_fitness
                pop_index += 1
                current_individual, current_fitness = population_fitness[pop_index]
            selected_individuals.append(current_individual)

        if self.verbose:
            print(f"\n{'─'*40}")
            print(f"  ✅  SUS selection complete")
            print(f"      Selected : {len(selected_individuals)} / {len(population_fitness)} individuals")
            print(f"{'─'*40}\n")

        return selected_individuals


if __name__ == "__main__":
    sus_method = StochasticUniversalSampling(selection_ratio=0.5, verbose=True)

    prompt_chain_population = [
        [("gemma3:270m", "Summarize the task and give a hint of the answer. ", "Make it concise and clear."),
         ("qwen2.5-coder:0.5b", "Provide 1 word answer only")],

        [("smollm:360m", "Summarize the task. ", "Give a hint of the completely wrong answer."),
         ("qwen2:0.5b", "Give a wrong answer. ", "Explain why the right answer is correct and the wrong answer is wrong.")],

        [("deepseek-coder:latest", "Another different prompt.")]
    ]
    initial_input = "What is the capital of France?"
    solution_output = "Paris"

    population_fitness = [
        (prompt_chain_population[0], 50),
        (prompt_chain_population[1], 30),
        (prompt_chain_population[2], 20)
    ]

    selected = sus_method.select(population_fitness)
    print(f"Final — Selected {len(selected)} individuals")