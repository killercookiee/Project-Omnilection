# Input a dictionary of individuals with their fitness scores
# Output a list of selected individuals based on roulette wheel selection

"""
Genetic_algorithm_processes/S1_selection/methods/selection/roulette_wheel.py
"""

import random


class RouletteWheelSelection:
    def __init__(self,
        selection_ratio: float = 0.5,
        verbose: bool = False
    ):
        self.selection_ratio = selection_ratio
        self.verbose = verbose

        if self.verbose:
            print(f"[RouletteWheelSelection] Initialized — selection ratio: {self.selection_ratio}")

    def select(self, population_fitness: list[tuple]) -> list[list[tuple]]:
        num_selections = max(1, int(len(population_fitness) * self.selection_ratio))
        total_fitness = sum(fitness for _, fitness in population_fitness)
        selected_individuals = []

        if self.verbose:
            print(f"\n[RouletteWheelSelection] Pool: {len(population_fitness)} | Selecting: {num_selections} | Total fitness: {total_fitness:.3f}")
            for i, (_, fitness) in enumerate(population_fitness):
                pct = (fitness / total_fitness * 100) if total_fitness > 0 else 0
                bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
                print(f"  Individual [{i+1}] fitness: {fitness:.3f}  [{bar}]  ({pct:.1f}% selection chance)")

        for _ in range(num_selections):
            pick = random.uniform(0, total_fitness)
            current = 0.0
            for individual, fitness in population_fitness:
                current += fitness
                if current >= pick:
                    selected_individuals.append(individual)
                    break

        if self.verbose:
            print(f"\n{'─'*40}")
            print(f"  ✅  Selection complete")
            print(f"      Selected : {len(selected_individuals)} / {len(population_fitness)} individuals")
            print(f"{'─'*40}\n")

        return selected_individuals


if __name__ == "__main__":
    selector = RouletteWheelSelection(selection_ratio=0.5, verbose=True)

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

    selected = selector.select(population_fitness)
    print(f"Final — Selected {len(selected)} individuals")