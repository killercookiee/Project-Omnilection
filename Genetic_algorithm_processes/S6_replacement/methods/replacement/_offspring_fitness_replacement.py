# Replaces the current population with offspring based on fitness scores, keeping the best individuals up to the population cap.

"""
Genetic_algorithm_processes/S1_selection/methods/replacement/offspring_fitness_replacement.py
"""

class OffspringFitnessReplacement:
    def __init__(self,
        population_cap: int = 10,
        verbose: bool = False
    ):
        self.population_cap = population_cap
        self.verbose = verbose

        if self.verbose:
            print(f"[OffspringFitnessReplacement] Initialized — population cap: {self.population_cap}")

    def replace(self, offspring_population_fitness: list[tuple[list[tuple], float]], current_population_fitness: list[tuple[list[tuple], float]]) -> list[tuple]:
        if self.verbose:
            print(f"\n[OffspringFitnessReplacement] Evaluating {len(offspring_population_fitness)} offspring | Current population: {len(current_population_fitness)}")
        # Combine, sort, and trim to population cap
        combined = current_population_fitness + offspring_population_fitness
        combined.sort(key=lambda x: x[1], reverse=True)
        combined = combined[:self.population_cap]

        if self.verbose:
            dropped = len(current_population_fitness) + len(offspring_population_fitness) - len(combined)
            scores = [score for _, score in combined]
            print(f"\n{'─'*40}")
            print(f"  ✅  Replacement complete")
            print(f"      Kept   : {len(combined)} / {len(current_population_fitness) + len(offspring_population_fitness)} individuals ({dropped} dropped)")
            print(f"      Best   : {scores[0]:.3f}  |  Worst kept: {scores[-1]:.3f}  |  Avg: {sum(scores)/len(scores):.3f}")
            print(f"{'─'*40}\n")

        return combined


if __name__ == "__main__":
    offspring_population_fitness = [
        ([("qwen2:0.5b", "This is an ", "input prompt"), ("qwen:0.5b", "This ", "another ", "input ", "prompt")], 0.85),
        ([("smollm:135m", "Different input prompt here"), ("qwen3:0.6b", "", "Yet another prompt input")], 0.72)
    ]
        
    current_population_fitness = [
        ([("codellama:latest", "Sample prompt one"), ("smollm:135m", "Sample prompt two"), ("smollm:135m", "Sample prompt three")], 0.91),
        ([("smollm:135m", "First prompt segment"), ("smollm:135m", "Second prompt segment")], 0.68),
        ([("smollm:135m", "Only one prompt in this chain")], 0.42)
    ]

    replacement_instance = OffspringFitnessReplacement(population_cap=4, verbose=True)
    new_population = replacement_instance.replace(offspring_population_fitness, current_population_fitness)
    print(f"Final population size: {len(new_population)} | Top score: {new_population[0][1]:.3f}")